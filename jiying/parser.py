"""Natural-language intent parsing for inbound JiYing messages.

A user message like ``帮我分析一下英伟达最近的走势`` or
``NVDA 2025-09-12 crypto`` is turned into a structured
:class:`AnalysisRequest` (ticker / trade date / asset type) using the same
LLM provider configured for TradingAgents (quick-think model, so the parsing
round-trip stays cheap). Responses follow the exact output contract of the
platform; failures degrade to a help message instead of a guess.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import re
from dataclasses import dataclass

from tradingagents.dataflows.symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

ASSET_TYPES = ("stock", "crypto")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-:=^]{1,20}$")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

HELP_TEXT = """**TradingAgents 使用说明**

把想分析的标的发给我即可，支持自然语言，例如：

- `分析一下英伟达 2025-09-12 的走势`
- `帮我看看 BTC-USD 最近怎么样（crypto）`
- `NVDA 2025-06-30`
- `分析一下贵州茅台`（A 股代码自动补交易所后缀）
- `/status` 查看服务配置与队列状态

规则：

| 参数 | 说明 |
| --- | --- |
| 标的 | 美股代码、A 股 6 位代码（自动补 `.SS`/`.SZ`）、或加密货币交易对（如 `NVDA`、`600519`、`BTC-USD`） |
| 交易日 | 可选，格式 `YYYY-MM-DD`，缺省为今天 |
| 资产类型 | 可选，`stock`（默认）或 `crypto` |

注意：韩国等同样使用 6 位数字代码的市场需显式写后缀（如 `005930.KS`）。
单次分析通常需要数分钟，期间会持续同步进度；完成后返回完整分析报告。
支持的分析师：市场 / 新闻 / 基本面 / 情绪，输出含多空辩论与最终评级。
"""

# Deterministic fast-path: "TICKER [YYYY-MM-DD] [stock|crypto]"
_FAST_PATH_RE = re.compile(
    r"^\s*([A-Za-z0-9.\-:=^]{1,20})"
    r"(?:\s+(\d{4}-\d{2}-\d{2}))?"
    r"(?:\s+(stock|crypto))?\s*[。.!]?\s*$",
    re.IGNORECASE,
)

_PARSE_PROMPT = """You parse user chat messages for a stock-analysis bot.
Extract the analysis target and reply with ONLY a JSON object, no prose, no
markdown fence, using exactly these keys:

{{"ticker": "<symbol, e.g. NVDA, AAPL, BTC-USD; empty string if absent>",
  "trade_date": "<YYYY-MM-DD if the user names a date, otherwise empty string>",
  "asset_type": "<stock|crypto, empty string if unspecified>"}}

Rules:
- Resolve company names to ticker symbols (英伟达 -> NVDA, 苹果 -> AAPL,
  特斯拉 -> TSLA, 比特币 -> BTC-USD).
- Chinese A-share companies resolve to their bare 6-digit board code WITHOUT
  any exchange suffix (洲际油气 -> 600759, 贵州茅台 -> 600519, 平安银行 ->
  000001, 宁德时代 -> 300750). The service appends .SS/.SZ automatically.
  Other non-US markets keep their suffixed symbol (三星电子 -> 005930.KS).
- Resolve relative dates ("今天", "明天", "latest") into YYYY-MM-DD where
  unambiguous; otherwise leave trade_date empty.
- asset_type is "crypto" only for cryptocurrency pairs or explicit crypto
  wording; everything else defaults to stock.

Current date: {today}

User message:
{message}"""


@dataclass(frozen=True)
class AnalysisRequest:
    """A validated analysis target."""

    ticker: str
    trade_date: str
    asset_type: str = "stock"

    def __post_init__(self) -> None:
        if not self.ticker:
            raise ValueError("ticker must not be empty")
        if self.asset_type not in ASSET_TYPES:
            raise ValueError(f"asset_type must be one of {ASSET_TYPES}")
        if not _DATE_RE.match(self.trade_date):
            raise ValueError(f"trade_date must be YYYY-MM-DD, got {self.trade_date!r}")

    @property
    def label(self) -> str:
        return f"{self.ticker} @ {self.trade_date} ({self.asset_type})"


class ParseError(ValueError):
    """Raised when a message cannot be mapped to an analysis request."""


def is_help_command(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {"/help", "/start", "help", "帮助", "使用说明"}


def is_status_command(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {"/status", "状态", "服务状态"}


def _normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not ticker:
        raise ParseError("empty ticker")
    if not _TICKER_RE.match(ticker):
        raise ParseError(f"invalid ticker: {raw!r}")
    # Shared data-layer normalization: appends the A-share exchange suffix
    # (600759 -> 600759.SS) and resolves crypto/forex forms; already-canonical
    # symbols pass through unchanged.
    return normalize_symbol(ticker)


def _validate_date(raw: str, today: _dt.date) -> str:
    if not raw:
        return today.isoformat()
    if not _DATE_RE.match(raw):
        raise ParseError(f"invalid trade_date: {raw!r}")
    try:
        _dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise ParseError(f"invalid trade_date: {raw!r}") from exc
    return raw


def parse_fast_path(text: str, today: _dt.date) -> AnalysisRequest | None:
    """Parse the unambiguous ``TICKER [date] [asset]`` form without an LLM."""
    match = _FAST_PATH_RE.match(text)
    if match is None:
        return None
    ticker, trade_date, asset_type = match.group(1), match.group(2), match.group(3)
    if _DATE_RE.match(ticker):
        # A bare date is not a ticker ("2026-09-01" alone).
        return None
    return AnalysisRequest(
        ticker=_normalize_ticker(ticker),
        trade_date=_validate_date(trade_date or "", today),
        asset_type=(asset_type or "stock").lower(),
    )


def extract_json_object(content: str) -> dict | None:
    """Pull the first JSON object out of an LLM response (fences tolerated)."""
    if not content:
        return None
    for candidate in _FENCE_RE.findall(content) + [content]:
        match = _JSON_BLOCK_RE.search(candidate)
        if match is None:
            continue
        try:
            data = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def request_from_payload(payload: dict, today: _dt.date) -> AnalysisRequest | None:
    """Validate an LLM-extracted payload; ``None`` means "no target found"."""
    if not payload:
        return None
    ticker_raw = str(payload.get("ticker") or "").strip()
    if not ticker_raw:
        return None
    return AnalysisRequest(
        ticker=_normalize_ticker(ticker_raw),
        trade_date=_validate_date(str(payload.get("trade_date") or "").strip(), today),
        asset_type=str(payload.get("asset_type") or "stock").strip().lower() or "stock",
    )


async def parse_request(
    text: str,
    llm,
    *,
    today: _dt.date | None = None,
) -> AnalysisRequest | None:
    """Turn a user message into an :class:`AnalysisRequest`.

    Returns ``None`` when the message contains no analysis target (caller
    should reply with :data:`HELP_TEXT`). Raises :class:`ParseError` when the
    LLM response is present but malformed.
    """
    if today is None:
        today = _dt.date.today()

    if is_help_command(text):
        return None

    fast = parse_fast_path(text, today)
    if fast is not None:
        return fast

    prompt = _PARSE_PROMPT.format(today=today.isoformat(), message=text.strip())
    # LangChain LLM ``invoke`` is blocking; keep it off the event loop.
    response = await asyncio.to_thread(llm.invoke, prompt)
    raw = _response_text(response)
    logger.debug("LLM parse response: %s", raw)
    payload = extract_json_object(raw)
    if payload is None:
        raise ParseError(f"LLM returned no JSON object (raw: {raw[:200]!r})")
    request = request_from_payload(payload, today)
    if request is None:
        logger.warning(
            "LLM response contained no analysis target (raw: %r)", raw[:200]
        )
    return request


def _response_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        # Normalized like tradingagents.llm_clients.base_client.normalize_content
        parts = [
            item.get("text", "")
            if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str)
            else ""
            for item in content
        ]
        return "\n".join(part for part in parts if part)
    return str(content) if content is not None else ""
