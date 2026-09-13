"""Format a completed TradingAgents run into JiYing markdown messages.

The full report (final decision, plans, analyst reports, debate records) is
split into a bounded sequence of markdown chunks so each stays comfortably
below the platform's 1 MiB per-frame limit while remaining readable in chat.
The first message always carries the headline: ticker, date and 5-tier rating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tradingagents.agents.utils.rating import RATING_REVIEW, is_review

logger = logging.getLogger(__name__)

_RATING_LABELS = {
    "Buy": "买入",
    "Overweight": "增持",
    "Hold": "持有",
    "Underweight": "减持",
    "Sell": "卖出",
    RATING_REVIEW: "待复核",
}


@dataclass(frozen=True)
class AnalysisOutcome:
    """Everything needed to render the report for one finished run."""

    ticker: str
    trade_date: str
    asset_type: str
    signal: str
    final_state: dict[str, Any]


def rating_label(signal: str) -> str:
    if is_review(signal):
        return _RATING_LABELS[RATING_REVIEW]
    return _RATING_LABELS.get(signal, signal)


def _section(title: str, content: Any) -> str | None:
    if not isinstance(content, str) or not content.strip():
        return None
    return f"## {title}\n\n{content.strip()}"


def _debate_section(title: str, state: Any, roles: list[str]) -> str | None:
    if not isinstance(state, dict):
        return None
    parts: list[str] = []
    history = state.get("history")
    if isinstance(history, str) and history.strip():
        parts.append(history.strip())
    for role in roles:
        value = state.get(role)
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(f"**{role}**: {value.strip()}")
    judge = state.get("judge_decision")
    if isinstance(judge, str) and judge.strip():
        parts.append(f"**裁决**: {judge.strip()}")
    if not parts:
        return None
    return f"## {title}\n\n" + "\n\n".join(parts)


def build_sections(outcome: AnalysisOutcome) -> list[str]:
    """Ordered markdown body sections (without the headline)."""
    state = outcome.final_state or {}
    sections: list[str] = []

    for title, key in (
        ("最终决策（组合经理）", "final_trade_decision"),
        ("投资计划", "investment_plan"),
        ("交易员计划", "trader_investment_plan"),
    ):
        rendered = _section(title, state.get(key))
        if rendered:
            sections.append(rendered)

    for title, key in (
        ("市场分析师报告", "market_report"),
        ("新闻研究员报告", "news_report"),
        ("基本面研究员报告", "fundamentals_report"),
        ("情绪分析师报告", "sentiment_report"),
    ):
        rendered = _section(title, state.get(key))
        if rendered:
            sections.append(rendered)

    rendered = _debate_section(
        "多空研究员辩论", state.get("investment_debate_state"),
        ["bull_history", "bear_history"],
    )
    if rendered:
        sections.append(rendered)

    rendered = _debate_section(
        "风险评估辩论", state.get("risk_debate_state"),
        ["aggressive_history", "conservative_history", "neutral_history"],
    )
    if rendered:
        sections.append(rendered)

    return sections


def build_headline(outcome: AnalysisOutcome) -> str:
    signal_line = f"**最终评级**: {outcome.signal}（{rating_label(outcome.signal)}）"
    if is_review(outcome.signal):
        signal_line += "\n\n> 注意：未能从决策文本中解析出有效评级，建议人工复核或重跑。"
    return (
        f"# TradingAgents 分析报告\n\n"
        f"**标的**: {outcome.ticker}（{outcome.asset_type}）\n\n"
        f"**交易日**: {outcome.trade_date}\n\n"
        f"{signal_line}"
    )


def _split_chunk(text: str, limit: int) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` characters.

    Prefers paragraph boundaries, then line boundaries, then hard cuts.
    """
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = limit
        pieces.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        pieces.append(remaining)
    return pieces


def format_report(outcome: AnalysisOutcome, *, chunk_chars: int, max_messages: int) -> list[str]:
    """Render the outcome as a list of markdown message bodies.

    Headline + sections are joined into one document and split on
    paragraph boundaries, so a short report stays a single message while
    the headline always opens the first chunk. ``(i/n)`` numbering is
    prefixed to every message once splitting occurs. Raises ``ValueError``
    if the limits are invalid; overflow trims trailing content.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if max_messages <= 0:
        raise ValueError("max_messages must be positive")

    document = "\n\n".join([build_headline(outcome), *build_sections(outcome)])
    chunks = _split_chunk(document, chunk_chars)

    if len(chunks) > max_messages:
        logger.warning(
            "Report for %s needs %d chunks, capping at %d",
            outcome.ticker, len(chunks), max_messages,
        )
        chunks = chunks[:max_messages]
        chunks[-1] += "\n\n---\n*报告过长，后续章节已被截断。*"

    return [
        f"({index}/{len(chunks)})\n\n{chunk}"
        for index, chunk in enumerate(chunks, start=1)
    ]
