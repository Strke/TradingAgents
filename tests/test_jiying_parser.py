"""Unit tests for natural-language parsing (jiying/parser.py)."""

import asyncio
import datetime as _dt

import pytest

from jiying.parser import (
    HELP_TEXT,
    AnalysisRequest,
    ParseError,
    extract_json_object,
    is_help_command,
    is_status_command,
    parse_fast_path,
    parse_request,
    request_from_payload,
)

pytestmark = pytest.mark.unit

TODAY = _dt.date(2026, 9, 13)


class _FakeLLM:
    """Mimics the LangChain LLM surface used by parse_request."""

    def __init__(self, content: str):
        self.content = content
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        return type("Response", (), {"content": self.content})()


class TestHelpCommand:
    def test_help_variants(self):
        assert is_help_command("/help")
        assert is_help_command(" /START ")
        assert is_help_command("帮助")
        assert is_help_command("使用说明")

    def test_non_help(self):
        assert not is_help_command("分析 NVDA")
        assert not is_help_command("/analyze NVDA")

    def test_status_command_variants(self):
        assert is_status_command("/status")
        assert is_status_command(" 状态 ")
        assert is_status_command("服务状态")
        assert not is_status_command("/help")
        assert not is_status_command("分析 NVDA")


class TestFastPath:
    def test_plain_ticker(self):
        request = parse_fast_path("NVDA", TODAY)
        assert request == AnalysisRequest("NVDA", TODAY.isoformat(), "stock")

    def test_ticker_with_date_and_crypto(self):
        request = parse_fast_path("BTC-USD 2026-09-01 crypto", TODAY)
        assert request == AnalysisRequest("BTC-USD", "2026-09-01", "crypto")

    def test_ticker_lowercased(self):
        request = parse_fast_path("aapl", TODAY)
        assert request.ticker == "AAPL"

    def test_natural_language_not_matched(self):
        assert parse_fast_path("帮我分析一下英伟达", TODAY) is None

    def test_date_only_rejected(self):
        # A bare date is not a ticker.
        assert parse_fast_path("2026-09-01", TODAY) is None

    def test_ashare_code_gets_exchange_suffix(self):
        # Bare 6-digit A-share codes normalize through the shared data layer.
        assert parse_fast_path("600759", TODAY).ticker == "600759.SS"
        assert parse_fast_path("600759 2026-09-11", TODAY).ticker == "600759.SS"
        assert parse_fast_path("000001", TODAY).ticker == "000001.SZ"
        assert parse_fast_path("300750", TODAY).ticker == "300750.SZ"

    def test_already_suffixed_symbols_untouched(self):
        assert parse_fast_path("600519.SS", TODAY).ticker == "600519.SS"
        assert parse_fast_path("005930.KS", TODAY).ticker == "005930.KS"
        assert parse_fast_path("NVDA", TODAY).ticker == "NVDA"
        assert parse_fast_path("BTC-USD", TODAY).ticker == "BTC-USD"


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json_object('{"ticker": "NVDA"}') == {"ticker": "NVDA"}

    def test_json_in_fence(self):
        content = '好的，结果如下：\n```json\n{"ticker": "AAPL"}\n```\n完毕'
        assert extract_json_object(content) == {"ticker": "AAPL"}

    def test_json_with_surrounding_prose(self):
        content = '解析结果 {"ticker": "TSLA", "trade_date": ""} 请查收'
        assert extract_json_object(content)["ticker"] == "TSLA"

    def test_no_json_returns_none(self):
        assert extract_json_object("抱歉，我没看懂。") is None
        assert extract_json_object("") is None

    def test_non_dict_json_returns_none(self):
        assert extract_json_object('["a"]') is None


class TestRequestFromPayload:
    def test_defaults(self):
        request = request_from_payload({"ticker": "nvda"}, TODAY)
        assert request == AnalysisRequest("NVDA", TODAY.isoformat(), "stock")

    def test_crypto_and_date(self):
        request = request_from_payload(
            {"ticker": "BTC-USD", "trade_date": "2026-01-02", "asset_type": "crypto"},
            TODAY,
        )
        assert request.asset_type == "crypto"
        assert request.trade_date == "2026-01-02"

    def test_empty_ticker_returns_none(self):
        assert request_from_payload({"ticker": ""}, TODAY) is None
        assert request_from_payload({}, TODAY) is None

    def test_bad_date_raises(self):
        with pytest.raises(ParseError):
            request_from_payload({"ticker": "NVDA", "trade_date": "09/01"}, TODAY)

    def test_bad_asset_type_raises(self):
        with pytest.raises(ValueError, match="asset_type"):
            request_from_payload({"ticker": "NVDA", "asset_type": "forex"}, TODAY)

    def test_ashare_ticker_from_llm_gets_suffix(self):
        request = request_from_payload({"ticker": "600759"}, TODAY)
        assert request.ticker == "600759.SS"


class TestParseRequest:
    def test_natural_language_via_llm(self):
        llm = _FakeLLM('{"ticker": "NVDA", "trade_date": "", "asset_type": ""}')
        request = asyncio.run(parse_request("帮我分析一下英伟达", llm, today=TODAY))
        assert request == AnalysisRequest("NVDA", TODAY.isoformat(), "stock")
        # The prompt carries the current date for relative-date resolution.
        assert TODAY.isoformat() in llm.calls[0]

    def test_fenced_llm_response(self):
        llm = _FakeLLM(
            "```json\n"
            '{"ticker": "BTC-USD", "trade_date": "2026-09-10", "asset_type": "crypto"}\n'
            "```"
        )
        request = asyncio.run(parse_request("看看比特币", llm, today=TODAY))
        assert request == AnalysisRequest("BTC-USD", "2026-09-10", "crypto")

    def test_help_returns_none_without_llm_call(self):
        llm = _FakeLLM("{}")
        assert asyncio.run(parse_request("/help", llm, today=TODAY)) is None
        assert llm.calls == []

    def test_no_target_returns_none(self):
        llm = _FakeLLM('{"ticker": "", "trade_date": "", "asset_type": ""}')
        assert asyncio.run(parse_request("你好呀", llm, today=TODAY)) is None

    def test_malformed_llm_output_raises(self):
        llm = _FakeLLM("抱歉，无法解析")
        with pytest.raises(ParseError, match="no JSON"):
            asyncio.run(parse_request("分析一下", llm, today=TODAY))

    def test_fast_path_skips_llm(self):
        llm = _FakeLLM('{"ticker": "WRONG"}')
        request = asyncio.run(parse_request("MSFT 2026-08-01", llm, today=TODAY))
        assert request.ticker == "MSFT"
        assert llm.calls == []


class TestHelpText:
    def test_help_text_mentions_key_rules(self):
        assert "YYYY-MM-DD" in HELP_TEXT
        assert "crypto" in HELP_TEXT
        # A-share guidance and the /status command are documented.
        assert ".SS" in HELP_TEXT
        assert "/status" in HELP_TEXT
        assert "005930.KS" in HELP_TEXT
