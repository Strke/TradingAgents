"""Unit tests for report formatting and chunking (jiying/report.py)."""

import pytest

from jiying.report import (
    AnalysisOutcome,
    build_headline,
    build_sections,
    format_report,
    rating_label,
)

pytestmark = pytest.mark.unit


def _outcome(signal: str = "Buy", **state) -> AnalysisOutcome:
    return AnalysisOutcome(
        ticker="NVDA",
        trade_date="2026-09-12",
        asset_type="stock",
        signal=signal,
        final_state=state,
    )


class TestRatingLabel:
    def test_known_ratings(self):
        assert rating_label("Buy") == "买入"
        assert rating_label("Sell") == "卖出"

    def test_review(self):
        assert rating_label("REVIEW") == "待复核"


class TestHeadline:
    def test_contains_key_fields(self):
        headline = build_headline(_outcome("Hold"))
        assert "NVDA" in headline
        assert "2026-09-12" in headline
        assert "Hold" in headline
        assert "持有" in headline

    def test_review_carries_warning(self):
        headline = build_headline(_outcome("REVIEW"))
        assert "待复核" in headline
        assert "复核" in headline


class TestBuildSections:
    def test_decision_first_then_analysts_then_debates(self):
        sections = build_sections(
            _outcome(
                final_trade_decision="FINAL",
                investment_plan="PLAN",
                market_report="MARKET",
                sentiment_report="SENT",
                investment_debate_state={"history": "DEBATE"},
            )
        )
        titles = [section.splitlines()[0] for section in sections]
        assert titles == [
            "## 最终决策（组合经理）",
            "## 投资计划",
            "## 市场分析师报告",
            "## 情绪分析师报告",
            "## 多空研究员辩论",
        ]

    def test_empty_and_missing_sections_omitted(self):
        sections = build_sections(_outcome(market_report="", news_report=None))
        assert sections == []

    def test_debate_deduplicates_role_histories(self):
        sections = build_sections(
            _outcome(
                risk_debate_state={
                    "history": "ALL",
                    "aggressive_history": "ALL",
                    "judge_decision": "JUDGE",
                }
            )
        )
        assert len(sections) == 1
        assert "ALL" in sections[0]
        assert "**裁决**: JUDGE" in sections[0]
        assert sections[0].count("ALL") == 1

    def test_non_dict_debate_ignored(self):
        assert build_sections(_outcome(investment_debate_state="oops")) == []


class TestFormatReport:
    def test_small_report_is_single_message(self):
        messages = format_report(
            _outcome("Buy", final_trade_decision="短期看多"),
            chunk_chars=3500,
            max_messages=12,
        )
        assert len(messages) == 1
        assert "(1/1)" in messages[0]
        assert "TradingAgents 分析报告" in messages[0]
        assert "短期看多" in messages[0]

    def test_chunking_respects_limit(self):
        state = {
            "final_trade_decision": "DECISION",
            "market_report": "M" * 5000,
            "news_report": "N" * 5000,
        }
        messages = format_report(_outcome(**state), chunk_chars=1000, max_messages=30)
        assert len(messages) > 1
        assert all(len(message) <= 1400 for message in messages)  # body + prefix
        # Sequential numbering
        assert f"(1/{len(messages)})" in messages[0]
        assert f"({len(messages)}/{len(messages)})" in messages[-1]

    def test_headline_always_first(self):
        messages = format_report(
            _outcome("Sell", market_report="S" * 9000),
            chunk_chars=1000,
            max_messages=30,
        )
        assert "TradingAgents 分析报告" in messages[0]
        assert "Sell" in messages[0]

    def test_overflow_caps_messages(self):
        state = {"market_report": "M" * 50_000}
        messages = format_report(_outcome(**state), chunk_chars=1000, max_messages=4)
        assert len(messages) == 4
        assert "截断" in messages[-1]

    def test_invalid_limits_raise(self):
        with pytest.raises(ValueError):
            format_report(_outcome(), chunk_chars=0, max_messages=1)
        with pytest.raises(ValueError):
            format_report(_outcome(), chunk_chars=100, max_messages=0)
