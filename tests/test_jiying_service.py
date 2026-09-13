"""Unit tests for the bridge service flow (jiying/service.py).

Uses a fake gateway client and injected parse/analysis functions, so no
network or LLM is involved: the tests exercise enqueue -> worker -> reply
-> ack ordering, dedup, help/error paths and failure semantics.
"""

import asyncio
import json

import pytest

from jiying.config import JiyingConfig
from jiying.parser import HELP_TEXT, AnalysisRequest
from jiying.protocol import parse_envelope
from jiying.report import AnalysisOutcome
from jiying.service import JiyingService
from jiying.ws_client import PERMANENT_RPC_CODES, RequestError

pytestmark = pytest.mark.unit

CONFIG = JiyingConfig(
    ws_url="wss://example.test/api/app/ws",
    app_id="app-1",
    app_secret="secret",
    status_interval_seconds=3600.0,
)


class FakeClient:
    connected = False

    def __init__(self):
        self.sent: list[dict] = []
        self.acked: list[int] = []
        self.statuses: list[str] = []
        self.topics_created: list[tuple[str, str]] = []
        self.fail_sends = False
        self.fail_ack = False
        self.fail_create_topic = False

    async def send_message(
        self, target_type, conversation_id, message, *, reply_to_message_id=None
    ):
        if self.fail_sends:
            raise ConnectionError("gateway down")
        self.sent.append(
            {
                "target_type": target_type,
                "conversation_id": conversation_id,
                "content": message.get("content", ""),
                "reply_to": reply_to_message_id,
            }
        )
        return {}

    async def ack(self, cursor):
        if self.fail_ack:
            raise ConnectionError("gateway down")
        self.acked.append(cursor)
        return {"cursor": cursor}

    async def create_topic(self, conversation_id, source_message_id):
        if self.fail_create_topic:
            raise ConnectionError("gateway down")
        self.topics_created.append((conversation_id, source_message_id))
        return {
            "conversation": {"id": "topic-conv-9", "type": "topic", "name": "话题"},
            "parent_conversation_id": conversation_id,
            "source_message_id": source_message_id,
            "created": True,
            "archived": False,
        }

    async def send_status(self, conversation_id, status):
        self.statuses.append(status)


def long_analysis_fn(request):
    """~50KB of report body: past the topic threshold and past the inline cap."""
    return AnalysisOutcome(
        ticker=request.ticker,
        trade_date=request.trade_date,
        asset_type=request.asset_type,
        signal="Hold",
        final_state={"market_report": "M" * 50_000},
    )


def message_event(
    cursor: int = 100,
    text: str = "分析 NVDA",
    conversation_id: str = "conv-1",
    message_id: str = "msg-1",
    conversation_type: str = "app",
):
    return parse_envelope(
        json.dumps(
            {
                "v": 1,
                "kind": "event",
                "id": f"evt-{cursor}",
                "cursor": cursor,
                "event": "message.created",
                "payload": {
                    "conversation": {"id": conversation_id, "type": conversation_type},
                    "sender": {"id": "user-1", "type": "user"},
                    "message": {
                        "id": message_id,
                        "seq": 1,
                        "body": {"type": "text", "content": text},
                    },
                },
            }
        )
    )


async def default_parse_fn(text):
    return AnalysisRequest("NVDA", "2026-09-12", "stock")


def default_analysis_fn(request):
    return AnalysisOutcome(
        ticker=request.ticker,
        trade_date=request.trade_date,
        asset_type=request.asset_type,
        signal="Buy",
        final_state={"final_trade_decision": "决策内容"},
    )


def make_service(client, *, parse_fn=None, analysis_fn=None, config=CONFIG):
    return JiyingService(
        config,
        client=client,
        parse_fn=parse_fn or default_parse_fn,
        analysis_fn=analysis_fn or default_analysis_fn,
        graph_config={"llm_provider": "openai"},
    )


async def drain(service):
    await service._queue.join()
    await asyncio.sleep(0)


class TestSuccessFlow:
    def test_report_sent_then_acked(self):
        async def scenario():
            client = FakeClient()
            service = make_service(client)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=100))
            await drain(service)
            worker.cancel()

            assert len(client.sent) == 1
            sent = client.sent[0]
            assert sent["target_type"] == "conversation"
            assert sent["conversation_id"] == "conv-1"
            assert sent["reply_to"] == "msg-1"
            assert "Buy" in sent["content"]
            assert "决策内容" in sent["content"]
            assert client.acked == [100]
            assert 100 in service._processed_cursors

        asyncio.run(scenario())

    def test_multi_chunk_report_sends_all_before_ack(self):
        def analysis_fn(request):
            return AnalysisOutcome(
                ticker=request.ticker,
                trade_date=request.trade_date,
                asset_type=request.asset_type,
                signal="Hold",
                final_state={"market_report": "M" * 10_000},
            )

        async def scenario():
            client = FakeClient()
            service = make_service(client, analysis_fn=analysis_fn)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=101))
            await drain(service)
            worker.cancel()

            assert len(client.sent) > 1
            assert client.acked == [101]

        asyncio.run(scenario())


class TestDedup:
    def test_replayed_cursor_processed_once(self):
        async def scenario():
            client = FakeClient()
            service = make_service(client)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=100))
            await drain(service)
            first_count = len(client.sent)

            await service._on_event(message_event(cursor=100))  # replay
            await drain(service)
            worker.cancel()

            assert len(client.sent) == first_count
            assert client.acked == [100]

        asyncio.run(scenario())


class TestHelpAndErrors:
    def test_unparseable_message_replies_help_and_acks(self):
        async def parse_fn(text):
            return None

        async def scenario():
            client = FakeClient()
            service = make_service(client, parse_fn=parse_fn)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=102, text="你好"))
            await drain(service)
            worker.cancel()

            assert client.sent[0]["content"] == HELP_TEXT
            assert client.acked == [102]

        asyncio.run(scenario())

    def test_status_command_replies_without_llm(self):
        async def must_not_be_called(text):
            raise AssertionError("parser must not run for /status")

        async def scenario():
            client = FakeClient()
            client.connected = True
            service = make_service(client, parse_fn=must_not_be_called)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=110, text="/status"))
            await drain(service)
            worker.cancel()

            content = client.sent[0]["content"]
            assert "服务状态" in content
            assert "provider" in content
            assert "已连接" in content
            assert client.acked == [110]

        asyncio.run(scenario())

    def test_help_command_replies_without_llm(self):
        async def must_not_be_called(text):
            raise AssertionError("parser must not run for /help")

        async def scenario():
            client = FakeClient()
            service = make_service(client, parse_fn=must_not_be_called)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=111, text="/help"))
            await drain(service)
            worker.cancel()

            assert client.sent[0]["content"] == HELP_TEXT
            assert client.acked == [111]

        asyncio.run(scenario())

    def test_parse_exception_replies_error_summary_not_help(self):
        async def parse_fn(text):
            raise RuntimeError("LLM endpoint unreachable")

        async def scenario():
            client = FakeClient()
            service = make_service(client, parse_fn=parse_fn)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=112, text="分析一下洲际油气"))
            await drain(service)
            worker.cancel()

            content = client.sent[0]["content"]
            assert "解析失败" in content
            assert "LLM endpoint unreachable" in content
            assert content != HELP_TEXT
            assert client.acked == [112]

        asyncio.run(scenario())

    def test_analysis_failure_replies_error_and_acks(self):
        def analysis_fn(request):
            raise RuntimeError("data vendor down")

        async def scenario():
            client = FakeClient()
            service = make_service(client, analysis_fn=analysis_fn)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=103))
            await drain(service)
            worker.cancel()

            assert len(client.sent) == 1
            assert "出错" in client.sent[0]["content"]
            assert client.acked == [103]

        asyncio.run(scenario())

    def test_send_failure_leaves_event_unacked(self):
        async def scenario():
            client = FakeClient()
            client.fail_sends = True
            service = make_service(client)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=104))
            await drain(service)
            worker.cancel()

            assert client.sent == []
            assert client.acked == []
            assert 104 not in service._processed_cursors

        asyncio.run(scenario())

    def test_ack_failure_leaves_event_unacked(self):
        async def scenario():
            client = FakeClient()
            client.fail_ack = True
            service = make_service(client)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=105))
            await drain(service)
            worker.cancel()

            assert len(client.sent) == 1
            assert client.acked == []
            assert 105 not in service._processed_cursors

        asyncio.run(scenario())


class TestTopicDelivery:
    def test_long_report_goes_to_topic_with_summary_in_main_chat(self):
        async def scenario():
            client = FakeClient()
            service = make_service(client, analysis_fn=long_analysis_fn)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=120))
            await drain(service)
            worker.cancel()

            # Topic created from the user's request message.
            assert client.topics_created == [("conv-1", "msg-1")]
            # Full report parts land in the topic conversation. They must NOT
            # carry a reply reference: the source message belongs to the
            # parent conversation, and cross-conversation quotes are rejected
            # by the server (invalid_request: 引用消息无效).
            topic_parts = [m for m in client.sent if m["conversation_id"] == "topic-conv-9"]
            assert len(topic_parts) > 6
            assert all("M" in m["content"] for m in topic_parts)
            assert all(m["reply_to"] is None for m in topic_parts)
            # Main chat receives only the headline summary + pointer, still
            # replying to the user's request message.
            main = [m for m in client.sent if m["conversation_id"] == "conv-1"]
            assert len(main) == 1
            assert "话题" in main[0]["content"]
            assert "Hold" in main[0]["content"]
            assert "完整报告共" in main[0]["content"]
            assert main[0]["reply_to"] == "msg-1"
            assert client.acked == [120]

        asyncio.run(scenario())

    def test_short_report_stays_inline(self):
        async def scenario():
            client = FakeClient()
            service = make_service(client)  # default small analysis
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=121))
            await drain(service)
            worker.cancel()

            assert client.topics_created == []
            assert all(m["conversation_id"] == "conv-1" for m in client.sent)
            assert client.acked == [121]

        asyncio.run(scenario())

    def test_topic_delivery_disabled_caps_inline(self):
        config = JiyingConfig(
            ws_url=CONFIG.ws_url,
            app_id=CONFIG.app_id,
            app_secret=CONFIG.app_secret,
            topic_delivery=False,
        )

        async def scenario():
            client = FakeClient()
            service = make_service(client, analysis_fn=long_analysis_fn, config=config)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=122))
            await drain(service)
            worker.cancel()

            assert client.topics_created == []
            assert len(client.sent) == config.max_report_messages
            assert "截断" in client.sent[-1]["content"]
            assert client.acked == [122]

        asyncio.run(scenario())

    def test_already_in_topic_gets_full_inline_report(self):
        async def scenario():
            client = FakeClient()
            service = make_service(client, analysis_fn=long_analysis_fn)
            worker = asyncio.create_task(service._worker())

            await service._on_event(
                message_event(cursor=123, conversation_id="topic-conv-1",
                              conversation_type="topic")
            )
            await drain(service)
            worker.cancel()

            assert client.topics_created == []  # never nest topics
            inline = [m for m in client.sent if m["conversation_id"] == "topic-conv-1"]
            assert len(inline) > 6  # high cap, no truncation
            assert client.acked == [123]

        asyncio.run(scenario())

    def test_topic_creation_failure_falls_back_inline(self):
        async def scenario():
            client = FakeClient()
            client.fail_create_topic = True
            service = make_service(client, analysis_fn=long_analysis_fn)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=124))
            await drain(service)
            worker.cancel()

            assert client.topics_created == []
            # Fallback keeps the high cap instead of truncating.
            assert len(client.sent) > 6
            assert client.acked == [124]

        asyncio.run(scenario())


class TestPermanentVsTransientFailures:
    """Doc 2.4: permanent RPC errors must not replay; transient ones must."""

    def test_permanent_rpc_error_notifies_and_acks(self):
        class TopicSendRejects(FakeClient):
            async def send_message(
                self, target_type, conversation_id, message, *,
                reply_to_message_id=None,
            ):
                if conversation_id == "topic-conv-9":
                    raise RequestError("invalid_request", "引用消息无效")
                return await super().send_message(
                    target_type, conversation_id, message,
                    reply_to_message_id=reply_to_message_id,
                )

        async def scenario():
            client = TopicSendRejects()
            service = make_service(client, analysis_fn=long_analysis_fn)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=130))
            await drain(service)
            worker.cancel()

            notice = [m for m in client.sent if "投递失败" in m["content"]]
            assert len(notice) == 1
            assert "invalid_request" in notice[0]["content"]
            assert "引用消息无效" in notice[0]["content"]
            # ACKed so the platform stops replaying (no repeated analyses).
            assert client.acked == [130]
            assert 130 in service._processed_cursors

        asyncio.run(scenario())

    def test_transient_rpc_error_leaves_unacked_for_replay(self):
        class TopicSendDrops(FakeClient):
            async def send_message(
                self, target_type, conversation_id, message, *,
                reply_to_message_id=None,
            ):
                if conversation_id == "topic-conv-9":
                    raise ConnectionError("gateway down")
                return await super().send_message(
                    target_type, conversation_id, message,
                    reply_to_message_id=reply_to_message_id,
                )

        async def scenario():
            client = TopicSendDrops()
            service = make_service(client, analysis_fn=long_analysis_fn)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=131))
            await drain(service)
            worker.cancel()

            assert client.acked == []
            assert 131 not in service._processed_cursors
            # No error notice either: the event will replay and retry.
            assert all("投递失败" not in m["content"] for m in client.sent)

        asyncio.run(scenario())

    def test_error_code_classification(self):
        assert RequestError("invalid_request", "x").is_permanent
        assert RequestError("forbidden", "x").is_permanent
        assert RequestError("not_found", "x").is_permanent
        assert not RequestError("internal_error", "x").is_permanent
        assert not RequestError(None, "x").is_permanent
        assert {
            "invalid_request", "forbidden", "not_found",
            "request_id_conflict", "response_too_large",
        } >= PERMANENT_RPC_CODES


class TestQueue:
    def test_queue_full_rejects_with_notice_and_acks(self):
        async def scenario():
            client = FakeClient()
            tight_config = JiyingConfig(
                ws_url=CONFIG.ws_url,
                app_id=CONFIG.app_id,
                app_secret=CONFIG.app_secret,
                queue_size=1,
            )
            service = make_service(client, config=tight_config)
            worker = asyncio.create_task(service._worker())

            # Occupy the queue; the second event must be rejected at once.
            released = asyncio.Event()

            async def slow_parse(text):
                await released.wait()
                return AnalysisRequest("NVDA", "2026-09-12", "stock")

            service._parse_fn = slow_parse
            await service._on_event(message_event(cursor=106))
            await service._on_event(message_event(cursor=107))
            await asyncio.sleep(0)

            released.set()
            await drain(service)
            worker.cancel()

            rejection = [m for m in client.sent if "排满" in m["content"]]
            assert len(rejection) == 1
            assert sorted(client.acked) == [106, 107]
            assert 106 in service._processed_cursors
            assert 107 in service._processed_cursors

        asyncio.run(scenario())

    def test_non_message_events_ignored(self):
        async def scenario():
            client = FakeClient()
            service = make_service(client)
            worker = asyncio.create_task(service._worker())

            await service._on_event(
                parse_envelope(
                    json.dumps(
                        {
                            "v": 1,
                            "kind": "event",
                            "id": "evt-x",
                            "cursor": 108,
                            "event": "topic.closed",
                            "payload": {"conversation_id": "topic-1"},
                        }
                    )
                )
            )
            await drain(service)
            worker.cancel()

            assert client.sent == []
            assert client.acked == []

        asyncio.run(scenario())


class TestHeartbeat:
    def test_status_sent_while_processing(self):
        async def parse_fn(text):
            await asyncio.sleep(0.05)
            return AnalysisRequest("NVDA", "2026-09-12", "stock")

        config = JiyingConfig(
            ws_url=CONFIG.ws_url,
            app_id=CONFIG.app_id,
            app_secret=CONFIG.app_secret,
            status_interval_seconds=0.01,
        )

        async def scenario():
            client = FakeClient()
            service = make_service(client, parse_fn=parse_fn, config=config)
            worker = asyncio.create_task(service._worker())

            await service._on_event(message_event(cursor=109))
            await drain(service)
            worker.cancel()

            assert any("正在分析 NVDA" in status for status in client.statuses)
            assert client.acked == [109]

        asyncio.run(scenario())
