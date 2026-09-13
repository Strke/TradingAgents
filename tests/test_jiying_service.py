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

pytestmark = pytest.mark.unit

CONFIG = JiyingConfig(
    ws_url="wss://example.test/api/app/ws",
    app_id="app-1",
    app_secret="secret",
    status_interval_seconds=3600.0,
)


class FakeClient:
    def __init__(self):
        self.sent: list[dict] = []
        self.acked: list[int] = []
        self.statuses: list[str] = []
        self.fail_sends = False
        self.fail_ack = False

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

    async def send_status(self, conversation_id, status):
        self.statuses.append(status)


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
