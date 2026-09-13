"""JiYing bridge service: event intake, serial analysis queue, replies.

Flow per ``message.created`` event (doc 3.4):

1. dedup by cursor (in-memory; single-instance deployment assumption)
2. enqueue; the worker processes strictly one analysis at a time
3. parse the message into ticker/date/asset (LLM-assisted)
4. run :meth:`TradingAgentsGraph.propagate` in a worker thread
5. deliver the chunked markdown report via ``message.send``
6. only then ``events.ack`` the cursor

Processing failures answer with an error notice and still ACK, so a
permanently failing message cannot wedge the queue (the platform would
otherwise replay it forever). Transient RPC failures before any reply is
delivered leave the event un-ACKed on purpose, so it replays after reconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass

from tradingagents.agents.utils.rating import is_review
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

from .config import JiyingConfig
from .parser import HELP_TEXT, AnalysisRequest, parse_request
from .protocol import Envelope
from .report import AnalysisOutcome, format_report
from .ws_client import JiyingClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Job:
    envelope: Envelope
    conversation_id: str
    conversation_type: str
    message_id: str
    text: str


class JiyingService:
    """Wires the gateway client to the TradingAgents engine."""

    def __init__(
        self,
        config: JiyingConfig,
        *,
        client: JiyingClient | None = None,
        parse_fn=None,
        analysis_fn=None,
        graph_config: dict | None = None,
    ):
        self._config = config
        self._client = client or JiyingClient(config, on_event=self._on_event)
        self._parse_fn = parse_fn
        self._analysis_fn = analysis_fn
        self._graph_config = graph_config

        self._queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=config.queue_size)
        self._processed_cursors: set[int] = set()
        self._graph: TradingAgentsGraph | None = None
        self._graph_lock = threading.Lock()
        self._worker_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        self._worker_task = asyncio.create_task(self._worker(), name="jiying-worker")
        try:
            await self._client.run_forever()
        finally:
            if self._worker_task is not None:
                self._worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._worker_task

    # ------------------------------------------------------------------ #
    # Event intake
    # ------------------------------------------------------------------ #

    async def _on_event(self, envelope: Envelope) -> None:
        event = envelope.event
        if event == "message.created":
            await self._enqueue_message(envelope)
        elif event == "topic.closed":
            payload = envelope.payload
            logger.info(
                "Topic %s closed; dropping related work",
                payload.get("conversation_id"),
            )
        elif event == "conversation.status":
            pass  # transient typing indicator, not interesting to us
        else:
            logger.debug("Ignoring event %s", event)

    async def _enqueue_message(self, envelope: Envelope) -> None:
        cursor = envelope.cursor
        if cursor is None:
            logger.warning("message.created without cursor; skipping")
            return
        if cursor in self._processed_cursors:
            logger.info("Deduplicating replayed cursor %s", cursor)
            return

        message = envelope.payload.get("message") or {}
        conversation = envelope.payload.get("conversation") or {}
        body = message.get("body") or {}
        text = str(body.get("content") or "").strip()
        job = _Job(
            envelope=envelope,
            conversation_id=str(conversation.get("id") or ""),
            conversation_type=str(conversation.get("type") or "conversation"),
            message_id=str(message.get("id") or ""),
            text=text,
        )

        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.warning("Queue full; rejecting cursor %s", cursor)
            await self._safe_send_text(
                job, "当前分析任务已排满，请稍后再试。", ack=True
            )

    # ------------------------------------------------------------------ #
    # Serial worker
    # ------------------------------------------------------------------ #

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._handle_job(job)
            except Exception:
                # Unexpected (usually connection) failure: do NOT ack, so the
                # platform replays this event after reconnect and no report
                # is silently lost. Expected failures are handled inside
                # _handle_job with a reply + ack.
                logger.exception(
                    "Job for cursor %s failed; leaving un-ACKed for replay",
                    job.envelope.cursor,
                )
            finally:
                self._queue.task_done()

    async def _handle_job(self, job: _Job) -> None:
        cursor = job.envelope.cursor
        assert cursor is not None

        try:
            request = await self._parse(job.text)
        except Exception:
            logger.exception("Failed to parse message (cursor %s)", cursor)
            await self._safe_send_text(job, HELP_TEXT, ack=True)
            return

        if request is None:
            await self._safe_send_text(job, HELP_TEXT, ack=True)
            return

        position = self._queue.qsize() + 1
        if position > 1:
            await self._safe_send_text(
                job,
                f"已收到 **{request.label}** 的分析请求，当前排队第 {position} 位，"
                "完成后自动推送完整报告。",
                ack=False,
            )

        heartbeat = asyncio.create_task(
            self._heartbeat(job, request), name="jiying-status"
        )
        try:
            outcome = await asyncio.to_thread(self._run_analysis, request)
        except Exception:
            logger.exception("Analysis failed for %s", request.label)
            await self._safe_send_text(
                job,
                f"分析 **{request.label}** 时出错，请稍后重试或联系管理员。",
                ack=True,
            )
            return
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

        messages = format_report(
            outcome,
            chunk_chars=self._config.report_chunk_chars,
            max_messages=self._config.max_report_messages,
        )
        for content in messages:
            await self._send(job, content)

        await self._client.ack(cursor)
        self._processed_cursors.add(cursor)
        logger.info(
            "Delivered %d-part report for %s (cursor %s)",
            len(messages), request.label, cursor,
        )

    # ------------------------------------------------------------------ #
    # Pieces
    # ------------------------------------------------------------------ #

    async def _parse(self, text: str) -> AnalysisRequest | None:
        if self._parse_fn is not None:
            return await self._parse_fn(text)
        # Graph construction is expensive/blocking: init off the event loop.
        llm = await asyncio.to_thread(self._ensure_graph_llm)
        return await parse_request(text, llm)

    def _ensure_graph_llm(self):
        graph = self._ensure_graph()
        return graph.quick_thinking_llm

    def _ensure_graph(self) -> TradingAgentsGraph:
        with self._graph_lock:
            if self._graph is None:
                config = (self._graph_config or DEFAULT_CONFIG).copy()
                logger.info(
                    "Initializing TradingAgentsGraph (provider=%s, deep=%s, quick=%s)",
                    config.get("llm_provider"),
                    config.get("deep_think_llm"),
                    config.get("quick_think_llm"),
                )
                self._graph = TradingAgentsGraph(debug=False, config=config)
            return self._graph

    def _run_analysis(self, request: AnalysisRequest) -> AnalysisOutcome:
        if self._analysis_fn is not None:
            return self._analysis_fn(request)
        graph = self._ensure_graph()
        final_state, signal = graph.propagate(
            request.ticker, request.trade_date, asset_type=request.asset_type
        )
        return AnalysisOutcome(
            ticker=request.ticker,
            trade_date=request.trade_date,
            asset_type=request.asset_type,
            signal="REVIEW" if is_review(signal) else str(signal),
            final_state=dict(final_state or {}),
        )

    async def _heartbeat(self, job: _Job, request: AnalysisRequest) -> None:
        started = asyncio.get_running_loop().time()
        while True:
            minutes = int((asyncio.get_running_loop().time() - started) // 60)
            label = (
                f"正在分析 {request.ticker}（{request.trade_date}），已进行 {minutes} 分钟"
                if minutes
                else f"正在分析 {request.ticker}（{request.trade_date}）"
            )
            await self._client.send_status(job.conversation_id, label)
            await asyncio.sleep(self._config.status_interval_seconds)

    async def _send(self, job: _Job, content: str) -> None:
        await self._client.send_message(
            "conversation",
            job.conversation_id,
            {"type": "markdown", "content": content},
            reply_to_message_id=job.message_id or None,
        )

    async def _safe_send_text(self, job: _Job, content: str, *, ack: bool) -> None:
        """Best-effort notice; never raises. ACKs when asked, swallowing errors."""
        try:
            await self._send(job, content)
        except Exception:
            logger.exception("Failed to deliver notice (cursor %s)", job.envelope.cursor)
            return
        if ack and job.envelope.cursor is not None:
            try:
                await self._client.ack(job.envelope.cursor)
                self._processed_cursors.add(job.envelope.cursor)
            except Exception:
                logger.exception(
                    "Failed to ACK cursor %s; event will replay",
                    job.envelope.cursor,
                )
