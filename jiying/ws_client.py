"""Async WebSocket client for the JiYing app gateway.

Responsibilities:
- establish the authenticated connection (App ID header + bearer secret)
- correlate RPC responses to requests via ``reply_to`` (order is not guaranteed)
- retry timed-out requests reusing the original request ID (idempotency, doc 2.3)
- reconnect with exponential backoff after any session failure

Notes:
- The server sends protocol-level Pings every 30s; the ``websockets`` library
  answers them automatically, so no application-level heartbeat is needed
  (doc 2.1). Client pings are disabled to stay strictly passive.
- Single frames are capped at 1 MiB by the platform; ``max_size`` mirrors that.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from .config import JiyingConfig
from .protocol import Envelope, ProtocolError, build_request, parse_envelope

logger = logging.getLogger(__name__)

# Hard cap matching the platform's 1 MiB per-frame limit.
MAX_FRAME_BYTES = 1024 * 1024


class NotConnectedError(RuntimeError):
    """Raised when an RPC is attempted while the session is down."""


class RequestError(RuntimeError):
    """Raised when the server answers an RPC with an error envelope."""

    def __init__(self, code: str | None, message: str):
        self.code = code
        super().__init__(f"JiYing RPC failed ({code}): {message}")


class RequestTimeoutError(RuntimeError):
    """Raised when an RPC exceeds its timeout after all retries."""


EventCallback = Callable[[Envelope], Awaitable[None]]


class JiyingClient:
    """Long-lived JiYing gateway client with RPC and event dispatch."""

    def __init__(self, config: JiyingConfig, on_event: EventCallback):
        self._config = config
        self._on_event = on_event
        self._ws: ClientConnection | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._backoff = config.reconnect_min_seconds
        self._stopping = False

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    async def run_forever(self) -> None:
        """Connect and keep reconnecting until :meth:`stop` is called."""
        headers = {
            "X-MagicChat-App-ID": self._config.app_id,
            "Authorization": f"Bearer {self._config.app_secret}",
        }
        while not self._stopping:
            try:
                async with connect(
                    self._config.ws_url,
                    additional_headers=headers,
                    max_size=MAX_FRAME_BYTES,
                    # Server drives keepalive via its own Pings (doc 2.1).
                    ping_interval=None,
                ) as ws:
                    logger.info("Connected to JiYing gateway %s", self._config.ws_url)
                    self._backoff = self._config.reconnect_min_seconds
                    self._ws = ws
                    await self._recv_loop(ws)
            except ConnectionClosed as exc:
                logger.warning("JiYing connection closed: %s", exc)
            except (OSError, asyncio.TimeoutError, ProtocolError) as exc:
                logger.warning("JiYing connection error: %s", exc)
            finally:
                self._ws = None
                self._fail_pending(NotConnectedError("connection lost"))
            if self._stopping:
                break
            delay = self._backoff
            self._backoff = min(self._backoff * 2, self._config.reconnect_max_seconds)
            delay *= 0.5 + random.random()  # jitter
            logger.info("Reconnecting in %.1fs", delay)
            await asyncio.sleep(delay)

    def stop(self) -> None:
        """Signal the run loop to exit after the current connection drops."""
        self._stopping = True
        if self._ws is not None:
            self._ws.close()
        self._fail_pending(NotConnectedError("client stopped"))

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def _recv_loop(self, ws: ClientConnection) -> None:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            envelope = parse_envelope(raw)
            if envelope.kind == "response":
                self._resolve_pending(envelope)
            elif envelope.kind == "event":
                try:
                    await self._on_event(envelope)
                except Exception:
                    logger.exception("Unhandled error in event callback")

    # ------------------------------------------------------------------ #
    # RPC
    # ------------------------------------------------------------------ #

    async def request(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> dict[str, Any]:
        """Invoke an RPC and return the response payload.

        Timed-out attempts are retried with the *same* request ID so the
        server's idempotency cache collapses duplicates (doc 2.3).
        """
        if timeout is None:
            timeout = self._config.request_timeout_seconds
        if retries is None:
            retries = self._config.request_retries

        if self._ws is None:
            raise NotConnectedError("not connected to JiYing gateway")

        request_id = uuid.uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        if len(self._pending) >= self._config.max_pending_requests:
            raise RequestError("client_overloaded", "too many pending requests")
        self._pending[request_id] = future

        frame = build_request(request_id, method, payload)
        try:
            last_error: Exception | None = None
            for _ in range(retries + 1):
                ws = self._ws
                if ws is None:
                    raise NotConnectedError("not connected to JiYing gateway")
                await ws.send(frame)
                try:
                    return await asyncio.wait_for(asyncio.shield(future), timeout)
                except asyncio.TimeoutError:
                    last_error = RequestTimeoutError(
                        f"request {method} timed out after {timeout}s"
                    )
                    logger.warning(
                        "Request %s (%s) timed out; retrying with same id",
                        method,
                        request_id,
                    )
            raise last_error  # type: ignore[misc]
        finally:
            self._pending.pop(request_id, None)

    def _resolve_pending(self, envelope: Envelope) -> None:
        reply_to = envelope.reply_to
        if reply_to is None:
            return
        future = self._pending.get(reply_to)
        if future is None or future.done():
            # Late duplicate response for a retried request: safe to ignore.
            return
        if envelope.ok:
            future.set_result(envelope.payload)
        else:
            future.set_exception(
                RequestError(
                    envelope.error_code,
                    str((envelope.error or {}).get("message", "unknown error")),
                )
            )

    def _fail_pending(self, exc: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)
        self._pending.clear()

    # ------------------------------------------------------------------ #
    # High-level helpers
    # ------------------------------------------------------------------ #

    async def send_message(
        self,
        target_type: str,
        conversation_id: str,
        message: dict[str, Any],
        *,
        reply_to_message_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a message via ``message.send`` (doc 4.3)."""
        payload: dict[str, Any] = {
            "target": {"type": target_type, "conversation_id": conversation_id},
            "message": message,
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id
        return await self.request("message.send", payload)

    async def send_text(
        self,
        target_type: str,
        conversation_id: str,
        content: str,
        *,
        reply_to_message_id: str | None = None,
    ) -> dict[str, Any]:
        return await self.send_message(
            target_type,
            conversation_id,
            {"type": "markdown", "content": content},
            reply_to_message_id=reply_to_message_id,
        )

    async def ack(self, cursor: int) -> dict[str, Any]:
        """Confirm reliable events up to ``cursor`` (doc 4.20)."""
        return await self.request("events.ack", {"cursor": cursor})

    async def send_status(self, conversation_id: str, status: str) -> None:
        """Best-effort typing status update; never raises (doc 4.21)."""
        try:
            await self.request(
                "conversation.status",
                {"conversation_id": conversation_id, "status": status},
            )
        except Exception as exc:
            logger.debug("Status update skipped: %s", exc)


def dumps_json(value: Any) -> str:
    """Small helper kept for tests/debugging of outbound frames."""
    return json.dumps(value, ensure_ascii=False)
