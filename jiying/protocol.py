"""JiYing WebSocket protocol (v1) envelope encoding and decoding.

Every inbound/outbound frame is a JSON Envelope:

- ``request``  : app -> server RPC (``id``, ``method``, ``payload``)
- ``response`` : server -> app RPC result (``reply_to``, ``ok``/``error``)
- ``event``    : server -> app push (``event``, ``cursor``, ``payload``)

See APPLICATION_DEVELOPMENT.md section 2.2 for the authoritative schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1

KIND_REQUEST = "request"
KIND_RESPONSE = "response"
KIND_EVENT = "event"


class ProtocolError(ValueError):
    """Raised when a frame cannot be parsed as a valid Envelope."""


@dataclass(frozen=True)
class Envelope:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    # request/response correlation
    id: str | None = None
    method: str | None = None
    reply_to: str | None = None
    ok: bool | None = None
    error: dict[str, Any] | None = None
    # event fields
    event: str | None = None
    cursor: int | None = None

    @property
    def error_code(self) -> str | None:
        if not self.error:
            return None
        code = self.error.get("code")
        return str(code) if code is not None else None

    def to_dict(self) -> dict[str, Any]:
        frame: dict[str, Any] = {"v": PROTOCOL_VERSION, "kind": self.kind}
        if self.kind == KIND_REQUEST:
            frame["id"] = self.id
            frame["method"] = self.method
            frame["payload"] = self.payload
        return frame


def parse_envelope(raw: str | bytes) -> Envelope:
    """Decode a raw WebSocket frame into an :class:`Envelope`."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Frame is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("Envelope must be a JSON object")

    kind = data.get("kind")
    if kind not in (KIND_REQUEST, KIND_RESPONSE, KIND_EVENT):
        raise ProtocolError(f"Unknown envelope kind: {kind!r}")

    payload = data.get("payload") or {}
    if not isinstance(payload, dict):
        raise ProtocolError("Envelope payload must be an object")

    cursor = data.get("cursor")
    if cursor is not None and not isinstance(cursor, int):
        raise ProtocolError("Envelope cursor must be an integer")

    error = data.get("error")
    if error is not None and not isinstance(error, dict):
        raise ProtocolError("Envelope error must be an object")

    ok = data.get("ok")
    if ok is not None and not isinstance(ok, bool):
        raise ProtocolError("Envelope ok must be a boolean")

    return Envelope(
        kind=kind,
        payload=payload,
        id=data.get("id"),
        method=data.get("method"),
        reply_to=data.get("reply_to"),
        ok=ok,
        error=error,
        event=data.get("event"),
        cursor=cursor,
    )


def build_request(request_id: str, method: str, payload: dict[str, Any]) -> str:
    """Serialize an outbound RPC request frame."""
    envelope = Envelope(
        kind=KIND_REQUEST, id=request_id, method=method, payload=payload
    )
    return json.dumps(envelope.to_dict(), ensure_ascii=False)
