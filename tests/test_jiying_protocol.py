"""Unit tests for the JiYing v1 envelope protocol (jiying/protocol.py)."""

import pytest

from jiying.protocol import (
    PROTOCOL_VERSION,
    Envelope,
    ProtocolError,
    build_request,
    parse_envelope,
)

pytestmark = pytest.mark.unit


class TestParseEnvelope:
    def test_parses_event_envelope(self):
        raw = (
            '{"v": 1, "kind": "event", "id": "evt-1", "cursor": 1287, '
            '"event": "message.created", "payload": {"message": {"body": '
            '{"type": "text", "content": "hi"}}}}'
        )
        envelope = parse_envelope(raw)
        assert envelope.kind == "event"
        assert envelope.event == "message.created"
        assert envelope.cursor == 1287
        assert envelope.payload["message"]["body"]["content"] == "hi"

    def test_parses_success_response(self):
        raw = (
            '{"v": 1, "kind": "response", "id": "srv-1", "reply_to": "req-1", '
            '"ok": true, "payload": {}}'
        )
        envelope = parse_envelope(raw)
        assert envelope.kind == "response"
        assert envelope.reply_to == "req-1"
        assert envelope.ok is True
        assert envelope.error is None

    def test_parses_error_response(self):
        raw = (
            '{"v": 1, "kind": "response", "reply_to": "req-1", "ok": false, '
            '"error": {"code": "forbidden", "message": "nope"}}'
        )
        envelope = parse_envelope(raw)
        assert envelope.ok is False
        assert envelope.error_code == "forbidden"

    def test_missing_payload_defaults_to_empty_dict(self):
        envelope = parse_envelope('{"v": 1, "kind": "event", "event": "x"}')
        assert envelope.payload == {}

    def test_invalid_json_raises(self):
        with pytest.raises(ProtocolError, match="not valid JSON"):
            parse_envelope("{not json")

    def test_non_object_raises(self):
        with pytest.raises(ProtocolError, match="JSON object"):
            parse_envelope("[1, 2, 3]")

    def test_unknown_kind_raises(self):
        with pytest.raises(ProtocolError, match="Unknown envelope kind"):
            parse_envelope('{"v": 1, "kind": "ping"}')

    def test_non_object_payload_raises(self):
        with pytest.raises(ProtocolError, match="payload"):
            parse_envelope('{"v": 1, "kind": "event", "payload": [1]}')

    def test_non_integer_cursor_raises(self):
        with pytest.raises(ProtocolError, match="cursor"):
            parse_envelope('{"v": 1, "kind": "event", "cursor": "1287"}')

    def test_bytes_input(self):
        envelope = parse_envelope(b'{"v": 1, "kind": "event", "event": "x"}')
        assert envelope.event == "x"


class TestBuildRequest:
    def test_roundtrip(self):
        frame = build_request("req-9", "events.ack", {"cursor": 5})
        envelope = parse_envelope(frame)
        assert envelope.kind == "request"
        assert envelope.id == "req-9"
        assert envelope.method == "events.ack"
        assert envelope.payload == {"cursor": 5}

    def test_frame_carries_protocol_version(self):
        import json

        frame = json.loads(build_request("id", "m", {}))
        assert frame["v"] == PROTOCOL_VERSION

    def test_unicode_payload_not_escaped(self):
        frame = build_request("id", "message.send", {"content": "分析报告"})
        assert "分析报告" in frame


class TestEnvelopeHelpers:
    def test_to_dict_only_serializes_requests(self):
        envelope = Envelope(kind="request", id="i", method="m", payload={"a": 1})
        assert envelope.to_dict() == {
            "v": PROTOCOL_VERSION,
            "kind": "request",
            "id": "i",
            "method": "m",
            "payload": {"a": 1},
        }

    def test_error_code_none_without_error(self):
        envelope = Envelope(kind="response", ok=True)
        assert envelope.error_code is None
