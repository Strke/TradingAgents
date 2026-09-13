"""Environment-driven configuration for the JiYing bridge service.

All values come from ``JIYING_*`` environment variables so the service can be
configured purely through ``.env`` / docker-compose, following the same
convention as the ``TRADINGAGENTS_*`` overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_WS_URL = "JIYING_WS_URL"
_APP_ID = "JIYING_APP_ID"
_APP_SECRET = "JIYING_APP_SECRET"


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean (true/false), got {raw!r}")


class ConfigError(ValueError):
    """Raised when required JiYing settings are missing or invalid."""


@dataclass(frozen=True)
class JiyingConfig:
    """Connection and runtime knobs for the bridge service."""

    ws_url: str
    app_id: str
    app_secret: str
    # Report delivery
    report_chunk_chars: int = 3500
    max_report_messages: int = 12
    # Long reports are delivered into a topic created from the user's message
    # (doc 4.8) instead of flooding the main conversation. The main chat then
    # only receives the headline summary plus a pointer to the topic.
    topic_delivery: bool = True
    topic_threshold_messages: int = 6
    topic_max_messages: int = 40
    # Progress status heartbeat sent while an analysis is running
    status_interval_seconds: float = 3.0
    # RPC behaviour
    request_timeout_seconds: float = 30.0
    request_retries: int = 2
    max_pending_requests: int = 64
    # Reconnect backoff (exponential between min and max)
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 60.0
    # Serial work queue bound; requests beyond it are rejected with a notice
    queue_size: int = 16

    def __post_init__(self) -> None:
        if not self.ws_url:
            raise ConfigError(f"Missing required setting: {_WS_URL}")
        if not self.ws_url.startswith(("ws://", "wss://")):
            raise ConfigError(f"{_WS_URL} must start with ws:// or wss://")
        if not self.app_id:
            raise ConfigError(f"Missing required setting: {_APP_ID}")
        if not self.app_secret:
            raise ConfigError(f"Missing required setting: {_APP_SECRET}")
        if self.report_chunk_chars <= 0:
            raise ConfigError("JIYING_REPORT_CHUNK_CHARS must be positive")
        if self.max_report_messages <= 0:
            raise ConfigError("JIYING_MAX_REPORT_MESSAGES must be positive")
        if self.topic_threshold_messages <= 0:
            raise ConfigError("JIYING_TOPIC_THRESHOLD_MESSAGES must be positive")
        if self.topic_max_messages <= 0:
            raise ConfigError("JIYING_TOPIC_MAX_MESSAGES must be positive")

    @classmethod
    def from_env(cls) -> JiyingConfig:
        return cls(
            ws_url=_env_str(_WS_URL),
            app_id=_env_str(_APP_ID),
            app_secret=_env_str(_APP_SECRET),
            report_chunk_chars=_env_int("JIYING_REPORT_CHUNK_CHARS", 3500),
            max_report_messages=_env_int("JIYING_MAX_REPORT_MESSAGES", 12),
            topic_delivery=_env_bool("JIYING_TOPIC_DELIVERY", True),
            topic_threshold_messages=_env_int("JIYING_TOPIC_THRESHOLD_MESSAGES", 6),
            topic_max_messages=_env_int("JIYING_TOPIC_MAX_MESSAGES", 40),
            status_interval_seconds=_env_float("JIYING_STATUS_INTERVAL_SECONDS", 3.0),
            request_timeout_seconds=_env_float("JIYING_REQUEST_TIMEOUT_SECONDS", 30.0),
            request_retries=_env_int("JIYING_REQUEST_RETRIES", 2),
            max_pending_requests=_env_int("JIYING_MAX_PENDING_REQUESTS", 64),
            reconnect_min_seconds=_env_float("JIYING_RECONNECT_MIN_SECONDS", 1.0),
            reconnect_max_seconds=_env_float("JIYING_RECONNECT_MAX_SECONDS", 60.0),
            queue_size=_env_int("JIYING_QUEUE_SIZE", 16),
        )
