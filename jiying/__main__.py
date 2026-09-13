"""Entry point: ``python -m jiying``.

Starts the standalone JiYing bridge server. Required environment:

- ``JIYING_WS_URL``      e.g. wss://server.example.com/api/app/ws
- ``JIYING_APP_ID``      App ID from the JiYing "应用接入信息" panel
- ``JIYING_APP_SECRET``  connection secret (never commit this)

TradingAgents behaviour (LLM provider, models, language, ...) is reused
from the shared ``TRADINGAGENTS_*`` / ``.env`` configuration.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys

from .config import ConfigError, JiyingConfig
from .service import JiyingService


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    try:
        config = JiyingConfig.from_env()
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    service = JiyingService(config)
    logger.info(
        "Starting JiYing bridge service (topic_delivery=%s, "
        "topic_threshold=%d, topic_max_messages=%d, inline_max_messages=%d, "
        "chunk_chars=%d, queue_size=%d)",
        config.topic_delivery,
        config.topic_threshold_messages,
        config.topic_max_messages,
        config.max_report_messages,
        config.report_chunk_chars,
        config.queue_size,
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(service.run())
    return 0


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    main()
