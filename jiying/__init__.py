"""JiYing (即应) platform bridge for TradingAgents.

This package implements a standalone WebSocket service that connects to the
JiYing app gateway, receives ``message.created`` events, runs the
TradingAgents analysis pipeline, and replies with the full markdown report.
See ``APPLICATION_DEVELOPMENT.md`` for the platform protocol.
"""
