from __future__ import annotations

import logging

LOG_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def resolve_level(level: int | str) -> int:
    if isinstance(level, str):
        return LOG_LEVELS.get(level.lower(), logging.INFO)
    return level
