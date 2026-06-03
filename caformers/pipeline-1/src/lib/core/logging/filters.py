from __future__ import annotations

import logging


class CategoryFilter(logging.Filter):
    def __init__(self, category: str) -> None:
        super().__init__()
        self.category = category

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "category", None) == self.category


class ErrorOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        from .context import get_context

        context = get_context()
        for key, value in context.items():
            if value is not None:
                setattr(record, key, value)
        return True
