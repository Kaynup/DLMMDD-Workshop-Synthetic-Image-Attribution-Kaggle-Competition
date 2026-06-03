from __future__ import annotations

import json
import logging


class ConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class DetailedFormatter(logging.Formatter):
    def __init__(self) -> None:
        fmt = (
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | "
            "%(funcName)s | %(message)s"
        )
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, str | int | float] = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "message": record.getMessage(),
        }
        for key in ("run_id", "fold", "model", "epoch", "category"):
            value = getattr(record, key, None)
            if value not in (None, ""):
                payload[key] = value
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        return json.dumps(payload, default=str)
