from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .filters import CategoryFilter, ErrorOnlyFilter
from .formatters import ConsoleFormatter, DetailedFormatter, JsonFormatter
from .levels import resolve_level


def _build_rotating_handler(
    path: Path,
    level: int | str = logging.DEBUG,
    formatter: logging.Formatter | None = None,
) -> logging.Handler:
    handler = RotatingFileHandler(
        path,
        maxBytes=25 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    handler.setLevel(resolve_level(level))
    handler.setFormatter(formatter or DetailedFormatter())
    return handler


def create_console_handler(level: int | str = logging.INFO) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(resolve_level(level))
    handler.setFormatter(ConsoleFormatter())
    return handler


def create_debug_handler(log_dir: Path) -> logging.Handler:
    path = log_dir / "debug" / "run.log"
    handler = _build_rotating_handler(path, level=logging.DEBUG)
    return handler


def create_error_handler(log_dir: Path) -> logging.Handler:
    path = log_dir / "errors" / "errors.log"
    handler = _build_rotating_handler(path, level=logging.ERROR)
    handler.addFilter(ErrorOnlyFilter())
    return handler


def create_training_handler(log_dir: Path) -> logging.Handler:
    path = log_dir / "training" / "training.log"
    handler = _build_rotating_handler(path, level=logging.INFO)
    handler.addFilter(CategoryFilter("training"))
    return handler


def create_metrics_handler(log_dir: Path) -> logging.Handler:
    path = log_dir / "training" / "metrics.log"
    handler = _build_rotating_handler(path, level=logging.INFO, formatter=JsonFormatter())
    handler.addFilter(CategoryFilter("metric"))
    return handler


def create_checkpoint_handler(log_dir: Path) -> logging.Handler:
    path = log_dir / "checkpoints" / "checkpoints.log"
    handler = _build_rotating_handler(path, level=logging.INFO)
    handler.addFilter(CategoryFilter("checkpoint"))
    return handler


def create_resource_handler(log_dir: Path) -> logging.Handler:
    path = log_dir / "resources" / "resources.log"
    handler = _build_rotating_handler(path, level=logging.INFO)
    handler.addFilter(CategoryFilter("resource"))
    return handler
