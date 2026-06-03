from __future__ import annotations

import logging
from pathlib import Path

from lib.core.utils import ensure_dir
from .context import set_context
from .filters import ContextFilter
from .handlers import (
    create_checkpoint_handler,
    create_console_handler,
    create_debug_handler,
    create_error_handler,
    create_metrics_handler,
    create_resource_handler,
    create_training_handler,
)
from .levels import resolve_level

pipeline_logger = logging.getLogger("pipeline")
data_logger = logging.getLogger("pipeline.data")
model_logger = logging.getLogger("pipeline.model")
training_logger = logging.getLogger("pipeline.training")
metric_logger = logging.getLogger("pipeline.metric")
checkpoint_logger = logging.getLogger("pipeline.checkpoint")
resource_logger = logging.getLogger("pipeline.resources")
inference_logger = logging.getLogger("pipeline.inference")


def _prepare_log_directories(root_dir: Path) -> None:
    ensure_dir(root_dir)
    for subdir in ("debug", "training", "errors", "checkpoints", "resources", "archive"):
        ensure_dir(root_dir / subdir)


def get_logger(name: str | None = None) -> logging.Logger:
    if not name:
        return pipeline_logger
    if name.startswith("pipeline"):
        return logging.getLogger(name)
    return logging.getLogger(f"pipeline.{name}")


def setup_logging(
    log_dir: Path | None = None,
    run_id: str | None = None,
    level: int | str = logging.DEBUG,
) -> logging.Logger:
    if log_dir is None:
        raise ValueError("log_dir must be provided")
    _prepare_log_directories(log_dir)
    logger = pipeline_logger
    if logger.handlers:
        return logger
    logger.setLevel(resolve_level(level))
    logger.propagate = False
    logger.addFilter(ContextFilter())
    logger.addHandler(create_console_handler())
    logger.addHandler(create_debug_handler(log_dir))
    logger.addHandler(create_error_handler(log_dir))
    logger.addHandler(create_training_handler(log_dir))
    logger.addHandler(create_metrics_handler(log_dir))
    logger.addHandler(create_checkpoint_handler(log_dir))
    logger.addHandler(create_resource_handler(log_dir))
    if run_id is not None:
        set_context(run_id=run_id)
    logger.info("Logging subsystem initialized.")
    return logger


def initialize_logger_safely(log_dir: Path, run_id: str | None = None) -> logging.Logger:
    logger = setup_logging(log_dir, run_id=run_id)
    logger.info(f"LOG_DIR={log_dir}")
    if run_id is not None:
        logger.info(f"RUN_ID={run_id}")
    return logger


def log_training(msg: str, *args, logger: logging.Logger = training_logger, **kwargs) -> None:
    extra = kwargs.pop("extra", {})
    extra = {"category": "training", **extra}
    logger.info(msg, *args, extra=extra, **kwargs)


def log_metric(msg: str, *args, logger: logging.Logger = metric_logger, **kwargs) -> None:
    extra = kwargs.pop("extra", {})
    extra = {"category": "metric", **extra}
    logger.info(msg, *args, extra=extra, **kwargs)


def log_checkpoint(msg: str, *args, logger: logging.Logger = checkpoint_logger, **kwargs) -> None:
    extra = kwargs.pop("extra", {})
    extra = {"category": "checkpoint", **extra}
    logger.info(msg, *args, extra=extra, **kwargs)


def log_resource(msg: str, *args, logger: logging.Logger = resource_logger, **kwargs) -> None:
    extra = kwargs.pop("extra", {})
    extra = {"category": "resource", **extra}
    logger.info(msg, *args, extra=extra, **kwargs)
