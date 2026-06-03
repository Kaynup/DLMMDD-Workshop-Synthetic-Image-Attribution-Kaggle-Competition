from __future__ import annotations

from typing import Any

_CONTEXT: dict[str, Any] = {
    "run_id": None,
    "fold": None,
    "model": None,
    "epoch": None,
}


def set_context(**kwargs: Any) -> None:
    for key, value in kwargs.items():
        if key in _CONTEXT:
            _CONTEXT[key] = value


def get_context() -> dict[str, Any]:
    return dict(_CONTEXT)


def reset_context() -> None:
    for key in _CONTEXT:
        _CONTEXT[key] = None
