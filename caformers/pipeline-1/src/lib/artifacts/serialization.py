from __future__ import annotations

import json
from pathlib import Path


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
