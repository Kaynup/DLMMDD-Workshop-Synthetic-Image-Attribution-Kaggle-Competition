from __future__ import annotations

from pathlib import Path
import torch

from lib.core.logging import checkpoint_logger
from lib.core.utils import ensure_dir


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    meta: dict | None = None,
) -> None:
    ensure_dir(path.parent)
    state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model_state_dict": state_model.state_dict(),
        "metadata": meta or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)
    ensure_dir(path.parent)
    with path.with_suffix(".json").open("w", encoding="utf-8") as fh:
        import json

        json.dump(meta or {}, fh, indent=2, default=str)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer=None,
    scheduler=None,
    scaler=None,
    strict: bool = True,
):
    checkpoint_logger.info(f"[checkpoint] loading {path}")
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {path}") from exc
    state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state_model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    meta = ckpt.get("metadata", {})
    checkpoint_logger.info(f"[checkpoint] loaded {path.name}")
    return model, meta


def prune_checkpoints(saved: list[dict], keep: int) -> list[dict]:
    if keep <= 0:
        return []
    saved = sorted(saved, key=lambda r: r["sv"], reverse=True)
    to_remove = saved[keep:]
    for item in to_remove:
        p = Path(item["path"])
        try:
            if p.exists():
                p.unlink()
            jf = p.with_suffix(".json")
            if jf.exists():
                jf.unlink()
            checkpoint_logger.info(f"[prune] removed {p.name} sv={item['sv']:.4f}")
        except Exception as exc:
            checkpoint_logger.warning(f"[prune] failed removing {p}: {exc}")
    return saved[:keep]
