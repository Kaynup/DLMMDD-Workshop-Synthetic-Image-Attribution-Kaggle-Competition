from __future__ import annotations

import psutil
import time
import torch

from lib.config.defaults import DISK_LIMIT_GIB, RAM_WARN_GIB, SESSION_BUDGET_SECS
from lib.config.paths import WORKING_ROOT
from lib.core.logging import log_resource, resource_logger


def _disk_gib() -> float:
    total_bytes = sum(
        f.stat().st_size
        for f in WORKING_ROOT.rglob("*")
        if f.is_file()
    )
    return total_bytes / (1024 ** 3)


def _ram_gib() -> float:
    return psutil.virtual_memory().used / (1024 ** 3)


def _gpu_memory_gib() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved": torch.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def _elapsed_h() -> float:
    return (time.time() - SESSION_START_TIME) / 3600.0  # type: ignore[name-defined]


def _remaining_h() -> float:
    return (SESSION_BUDGET_SECS - (time.time() - SESSION_START_TIME)) / 3600.0


def log_resources(tag: str = "") -> None:
    disk = _disk_gib()
    ram = _ram_gib()
    gpu = _gpu_memory_gib()
    remaining = max(0.0, _remaining_h())
    msg = (
        f"[resources{' ' + tag if tag else ''}] "
        f"disk={disk:.2f}GiB | "
        f"ram={ram:.1f}GiB | "
        f"elapsed={_elapsed_h():.2f}h | "
        f"remaining={remaining:.2f}h"
    )
    if torch.cuda.is_available():
        msg += (
            f" | cuda_alloc={gpu['allocated']:.2f}GiB "
            f"| cuda_reserved={gpu['reserved']:.2f}GiB "
            f"| cuda_peak={gpu['max_allocated']:.2f}GiB"
        )
    log_resource(msg)
    if disk > DISK_LIMIT_GIB:
        raise RuntimeError(f"DISK LIMIT EXCEEDED: {disk:.2f} > {DISK_LIMIT_GIB:.2f} GiB")
    if ram > RAM_WARN_GIB:
        resource_logger.warning(f"[ram] High RAM usage: {ram:.1f} GiB")
    if torch.cuda.is_available():
        allocated = gpu["allocated"]
        reserved = gpu["reserved"]
        if reserved > 0.1 and allocated / reserved < 0.6:
            resource_logger.warning(
                "[cuda] High memory fragmentation detected "
                f"(allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB)"
            )


def budget_ok(reserve_minutes: float = 5.0) -> bool:
    reserve_secs = reserve_minutes * 60.0
    elapsed = time.time() - SESSION_START_TIME  # type: ignore[name-defined]
    return elapsed < (SESSION_BUDGET_SECS - reserve_secs)
