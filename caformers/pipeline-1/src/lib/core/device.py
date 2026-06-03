from __future__ import annotations

import os
import torch

from lib.config.defaults import TRAIN_CFG
from lib.config.defaults import MAX_CPU_CORES

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE_TYPE = DEVICE.type
NUM_DEVICES = torch.cuda.device_count() if DEVICE_TYPE == "cuda" else 0
USE_AMP = DEVICE_TYPE == "cuda" and bool(TRAIN_CFG["use_amp"])
USE_COMPILE = bool(TRAIN_CFG["use_compile"])
AMP_DTYPE = (
    torch.bfloat16
    if DEVICE_TYPE == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16
)
_cpu_count = os.cpu_count() or 2
usable_cores = min(_cpu_count, MAX_CPU_CORES)
# Keep a small default cap for workers per GPU, but respect user's cap
NUM_WORKERS = min(4, usable_cores) if DEVICE_TYPE == "cuda" else 0
try:
    torch.set_num_threads(usable_cores)
except Exception:
    pass
PIN_MEMORY = DEVICE_TYPE == "cuda"

if DEVICE_TYPE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
