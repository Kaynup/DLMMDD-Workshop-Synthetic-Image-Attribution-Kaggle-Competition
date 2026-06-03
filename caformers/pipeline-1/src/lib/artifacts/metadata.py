from __future__ import annotations

from lib.config.defaults import BLEND_MODE, TTA_N, USE_TTA
from lib.core.device import DEVICE_TYPE


def build_submission_metadata(preds, conf, num_models: int) -> dict:
    return {
        "timestamp": __import__("pandas", fromlist=["Timestamp"]).Timestamp.now("UTC").isoformat(),
        "num_models": int(num_models),
        "blend_mode": str(BLEND_MODE),
        "tta_enabled": bool(USE_TTA),
        "tta_passes": int(TTA_N if USE_TTA else 0),
        "num_predictions": int(len(preds)),
        "mean_confidence": float(conf.mean()),
        "std_confidence": float(conf.std()),
        "min_confidence": float(conf.min()),
        "max_confidence": float(conf.max()),
        "prediction_distribution": {str(k): int(v) for k, v in __import__("pandas", fromlist=["Series"]).Series(preds).value_counts().sort_index().items()},
        "device_type": DEVICE_TYPE,
        "amp_enabled": bool(USE_TTA),
        "active_models": list(__import__("lib.config.model_registry", fromlist=["ACTIVE_MODELS"]).ACTIVE_MODELS),
    }
