from __future__ import annotations

import numpy as np
from pathlib import Path

from lib.config.defaults import BLEND_MODE, BLEND_WEIGHT_METRIC, RUN_INFERENCE_ONLY, USE_TTA, TTA_N, TRAIN_CFG
from lib.config.paths import FINAL_DIR, INFERENCE_ONLY_PATH
from lib.core.logging import inference_logger
from lib.core.utils import clear_gpu_memory
from lib.data.loader import make_loader
from lib.inference.predictor import infer_single, load_model_from_ckpt
from lib.inference.stacking import run_stacking_ensemble
from lib.inference.tta import infer_tta


def build_weighting_value(meta: dict) -> float:
    if BLEND_WEIGHT_METRIC == "val_f1_macro":
        return float(meta.get("val_metrics", {}).get("f1_macro", meta.get("selection_value", 0.0)))
    if BLEND_WEIGHT_METRIC == "accuracy":
        return float(meta.get("val_metrics", {}).get("accuracy", meta.get("selection_value", 0.0)))
    return float(meta.get("selection_value", meta.get("val_metrics", {}).get("f1_macro", 0.0)))


def run_ensemble_inference(train_meta, test_meta, y_train, fold_metadata):
    inference_logger.info("=" * 96)
    inference_logger.info("STAGE 6 — INFERENCE / TEST-SET PREDICTION")
    inference_logger.info("=" * 96)
    inference_logger.info(f"[blend] mode={BLEND_MODE} | weight_metric={BLEND_WEIGHT_METRIC}")
    if test_meta.empty:
        inference_logger.warning("test_meta is empty — skipping inference.")
        return None
    if RUN_INFERENCE_ONLY:
        ckpt_paths = sorted(Path(INFERENCE_ONLY_PATH).glob("**/*best.pt"))
    else:
        ckpt_paths = sorted(FINAL_DIR.rglob("fold_*_best.pt"))
    inference_logger.info(f"[inference] found {len(ckpt_paths)} checkpoint(s)")
    if not ckpt_paths:
        inference_logger.error("No checkpoints found.")
        return None
    if BLEND_MODE in {"simple_avg", "weighted_avg"}:
        all_probs: list[np.ndarray] = []
        all_weights: list[float] = []
        test_paths = test_meta["full_path"].astype(str).to_numpy()
        for idx, ckpt_path in enumerate(ckpt_paths):
            model, meta, model_name = load_model_from_ckpt(ckpt_path)
            img_sz = int(meta.get("image_size", 224))
            batch_size = int(meta.get("batch_size", 32))
            if USE_TTA:
                probs = infer_tta(model=model, paths=test_paths, img_sz=img_sz, batch_size=batch_size, n=TTA_N)
            else:
                loader = make_loader(test_paths, image_size=img_sz, batch_size=batch_size, shuffle=False, drop_last=False)
                probs = infer_single(model, loader)
            inference_logger.info(f"[predict] ckpt={idx} | model={model_name} | shape={probs.shape}")
            all_probs.append(probs.astype(np.float32))
            weight = max(float(build_weighting_value(meta)), 1e-6)
            all_weights.append(weight)
            inference_logger.info(f"[weight] value={weight:.6f} | selection={meta.get('selection_value', float('nan')):.6f}")
            del model
            clear_gpu_memory()
        stacked = np.stack(all_probs, axis=0)
        weights = np.asarray(all_weights, dtype=np.float32)
        weights = np.nan_to_num(weights, nan=1e-6, posinf=1e-6, neginf=1e-6)
        if BLEND_MODE == "simple_avg":
            ensemble = stacked.mean(axis=0)
            inference_logger.info("[ensemble] simple_avg applied")
        else:
            weights = weights / max(weights.sum(), 1e-6)
            inference_logger.info(f"[ensemble] weighted_avg weights={weights.round(4).tolist()}")
            ensemble = np.average(stacked, axis=0, weights=weights)
        ensemble = ensemble.astype(np.float32)
        preds = ensemble.argmax(axis=1).astype(np.int64)
        conf = ensemble.max(axis=1).astype(np.float32)
        inference_logger.info(
            f"[ensemble] models={len(all_probs)} | mean_conf={conf.mean():.4f} | "
            f"min={conf.min():.4f} | max={conf.max():.4f}"
        )
        return ensemble, preds, conf, test_meta, len(ckpt_paths)
    return run_stacking_ensemble(train_meta, test_meta, y_train, fold_metadata, TRAIN_CFG["blend"]["stacking_learner"])
