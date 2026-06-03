from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier

from lib.config.defaults import ACTIVE_MODELS, NUM_CLASSES
from lib.config.paths import FINAL_DIR
from lib.core.logging import inference_logger
from lib.core.utils import clear_gpu_memory
from lib.data.loader import make_loader
from lib.inference.predictor import load_model_from_ckpt
from lib.training.metrics import compute_metrics
from lib.training.evaluation import predict_loader


def make_stacking_learner(name: str):
    name = name.lower().strip()
    if name == "logreg":
        return LogisticRegression(max_iter=3000, solver="lbfgs")
    if name == "ridge":
        return RidgeClassifier()
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256,), activation="relu", max_iter=500, random_state=42)
    raise ValueError(f"Unknown stacking learner: {name}")


def predict_proba_from_learner(learner, x: np.ndarray) -> np.ndarray:
    if hasattr(learner, "predict_proba"):
        return np.asarray(learner.predict_proba(x), dtype=np.float32)
    scores = np.asarray(learner.decision_function(x), dtype=np.float32)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def run_stacking_ensemble(train_meta, test_meta, y_train, fold_metadata, stacking_learner: str):
    model_ckpts: dict[str, list[Path]] = {}
    for model_name in ACTIVE_MODELS:
        model_ckpts[model_name] = sorted((FINAL_DIR / model_name).glob("fold_*_best.pt"))
        inference_logger.info(f"[stacking] model={model_name} | ckpts={len(model_ckpts[model_name])}")
    n_train = len(train_meta)
    n_classes = NUM_CLASSES
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    test_paths = test_meta["full_path"].astype(str).to_numpy()
    for model_name, ckpts in model_ckpts.items():
        if not ckpts:
            raise RuntimeError(f"No checkpoints for stacking: {model_name}")
        model_oof = np.zeros((n_train, n_classes), dtype=np.float32)
        model_test_fold_probs: list[np.ndarray] = []
        for ckpt_path in ckpts:
            model, meta, loaded_name = load_model_from_ckpt(ckpt_path)
            fold_idx = int(meta.get("fold_idx", -1))
            if fold_idx not in fold_metadata:
                raise RuntimeError(f"Fold metadata missing for fold_idx={fold_idx}")
            img_sz = int(meta.get("image_size", model.config.image_size))
            batch_size = int(meta.get("batch_size", 32))
            va_idx = fold_metadata[fold_idx]["val_indices"]
            va_rows = train_meta.iloc[va_idx].reset_index(drop=True)
            va_loader = make_loader(
                va_rows["full_path"].astype(str).to_numpy(),
                va_rows["y"].astype(np.int64).to_numpy(),
                image_size=img_sz,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
            _, _, va_probs = predict_loader(model, va_loader)
            model_oof[va_idx] = va_probs.astype(np.float32)
            inference_logger.info(f"[stacking] OOF filled | model={model_name} | fold={fold_idx} | shape={va_probs.shape}")
            test_loader = make_loader(test_paths, image_size=img_sz, batch_size=batch_size, shuffle=False, drop_last=False)
            _, _, te_probs = predict_loader(model, test_loader)
            model_test_fold_probs.append(te_probs.astype(np.float32))
            del model
            clear_gpu_memory()
        model_test_avg = np.mean(np.stack(model_test_fold_probs, axis=0), axis=0).astype(np.float32)
        train_blocks.append(model_oof)
        test_blocks.append(model_test_avg)
        inference_logger.info(f"[stacking] model={model_name} | train_block={model_oof.shape} | test_block={model_test_avg.shape}")
    X_train = np.concatenate(train_blocks, axis=1)
    X_test = np.concatenate(test_blocks, axis=1)
    inference_logger.info(f"[stacking] X_train={X_train.shape} | X_test={X_test.shape}")
    stacker = make_stacking_learner(stacking_learner)
    inference_logger.info(f"[stacking] fitting learner={stacking_learner}")
    stacker.fit(X_train, y_train)
    ensemble = predict_proba_from_learner(stacker, X_test).astype(np.float32)
    preds = ensemble.argmax(axis=1).astype(np.int64)
    conf = ensemble.max(axis=1).astype(np.float32)
    inference_logger.info(
        f"[stacking] learner={stacking_learner} | models={len(model_ckpts)} | "
        f"mean_conf={conf.mean():.4f} | min={conf.min():.4f} | max={conf.max():.4f}"
    )
    return ensemble, preds, conf
