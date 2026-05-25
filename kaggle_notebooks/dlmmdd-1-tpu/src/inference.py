from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from .data import build_tf_dataset
from .config import (
    ACTIVE_MODELS,
    BLEND_MODE,
    FINAL_MODELS_DIR,
    INFERENCE_DIR,
    INFERENCE_ONLY_PATH,
    MODEL_REGISTRY,
    NUM_CLASSES,
    RUN_INFERENCE_ONLY,
    SEED,
    STACKING_LEARNER,
    TTA_N,
    USE_TTA,
    WORKING_ROOT,
)
from .metrics import MetricsComputer
from .utils import log_resources, save_json


def load_model_from_ckpt(ckpt_path):
    ckpt_path = Path(ckpt_path)
    model = tf.keras.models.load_model(ckpt_path)
    meta_path = ckpt_path.with_suffix('.json')
    metadata = {}
    if meta_path.exists():
        metadata = pd.read_json(meta_path, typ='series').to_dict()
    model_name = metadata.get('model_name', ACTIVE_MODELS[0])
    val_metrics = metadata.get('val_metrics', {})
    print(
        f'  [load] {ckpt_path.name}  arch={model_name}  epoch={metadata.get("epoch", "?")}  '
        f'val_f1={val_metrics.get("f1_macro", float("nan")):.4f}  gen={val_metrics.get("generalization_score", float("nan")):.4f}'
    )
    return model, metadata, model_name


def infer_single_pass(model, dataset):
    probs = model.predict(dataset, verbose=0)
    return np.asarray(probs)


def infer_with_tta(model, paths, img_sz, batch_size, n_tta=4):
    cache_path = INFERENCE_DIR / f'test_{img_sz}.cache'
    clean_ds = build_tf_dataset(paths, labels=None, image_size=img_sz, batch_size=batch_size, cache=str(cache_path))
    probs = infer_single_pass(model, clean_ds)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    def _augment_batch(images):
        return augmentation(images, training=True)

    for idx in range(n_tta):
        tta_ds = build_tf_dataset(paths, labels=None, image_size=img_sz, batch_size=batch_size, cache=str(cache_path))
        tta_ds = tta_ds.map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        probs += infer_single_pass(model, tta_ds)
        print(f'    TTA {idx + 1}/{n_tta}')

    return probs / float(n_tta + 1)


def _predict_class_probabilities(model, paths, img_sz, batch_size):
    dataset = build_tf_dataset(paths, labels=None, image_size=img_sz, batch_size=batch_size, cache=str(INFERENCE_DIR / f'test_{img_sz}.cache'))
    return infer_single_pass(model, dataset)


def run_ensemble_inference(train_meta, test_meta, y_train, fold_metadata):
    if test_meta.empty:
        print('test_meta empty -- skipped.')
        return None

    print(f'\nRunning inference on {len(test_meta)} test images ...')
    log_resources('inference start')

    if RUN_INFERENCE_ONLY:
        ckpt_paths = sorted(Path(INFERENCE_ONLY_PATH).glob('**/*best.keras'))
    else:
        ckpt_paths = sorted(FINAL_MODELS_DIR.rglob('fold_*_best.keras'))

    print(f'Found {len(ckpt_paths)} checkpoint(s).')

    all_probs = []
    all_weights = []
    all_oof = []

    test_paths = test_meta['full_path'].astype(str).to_numpy()

    for checkpoint_path in ckpt_paths:
        model, metadata, model_name = load_model_from_ckpt(checkpoint_path)
        img_sz = int(metadata.get('image_size', MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[ACTIVE_MODELS[0]])['image_size']))
        batch_size = int(metadata.get('batch_size', MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[ACTIVE_MODELS[0]])['batch_size']))

        if USE_TTA:
            probs = infer_with_tta(model, test_paths, img_sz, batch_size, TTA_N)
        else:
            probs = _predict_class_probabilities(model, test_paths, img_sz, batch_size)

        all_probs.append(probs)
        weight = metadata.get('val_metrics', {}).get('f1_macro', metadata.get('val_metrics', {}).get('accuracy', 1.0))
        all_weights.append(weight)
        print(f'  [weight] {checkpoint_path.name}  f1={weight:.4f}')

        if BLEND_MODE == 'stacking':
            match = re.search(r'fold_(\d+)', checkpoint_path.stem)
            fold_idx = int(match.group(1)) if match else -1
            if fold_idx >= 0 and fold_idx in fold_metadata:
                val_idx = fold_metadata[fold_idx]['val_indices']
                val_df = train_meta.iloc[val_idx].reset_index(drop=True)
                val_paths = val_df['full_path'].astype(str).to_numpy()
                val_ds = build_tf_dataset(val_paths, labels=None, image_size=img_sz, batch_size=batch_size, cache=str(INFERENCE_DIR / f'fold_{fold_idx}_{img_sz}.cache'))
                oof_probs = infer_single_pass(model, val_ds)
                all_oof.append((fold_idx, val_idx, oof_probs))

        del model
        gc.collect()

    if not all_probs:
        print('No checkpoints found.')
        return None

    stacked_probs = np.stack(all_probs, 0)
    weights = np.asarray(all_weights, np.float32)
    weights = np.nan_to_num(weights, nan=1e-6)
    weights = np.maximum(weights, 1e-6)

    if BLEND_MODE == 'simple_avg':
        print('\n[ensemble] simple average')
        ensemble = stacked_probs.mean(0)
    elif BLEND_MODE == 'weighted_avg':
        weights = weights / weights.sum()
        print(f'\n[ensemble] weighted average  w={weights.round(3).tolist()}')
        ensemble = np.average(stacked_probs, 0, weights=weights)
    elif BLEND_MODE == 'rank_avg':
        print('\n[ensemble] rank average')
        from scipy.stats import rankdata

        ranked = np.stack([np.apply_along_axis(rankdata, 1, stacked_probs[idx]) for idx in range(stacked_probs.shape[0])], 0)
        ensemble = ranked.mean(0)
    elif BLEND_MODE == 'stacking':
        print(f'\n[ensemble] stacking meta-learner={STACKING_LEARNER}')
        if not all_oof:
            print('  [stacking] no OOF data -- falling back to weighted_avg')
            weights = weights / weights.sum()
            ensemble = np.average(stacked_probs, 0, weights=weights)
        else:
            n_train = len(train_meta)
            n_feat = len(all_oof) * NUM_CLASSES
            X_oof = np.zeros((n_train, n_feat), np.float32)
            X_test_stack = np.zeros((stacked_probs.shape[1], n_feat), np.float32)
            for idx, (fold_idx, val_idx, oof_probs) in enumerate(all_oof):
                sl = slice(idx * NUM_CLASSES, (idx + 1) * NUM_CLASSES)
                X_oof[val_idx, sl] = oof_probs
                X_test_stack[:, sl] = stacked_probs[idx]
            y_oof = y_train

            if STACKING_LEARNER == 'logreg':
                meta = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial', random_state=SEED)
            elif STACKING_LEARNER == 'ridge':
                meta = RidgeClassifier(alpha=1.0)
            elif STACKING_LEARNER == 'mlp':
                meta = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500, random_state=SEED, early_stopping=True)
            else:
                raise ValueError(f'Unknown STACKING_LEARNER: {STACKING_LEARNER}')

            print(f'  Fitting {STACKING_LEARNER} on {X_oof.shape} OOF features...')
            meta.fit(X_oof, y_oof)
            meta_f1 = f1_score(y_oof, meta.predict(X_oof), average='macro', zero_division=0)
            print(f'  OOF train f1_macro={meta_f1:.4f}')

            if hasattr(meta, 'predict_proba'):
                ensemble = meta.predict_proba(X_test_stack)
            else:
                decision = meta.decision_function(X_test_stack)
                ensemble = np.exp(decision) / np.exp(decision).sum(1, keepdims=True)

            save_json({'learner': STACKING_LEARNER, 'oof_f1_macro': float(meta_f1)}, INFERENCE_DIR / 'stacking_meta.json')
    else:
        raise ValueError(f'Unknown BLEND_MODE: {BLEND_MODE}')

    preds = ensemble.argmax(1).astype(int)
    conf = ensemble.max(1)

    submission_df = test_meta[['ID']].copy()
    submission_df['TARGET'] = preds
    submission_df.to_csv(WORKING_ROOT / 'submission.csv', index=False)
    submission_df.to_csv(INFERENCE_DIR / 'submission.csv', index=False)

    pd.DataFrame({'ID': test_meta['ID'].values, 'predicted_class': preds, 'confidence': conf}).to_csv(INFERENCE_DIR / 'prediction_confidence.csv', index=False)

    save_json({
        'timestamp': pd.Timestamp.utcnow().isoformat(),
        'num_models': len(ckpt_paths),
        'blend_mode': BLEND_MODE,
        'tta_passes': TTA_N if USE_TTA else 0,
        'mean_confidence': float(conf.mean()),
        'min_confidence': float(conf.min()),
        'max_confidence': float(conf.max()),
        'num_predictions': len(preds),
    }, INFERENCE_DIR / 'submission_metadata.json')

    print('\nSubmission preview:')
    print(submission_df.head(10).to_string(index=False))
    print(f'Avg confidence: {conf.mean():.4f}  Min: {conf.min():.4f}  Max: {conf.max():.4f}')
    print(pd.Series(preds).value_counts().sort_index().to_string())
    log_resources('inference done')
    return submission_df
