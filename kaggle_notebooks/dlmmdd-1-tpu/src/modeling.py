from __future__ import annotations

import gc
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from .augmentations import build_tf_dataset
from .checkpoints import CheckpointManager, CheckpointSelector
from .config import (
    CHECKPOINT_KEEP_TOP_K,
    DEVICE_TYPE,
    EARLY_STOP_PATIENCE,
    FINAL_MODELS_DIR,
    LABEL_SMOOTHING,
    LOG_DIR,
    MODEL_REGISTRY,
    NUM_CLASSES,
    NUM_EPOCHS,
    AUTO,
    PLATEAU_FACTOR,
    PLATEAU_MIN_LR,
    PLATEAU_PATIENCE,
    RUN_TRAINING,
    SEED,
    STRATEGY,
    USE_LR_PLATEAU,
)
from .metrics import MetricsComputer
from .utils import log_resources, save_json, time_budget_ok


def build_model(model_name, num_classes, dropout=0.2, image_size=224):
    registry = MODEL_REGISTRY.get(model_name)
    if registry is None:
        raise ValueError(f'Unknown model: {model_name}')

    builder = registry['builder']
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name='image')
    x = builder(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs, name=model_name)


class EpochCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, fold_dir, train_eval_ds, val_ds, y_train, y_val, model_name, model_cfg):
        super().__init__()
        self.fold_dir = Path(fold_dir)
        self.train_eval_ds = train_eval_ds
        self.val_ds = val_ds
        self.y_train = np.asarray(y_train, dtype=np.int64)
        self.y_val = np.asarray(y_val, dtype=np.int64)
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.history = []
        self.saved = []
        self.best_val_f1 = 0.0
        self.no_improve = 0
        self.fold_dir.mkdir(parents=True, exist_ok=True)

    def _predict(self, ds):
        probs = self.model.predict(ds, verbose=0)
        return np.asarray(probs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_probs = self._predict(self.train_eval_ds)
        val_probs = self._predict(self.val_ds)
        train_pred = train_probs.argmax(axis=1)
        val_pred = val_probs.argmax(axis=1)

        train_m = MetricsComputer.compute_epoch_metrics(self.y_train, train_pred)
        val_m = MetricsComputer.compute_epoch_metrics(self.y_val, val_pred)
        train_m['loss'] = float(logs.get('loss')) if logs.get('loss') is not None else None
        val_m['loss'] = float(logs.get('val_loss')) if logs.get('val_loss') is not None else None
        val_m['generalization_score'] = MetricsComputer.generalization_score(train_m, val_m)
        selection_value = CheckpointSelector.get_selection_metric_value(val_m)

        checkpoint_path = self.fold_dir / f'epoch_{epoch:03d}.keras'
        self.model.save(checkpoint_path)
        save_json(
            {
                'model_name': self.model_name,
                'fold_idx': int(self.model_cfg['fold_idx']),
                'epoch': int(epoch),
                'image_size': int(self.model_cfg['image_size']),
                'batch_size': int(self.model_cfg['batch_size']),
                'train_metrics': train_m,
                'val_metrics': val_m,
                'selection_value': float(selection_value),
            },
            checkpoint_path.with_suffix('.json'),
        )

        self.saved.append({'epoch': epoch, 'checkpoint_path': str(checkpoint_path), 'selection_value': selection_value})
        self.saved = CheckpointManager.prune_to_top_k_checkpoints(self.saved, CHECKPOINT_KEEP_TOP_K)

        self.history.append({
            'epoch': int(epoch),
            'checkpoint_path': str(checkpoint_path),
            'train': train_m,
            'val': val_m,
            'learning_rate': float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),
            'selection_value': float(selection_value),
        })

        if val_m['f1_macro'] > self.best_val_f1 + 1e-4:
            self.best_val_f1 = val_m['f1_macro']
            self.no_improve = 0
        else:
            self.no_improve += 1
            if self.no_improve >= EARLY_STOP_PATIENCE:
                self.model.stop_training = True


def _make_optimizer(lr):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def train_fold(fold_idx, fold_info, train_df, model_name, mcfg, arch_dir):
    print(f"\n{'=' * 68}")
    print(f"  MODEL={model_name}  FOLD={fold_idx}  train={fold_info['train_count']}  val={fold_info['val_count']}")
    print(f"{'=' * 68}")

    if 'full_path' not in train_df.columns:
        raise ValueError('train_df must include a full_path column before training')

    fold_dir = Path(arch_dir) / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(mcfg['image_size'])
    batch_size = int(mcfg['batch_size'])
    lr = float(mcfg['lr'])
    dropout = float(mcfg['dropout'])

    train_rows = train_df.iloc[fold_info['train_indices']].reset_index(drop=True)
    val_rows = train_df.iloc[fold_info['val_indices']].reset_index(drop=True)
    x_train = train_rows['full_path'].astype(str).to_numpy()
    y_train = train_rows['y'].astype(np.int64).to_numpy()
    x_val = val_rows['full_path'].astype(str).to_numpy()
    y_val = val_rows['y'].astype(np.int64).to_numpy()

    train_ds = build_tf_dataset(x_train, y_train, image_size=image_size, batch_size=batch_size, training=True)
    train_eval_ds = build_tf_dataset(
        x_train,
        y_train,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        cache=str(fold_dir / 'train_eval.cache'),
    )
    val_ds = build_tf_dataset(
        x_val,
        y_val,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        cache=str(fold_dir / 'val.cache'),
    )

    with STRATEGY.scope():
        model = build_model(model_name, NUM_CLASSES, dropout=dropout, image_size=image_size)
        model.compile(
            optimizer=_make_optimizer(lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
            jit_compile=(DEVICE_TYPE == 'tpu'),
            steps_per_execution=16 if DEVICE_TYPE == 'tpu' else 1,
        )

    callback = EpochCheckpointCallback(
        fold_dir=fold_dir,
        train_eval_ds=train_eval_ds,
        val_ds=val_ds,
        y_train=y_train,
        y_val=y_val,
        model_name=model_name,
        model_cfg={**mcfg, 'fold_idx': fold_idx, 'image_size': image_size, 'batch_size': batch_size},
    )

    callbacks = [callback]
    if USE_LR_PLATEAU:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                mode='max',
                patience=PLATEAU_PATIENCE,
                factor=PLATEAU_FACTOR,
                min_lr=PLATEAU_MIN_LR,
                verbose=1,
            )
        )

    if not time_budget_ok():
        print('  [TIME BUDGET] Skipping fold before training starts.')
        return []

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    history = callback.history
    save_json(history, LOG_DIR / f'{model_name[:30]}_history_fold_{fold_idx}.json')

    if history:
        best = max(history, key=lambda row: row['selection_value'])
        best_src = Path(best['checkpoint_path'])
        best_dst_dir = FINAL_MODELS_DIR / model_name
        best_dst_dir.mkdir(parents=True, exist_ok=True)
        best_dst = best_dst_dir / f'fold_{fold_idx}_best.keras'
        shutil.copy2(best_src, best_dst)
        shutil.copy2(best_src.with_suffix('.json'), best_dst.with_suffix('.json'))
        print(f"\n  Done. Best E={best['epoch']}  val_f1={best['val']['f1_macro']:.4f}  gen={best['val']['generalization_score']:.4f}")

    log_resources(f'after fold {fold_idx}')
    gc.collect()
    return history
