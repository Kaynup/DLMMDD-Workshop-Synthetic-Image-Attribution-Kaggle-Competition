from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from lib.config.defaults import (
    CHECKPOINT_KEEP_TOP_K,
    CHECKPOINT_SELECTION_METRIC,
    EARLY_STOP_PATIENCE,
    GRAD_ACCUM_STEPS,
    NUM_CLASSES,
    NUM_EPOCHS,
    SEED,
    TRAIN_EVAL_EACH_EPOCH,
    USE_LR_PLATEAU,
)
from lib.config.paths import FINAL_DIR, LOG_DIR
from lib.core.device import AMP_DTYPE, DEVICE, PIN_MEMORY, USE_AMP
from lib.core.logging import (
    checkpoint_logger,
    log_checkpoint,
    log_metric,
    log_resource,
    log_training,
    training_logger,
)
from lib.core.resources import budget_ok, log_resources
from lib.core.seed import set_seed
from lib.core.utils import ensure_dir, save_json
from lib.data.loader import make_loader
from lib.models.factory import build_model
from lib.training.checkpointing import prune_checkpoints, save_checkpoint
from lib.training.checkpointing import load_checkpoint
from lib.training.evaluation import evaluate_loader
from lib.training.losses import cross_entropy_soft, smooth_one_hot
from lib.training.metrics import compute_metrics, generalization_score, get_selection_value
from lib.training.regularization import apply_batch_regularization


def train_fold(
    fold_idx: int,
    fold_info: dict,
    train_df,
    model_name: str,
    mcfg: dict,
    arch_ckpt_dir: Path,
) -> list[dict]:
    set_seed(SEED + fold_idx)
    cfg = mcfg
    reg = cfg["regularization"]
    log_training(
        training_logger,
        f"[fold {fold_idx}] START | model={model_name} | img={cfg['image_size']} | "
        f"bs={cfg['batch_size']} | lr={cfg['lr']:.1e} | wd={reg['weight_decay']} | "
        f"train={fold_info['train_count']} | val={fold_info['val_count']}"
    )
    img_sz = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    weight_decay = float(reg.get("weight_decay", 0.0))
    label_smoothing = float(reg.get("label_smoothing", 0.0))
    grad_clip_norm = float(reg.get("grad_clip_norm", 1.0))
    grad_accum_steps = max(1, int(GRAD_ACCUM_STEPS))
    fold_dir = ensure_dir(arch_ckpt_dir / f"fold_{fold_idx}")
    tr_rows = train_df.iloc[fold_info["train_indices"]].reset_index(drop=True)
    va_rows = train_df.iloc[fold_info["val_indices"]].reset_index(drop=True)
    x_tr = tr_rows["full_path"].astype(str).to_numpy()
    y_tr = tr_rows["y"].astype(np.int64).to_numpy()
    x_va = va_rows["full_path"].astype(str).to_numpy()
    y_va = va_rows["y"].astype(np.int64).to_numpy()
    train_loader = make_loader(
        x_tr,
        y_tr,
        image_size=img_sz,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        fold_seed=SEED + fold_idx,
    )
    val_loader = make_loader(
        x_va,
        y_va,
        image_size=img_sz,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        fold_seed=SEED + fold_idx,
    )
    train_eval_loader = None
    if TRAIN_EVAL_EACH_EPOCH:
        train_eval_loader = make_loader(
            x_tr,
            y_tr,
            image_size=img_sz,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            fold_seed=SEED + fold_idx,
        )
    model = build_model(model_name, mcfg=mcfg)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if USE_LR_PLATEAU:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.get("plateau_factor", 0.5)),
            patience=int(cfg.get("plateau_patience", 3)),
            min_lr=float(cfg.get("plateau_min_lr", 1e-7)),
        )
    criterion = cross_entropy_soft
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    history: list[dict] = []
    saved_ckpts: list[dict] = []
    best_sel_value = -float("inf")
    no_improve_cnt = 0
    # Resume support: look for latest checkpoint in fold dir
    start_epoch = 0
    try:
        fold_dir = ensure_dir(arch_ckpt_dir / f"fold_{fold_idx}")
        ckpts = list(fold_dir.glob("epoch_*.pt"))
        if ckpts:
            # pick highest epoch by filename
            def epoch_from_path(p):
                try:
                    return int(p.stem.split("_")[1])
                except Exception:
                    return int(p.stat().st_mtime)

            latest = sorted(ckpts, key=epoch_from_path)[-1]
            model, meta = load_checkpoint(latest, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, strict=False)
            recovered_epoch = int(meta.get("epoch", -1))
            start_epoch = recovered_epoch + 1
            log_training(training_logger, f"[resume] recovered from {latest.name} epoch={recovered_epoch} starting at epoch={start_epoch}")
    except Exception as exc:
        log_training(training_logger, f"[resume] no checkpoint recovered: {exc}")
    for epoch in range(start_epoch, NUM_EPOCHS):
        if not budget_ok():
            log_training(training_logger, "[budget] Time limit reached — stopping training.")
            break
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        grad_norms: list[float] = []
        processed_images = 0
        all_train_preds: list[np.ndarray] = []
        all_train_labels: list[np.ndarray] = []
        for step, (imgs, lbls) in enumerate(train_loader):
            imgs = imgs.to(DEVICE, non_blocking=True).contiguous(memory_format=torch.channels_last)
            lbls = lbls.to(DEVICE, non_blocking=True)
            soft_targets = smooth_one_hot(lbls, NUM_CLASSES, label_smoothing)
            imgs, soft_targets = apply_batch_regularization(imgs, soft_targets, reg)
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, soft_targets) / grad_accum_steps
            scaler.scale(loss).backward()
            do_step = (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader)
            if do_step:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm).item())
                grad_norms.append(grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = int((preds == lbls).sum().item())
            all_train_preds.append(preds.cpu().numpy())
            all_train_labels.append(lbls.cpu().numpy())
            bs = int(imgs.size(0))
            processed_images += bs
            running_seen += bs
            running_correct += correct
            running_loss += float(loss.item()) * bs * grad_accum_steps
            loss_val = float(loss.item()) * grad_accum_steps
            if not np.isfinite(loss_val):
                training_logger.warning(f"[train] non-finite loss epoch={epoch} step={step} loss={loss_val}")
            if step % 50 == 0:
                train_acc = running_correct / max(running_seen, 1)
                log_training(
                    training_logger,
                    f"epoch={epoch:02d} step={step:04d} loss={loss_val:.4f} acc={train_acc:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )
            del imgs, lbls, soft_targets, logits, preds, loss
        train_loss = running_loss / max(running_seen, 1)
        train_acc = running_correct / max(running_seen, 1)
        if TRAIN_EVAL_EACH_EPOCH and train_eval_loader is not None:
            train_eval = evaluate_loader(model, train_eval_loader, criterion=criterion)
            train_m = train_eval.copy()
        else:
            train_m = compute_metrics(np.concatenate(all_train_labels), np.concatenate(all_train_preds))
            train_m["accuracy"] = float(train_acc)
            train_m["loss"] = float(train_loss)
        val_eval = evaluate_loader(model, val_loader, criterion=criterion)
        val_m = val_eval.copy()
        for d in (train_m, val_m):
            d.pop("probs", None)
            d.pop("y_true", None)
            d.pop("y_pred", None)
        train_m["loss"] = float(train_loss)
        val_m["generalization_score"], gen_parts = generalization_score(train_m, val_m, return_parts=True)
        selection_value = get_selection_value(train_m, val_m, checkpoint_metric=CHECKPOINT_SELECTION_METRIC)
        per_class_f1 = val_eval.get("per_class_f1", [0.0] * NUM_CLASSES)
        worst3 = sorted(enumerate(per_class_f1), key=lambda t: t[1])[:3]
        epoch_time = time.time() - epoch_start
        imgs_per_sec = processed_images / max(epoch_time, 1e-6)
        train_gap = train_m["f1_macro"] - val_m["f1_macro"]
        epoch_summary_text = (
            f"\n{'─'*100}\n"
            f"Epoch {epoch:02d}/{NUM_EPOCHS - 1} | fold={fold_idx} | model={model_name}\n\n"
            f"TRAIN loss={train_m['loss']:.4f} | acc={train_m['accuracy']:.4f} | f1={train_m['f1_macro']:.4f}\n"
            f"VAL   loss={val_m['loss']:.4f} | acc={val_m['accuracy']:.4f} | f1={val_m['f1_macro']:.4f}\n\n"
            f"gap={train_gap:+.4f} | gen_score={val_m['generalization_score']:.4f} | selection={selection_value:.4f}\n\n"
            f"throughput={imgs_per_sec:.0f} img/s | lr={optimizer.param_groups[0]['lr']:.2e} | grad_clip={grad_clip_norm:.2f}\n\n"
            f"worst_classes={worst3}\n"
            f"{'─'*100}"
        )
        log_training(training_logger, epoch_summary_text)
        log_metric(
            json.dumps({
                "epoch": epoch,
                "fold": fold_idx,
                "train_loss": train_m["loss"],
                "train_acc": train_m["accuracy"],
                "train_f1": train_m["f1_macro"],
                "val_loss": val_m["loss"],
                "val_acc": val_m["accuracy"],
                "val_f1": val_m["f1_macro"],
                "generalization_score": val_m["generalization_score"],
                "selection_value": float(selection_value),
                "lr": optimizer.param_groups[0]["lr"],
            })
        )
        ckpt_path = fold_dir / f"epoch_{epoch:03d}.pt"
        meta = {
            "model_name": model_name,
            "fold_idx": fold_idx,
            "epoch": epoch,
            "image_size": img_sz,
            "batch_size": batch_size,
            "train_metrics": train_m,
            "val_metrics": val_m,
            "selection_metric": CHECKPOINT_SELECTION_METRIC,
            "selection_value": float(selection_value),
            "generalization_parts": gen_parts,
            "regularization": reg,
        }
        save_checkpoint(ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, meta=meta)
        saved_ckpts.append({"path": str(ckpt_path), "sv": float(selection_value)})
        saved_ckpts = prune_checkpoints(saved_ckpts, CHECKPOINT_KEEP_TOP_K)
        history.append({"epoch": epoch, "train": train_m, "val": val_m, "sv": float(selection_value), "generalization_parts": gen_parts})
        if scheduler is not None:
            scheduler.step(selection_value)
        if selection_value > best_sel_value + 1e-4:
            best_sel_value = selection_value
            no_improve_cnt = 0
            best_dst = ensure_dir(FINAL_DIR / model_name) / f"fold_{fold_idx}_best.pt"
            shutil.copy2(ckpt_path, best_dst)
            shutil.copy2(ckpt_path.with_suffix(".json"), best_dst.with_suffix(".json"))
            log_checkpoint(
                f"BEST_CHECKPOINT fold={fold_idx} epoch={epoch} selection_value={best_sel_value:.4f} -> {best_dst.name}"
            )
        else:
            no_improve_cnt += 1
            log_training(training_logger, f"[early_stop] no improvement {no_improve_cnt}/{EARLY_STOP_PATIENCE}")
            if no_improve_cnt >= EARLY_STOP_PATIENCE:
                log_training(training_logger, f"[early_stop] triggered at epoch={epoch}")
                break
        log_resources(f"fold={fold_idx} epoch={epoch}")
    save_json(history, LOG_DIR / f"{model_name[:30]}_fold{fold_idx}_history.json")
    del model, optimizer, scheduler, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    log_resources(f"after fold {fold_idx}")
    return history
