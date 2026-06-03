from __future__ import annotations

from lib.config.defaults import ACTIVE_MODELS, RUN_TRAINING
from lib.config.paths import LOG_DIR
from lib.core.logging import pipeline_logger
from lib.core.utils import save_json
from lib.core.resources import budget_ok
from lib.data.loader import make_loader
from lib.training.train_fold import train_fold


def run_training_pipeline(train_df, test_df, y_train, fold_metadata):
    if not RUN_TRAINING:
        pipeline_logger.info("RUN_TRAINING is disabled.")
        return {}
    all_history: dict[str, dict] = {}
    for model_name in ACTIVE_MODELS:
        if not budget_ok():
            pipeline_logger.warning("[budget] session budget exhausted")
            break
        mcfg = __import__("lib.config.model_registry", fromlist=["MODEL_REGISTRY"]).MODEL_REGISTRY[model_name]
        arch_dir = __import__("lib.config.paths", fromlist=["CHECKPOINT_DIR"]).CHECKPOINT_DIR / model_name
        model_hist: dict[int, list[dict]] = {}
        for fold_idx, fold_info in fold_metadata.items():
            if not budget_ok():
                pipeline_logger.warning("[budget] session budget exhausted")
                break
            fold_history = train_fold(
                fold_idx=fold_idx,
                fold_info=fold_info,
                train_df=train_df,
                model_name=model_name,
                mcfg=mcfg,
                arch_ckpt_dir=arch_dir,
            )
            model_hist[fold_idx] = fold_history
            __import__("lib.core.utils", fromlist=["clear_gpu_memory"]).clear_gpu_memory()
        all_history[model_name] = model_hist
        save_json(model_hist, LOG_DIR / f"{model_name[:30]}_all_folds.json")
    save_json(all_history, LOG_DIR / "full_training_history.json")
    return all_history
