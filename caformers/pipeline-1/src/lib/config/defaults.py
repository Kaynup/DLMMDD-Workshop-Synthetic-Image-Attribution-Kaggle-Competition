from __future__ import annotations

CONTROL_PANEL: dict[str, dict] = {
    "training": {
        "seed": 42,
        "num_classes": 10,
        "num_folds": 1,
        "num_epochs": 30,
        "checkpoint_selection_metric": "generalization_score",
        "early_stop_patience": 3,
        "checkpoint_keep_top_k": 4,
        "use_lr_plateau": True,
        "plateau_patience": 3,
        "plateau_factor": 0.5,
        "plateau_min_lr": 1e-7,
        "use_amp": True,
        "use_compile": False,
        "grad_accum_steps": 1,
        "deterministic": False,
    },
    "regularization_defaults": {
        "weight_decay": 1e-4,
        "label_smoothing": 0.09,
        "mixup_alpha": 0.0,
        "mixup_prob": 0.0,
        "cutmix_alpha": 0.0,
        "cutmix_prob": 0.0,
        "dropout": 0.20,
        "grad_clip_norm": 1.0,
    },
    "blend": {
        "mode": "stacking",
        "weight_metric": "selection_value",
        "stacking_learner": "logreg",
        "use_tta": False,
        "tta_n": 4,
    },
    "monitoring": {
        "verbose_training": True,
        "train_eval_each_epoch": False,
        "print_cuda_memory_every": 0,
    },
    "generalization": {
        "val_weight": 1.0,
        "low_train_reward": 0.15,
        "overfit_penalty": 2.0,
        "balance_penalty": 0.5,
    },
    "resources": {
        "session_budget_secs": 8.5 * 3600,
        "disk_limit_gib": 17.0,
        "ram_warn_gib": 26.0,
        "max_cpu_cores": 12,
        "max_gpu_vram_gib": 6.0,
    },
    "runtime": {
        "run_training": True,
        "run_inference_only": False,
        "inference_only_path": "/kaggle/input/models/punyakdei/pipe-1/pytorch/default/1",
    },
    "extra_verbose": {
        "print_model": False,
    },
}

TRAIN_CFG = CONTROL_PANEL["training"]
BLEND_CFG = CONTROL_PANEL["blend"]
MONITOR_CFG = CONTROL_PANEL["monitoring"]
RESOURCE_CFG = CONTROL_PANEL["resources"]
RUNTIME_CFG = CONTROL_PANEL["runtime"]
EXTRA_VERBOSE = CONTROL_PANEL["extra_verbose"]

SEED = int(TRAIN_CFG["seed"])
NUM_CLASSES = int(TRAIN_CFG["num_classes"])
NUM_FOLDS = int(TRAIN_CFG["num_folds"])
NUM_EPOCHS = int(TRAIN_CFG["num_epochs"])

SESSION_BUDGET_SECS = float(RESOURCE_CFG["session_budget_secs"])
DISK_LIMIT_GIB = float(RESOURCE_CFG["disk_limit_gib"])
RAM_WARN_GIB = float(RESOURCE_CFG["ram_warn_gib"])
MAX_CPU_CORES = int(RESOURCE_CFG.get("max_cpu_cores", 12))
MAX_GPU_VRAM_GIB = float(RESOURCE_CFG.get("max_gpu_vram_gib", 6.0))

RUN_TRAINING = bool(RUNTIME_CFG["run_training"])
RUN_INFERENCE_ONLY = bool(RUNTIME_CFG["run_inference_only"])
INFERENCE_ONLY_PATH = str(RUNTIME_CFG["inference_only_path"])
PRINT_MODEL = bool(EXTRA_VERBOSE["print_model"])
