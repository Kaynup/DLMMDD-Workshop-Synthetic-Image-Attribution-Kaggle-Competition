from __future__ import annotations

from pathlib import Path

from .utils import get_selection_metric_value


class CheckpointManager:
    @staticmethod
    def prune_to_top_k_checkpoints(saved, keep_top_k):
        while len(saved) > keep_top_k:
            weakest_idx = min(range(len(saved)), key=lambda i: saved[i]['selection_value'])
            weakest = saved.pop(weakest_idx)
            checkpoint_path = Path(weakest['checkpoint_path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            print(f'  [prune] {checkpoint_path.name}  sv={weakest["selection_value"]:.4f}')
        return saved

    @staticmethod
    def cleanup_fold_checkpoints(fold_dir, keep_paths):
        keep = set(str(path) for path in keep_paths)
        for path in Path(fold_dir).glob('epoch_*.pth'):
            if str(path) not in keep:
                path.unlink()


class CheckpointSelector:
    @staticmethod
    def get_selection_metric_value(metrics):
        return get_selection_metric_value(metrics)

    @staticmethod
    def best_checkpoint(history):
        return max(history, key=lambda row: row['selection_value'])
