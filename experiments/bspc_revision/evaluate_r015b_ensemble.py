"""Evaluate R015b 3-seed ensemble validation/audit macro-F1.

Run from experiments/bspc_revision:
    python evaluate_r015b_ensemble.py \
      --subjects 1 10 15 17 27 40 \
      --checkpoint-dirs ../../checkpoints_exploratory/r015b_full_seed1,../../checkpoints_exploratory/r015b_full_seed2,../../checkpoints_exploratory/r015b_full_seed3 \
      --pooling mean \
      --output ../../results/bspc_revision_v2/r015b_full_ensemble_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import LSTMModel

try:
    from .leakage_free_db2 import (
        NUM_CLASSES,
        WindowDataset,
        evaluate,
        subject_protocol,
    )
except ImportError:  # Direct execution from experiments/bspc_revision.
    from leakage_free_db2 import (
        NUM_CLASSES,
        WindowDataset,
        evaluate,
        subject_protocol,
    )


class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([torch.softmax(model(x), dim=1) for model in self.models]).mean(dim=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 10, 15, 17, 27, 40])
    parser.add_argument("--checkpoint-dirs", type=str, required=True)
    parser.add_argument("--pooling", choices=("last", "mean", "attention"), default="mean")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r015b_full_ensemble_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_dirs = [Path(p.strip()) for p in args.checkpoint_dirs.split(",") if p.strip()]
    if len(checkpoint_dirs) < 2:
        raise ValueError("need at least two checkpoint dirs for ensemble")

    class Args:
        pass

    protocol_args = Args()
    protocol_args.data_root = args.data_root
    protocol_args.output_dir = Path("../../results/tmp_ensemble")
    protocol_args.window_ms = 300.0
    protocol_args.step_ms = 50.0
    protocol_args.fs = 2000
    protocol_args.purge_ms = 300.0
    protocol_args.split_seed = 20260815
    protocol_args.max_train_windows = 0
    protocol_args.max_validation_windows = 0
    protocol_args.max_audit_windows = 0

    results = {}
    for subject in args.subjects:
        summary, grouped, trials, mean, std = subject_protocol(
            subject, protocol_args, write_window_manifest=False
        )
        data_path = args.data_root / f"S{subject}_data.npy"
        models = []
        for ckpt_dir in checkpoint_dirs:
            ckpt_path = ckpt_dir / f"S{subject:02d}_best.pth"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = LSTMModel(input_size=12, hidden_size=256, num_layers=3,
                              num_classes=NUM_CLASSES, dropout=0.3, pooling=args.pooling)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            models.append(model)
        ensemble = EnsembleModel(models).to(device)
        metrics = {}
        for split in ("validation", "audit"):
            dataset = WindowDataset(data_path, grouped[split], mean, std)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=device.type == "cuda")
            metrics[split] = evaluate(ensemble, loader, device)
        results[subject] = metrics
        print(f"S{subject:02d} val_macro_f1={metrics['validation']['macro_f1']:.4f} "
              f"audit_macro_f1={metrics['audit']['macro_f1']:.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
