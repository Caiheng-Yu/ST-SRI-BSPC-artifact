"""评估 R004 检查点能否支持严格 `restimulus` onset 对齐审计。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import LSTMModel
from experiments.bspc_revision.leakage_free_db2 import (
    build_trial_records,
    confusion_metrics,
)
from experiments.bspc_revision.onset_protocol import (
    OnsetAuditDataset,
    build_onset_audit_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--phases-ms", nargs="+", type=float, default=[0, 50, 100, 150])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints_bspc_v2" / "r004_full_s2_s3",
    )
    parser.add_argument(
        "--r004-summary",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "bspc_revision_v2"
        / "r004_full_s2_s3"
        / "protocol_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r004_onset_gate",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ordinary-min-accuracy", type=float, default=0.65)
    parser.add_argument("--ordinary-min-macro-f1", type=float, default=0.50)
    parser.add_argument("--phase150-min-accuracy", type=float, default=0.50)
    parser.add_argument("--phase150-max-rest-rate", type=float, default=0.40)
    return parser.parse_args()


def load_r004_metrics(path: Path) -> dict[int, dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        int(result["subject"]): {
            "accuracy": float(result["audit_metrics"]["accuracy"]),
            "macro_f1": float(result["audit_metrics"]["macro_f1"]),
        }
        for result in payload["training_results"]
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ordinary_metrics = load_r004_metrics(args.r004_summary)
    prediction_rows = []
    phase_confusions = {
        phase: np.zeros((18, 18), dtype=np.int64) for phase in args.phases_ms
    }
    phase_target_probabilities = defaultdict(list)
    phase_argmax_probabilities = defaultdict(list)

    for subject in args.subjects:
        checkpoint_path = args.checkpoint_dir / f"S{subject:02d}_best.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mean = np.asarray(checkpoint["training_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["training_std"], dtype=np.float32)
        labels = np.load(args.data_root / f"S{subject}_label.npy", mmap_mode="r")
        purge_samples = int(round(300.0 * 2000 / 1000.0))
        trials = build_trial_records(labels, subject, split_seed=20260815, purge_samples=purge_samples)
        records = build_onset_audit_records(
            trials,
            labels,
            split="audit",
            phases_ms=args.phases_ms,
            fs=2000,
            window_samples=600,
            block_samples=10,
        )
        by_phase = defaultdict(list)
        for record in records:
            by_phase[record.phase_ms].append(record)

        model = LSTMModel(input_size=12, hidden_size=256, num_layers=3, num_classes=18, dropout=0.3)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        for phase in args.phases_ms:
            phase_records = by_phase[float(phase)]
            dataset = OnsetAuditDataset(
                args.data_root / f"S{subject}_data.npy", phase_records, mean, std
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            offset = 0
            with torch.no_grad():
                for inputs, targets in loader:
                    probabilities = torch.softmax(model(inputs.to(device)), dim=1).cpu()
                    predictions = probabilities.argmax(dim=1)
                    for row_index, (target, prediction) in enumerate(zip(targets, predictions)):
                        record = phase_records[offset + row_index]
                        target_index = int(target)
                        prediction_index = int(prediction)
                        phase_confusions[phase][target_index, prediction_index] += 1
                        target_probability = float(probabilities[row_index, target_index])
                        argmax_probability = float(probabilities[row_index, prediction_index])
                        phase_target_probabilities[phase].append(target_probability)
                        phase_argmax_probabilities[phase].append(argmax_probability)
                        prediction_rows.append(
                            {
                                **record.__dict__,
                                "target_probability": target_probability,
                                "predicted_class": prediction_index,
                                "predicted_probability": argmax_probability,
                                "target_correct": int(target_index == prediction_index),
                            }
                        )
                    offset += len(targets)

    phase_metrics = {}
    for phase in args.phases_ms:
        metrics = confusion_metrics(phase_confusions[phase])
        total = int(phase_confusions[phase].sum())
        metrics.update(
            {
                "rest_prediction_rate": float(phase_confusions[phase][:, 0].sum() / total),
                "mean_target_probability": float(np.mean(phase_target_probabilities[phase])),
                "mean_argmax_probability": float(np.mean(phase_argmax_probabilities[phase])),
            }
        )
        phase_metrics[str(float(phase))] = metrics

    ordinary_pass = all(
        ordinary_metrics[subject]["accuracy"] >= args.ordinary_min_accuracy
        and ordinary_metrics[subject]["macro_f1"] >= args.ordinary_min_macro_f1
        for subject in args.subjects
    )
    phase150 = phase_metrics.get("150.0")
    phase150_pass = phase150 is not None and (
        phase150["accuracy"] >= args.phase150_min_accuracy
        and phase150["rest_prediction_rate"] <= args.phase150_max_rest_rate
    )
    gate = {
        "r004_reusable_for_ordinary_training": ordinary_pass,
        "r004_reusable_for_onset_phase_150_audit": phase150_pass,
        "launch_r005_ordinary_training": ordinary_pass,
        "launch_strict_onset_st_sri_main_analysis": phase150_pass,
        "scope": (
            "R005 只扩展与 R004 同协议的普通窗口众数分类训练，因此由普通独立审计门控制；"
            "严格 onset ST-SRI 主分析由逐相位门单独控制。phase 0/50/100/150 未通过时"
            "只能作为诊断，不能自动解释为动作识别或动作类别决策。"
        ),
        "thresholds": {
            "ordinary_min_accuracy_per_subject": args.ordinary_min_accuracy,
            "ordinary_min_macro_f1_per_subject": args.ordinary_min_macro_f1,
            "phase150_min_pooled_accuracy": args.phase150_min_accuracy,
            "phase150_max_pooled_rest_prediction_rate": args.phase150_max_rest_rate,
        },
    }
    payload = {
        "device": str(device),
        "subjects": args.subjects,
        "phases_ms": args.phases_ms,
        "ordinary_audit_metrics": ordinary_metrics,
        "phase_metrics": phase_metrics,
        "gate": gate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    write_csv(args.output_dir / "onset_predictions.csv", prediction_rows)
    print(json.dumps(gate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
