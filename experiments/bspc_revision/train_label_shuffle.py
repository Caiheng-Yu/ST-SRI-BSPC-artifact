"""R006 备用：标签打乱训练（阴性对照，需 GPU，待 R005b 完成后排队）。

训练协议与 R005a/R005b 完全一致（轮数、早停、类别权重、逐受试者种子、
macro-F1 选模），唯一区别是训练窗口的众数标签被按受试者种子确定性置乱。
模型因此经历完整训练流程但未学到真实的输入→标签映射，作为 ST-SRI
全流程零分布的“训练过但未学到”对照。

验证集和审计集标签保持不变，检查点仍由验证集 macro-F1 选择；
这保证阴性对照的选模协议与主模型一致。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from experiments.bspc_revision.leakage_free_db2 import (  # noqa: E402
    WindowRecord,
    json_ready_config,
    load_completed_subject_result,
    seed_everything,
    subject_protocol,
    train_subject,
    training_signature,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=18,
        help="Total label count including rest (DB2=18, E2=24).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r006_label_shuffle",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints_bspc_v2" / "r006_label_shuffle",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--window-ms", type=float, default=300.0)
    parser.add_argument("--step-ms", type=float, default=50.0)
    parser.add_argument("--fs", type=int, default=2000)
    parser.add_argument("--purge-ms", type=float, default=300.0)
    parser.add_argument("--split-seed", type=int, default=20260815)
    parser.add_argument("--train-seed", type=int, default=20260815)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=("plateau", "cosine"), default="plateau")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pooling", choices=("last", "mean", "attention"), default="last")
    parser.add_argument("--arch", choices=("lstm", "tcn", "transformer", "resnet1d"), default="lstm")
    parser.add_argument("--selection-metric", choices=("accuracy", "macro_f1", "balanced_accuracy"), default="macro_f1")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--per-subject-seed", action="store_true", default=True)
    # Label-shuffle controls are independent per subject; two workers overlap
    # memory-mapped window preparation with GPU training without changing the
    # split, batch order, or optimization settings.
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-validation-windows", type=int, default=0)
    parser.add_argument("--max-audit-windows", type=int, default=0)
    parser.add_argument("--skip-window-manifests", action="store_true", default=True)
    parser.add_argument("--label-shuffle-seed", type=int, default=20260816)
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def shuffle_train_window_labels(
    grouped: dict[str, list[WindowRecord]],
    seed: int,
) -> dict[str, list[WindowRecord]]:
    """在训练窗口内部对众数标签做确定性置乱；验证和审计分区保持不变。"""
    train_records = grouped["train"]
    labels = np.array([record.modal_label for record in train_records], dtype=np.int64)
    shuffled = labels[np.random.default_rng(seed).permutation(labels.size)]
    new_train = [
        dataclasses.replace(record, modal_label=int(shuffled[index]))
        for index, record in enumerate(train_records)
    ]
    return {"train": new_train, "validation": grouped["validation"], "audit": grouped["audit"]}


def main() -> None:
    args = parse_args()
    subjects = list(range(1, 41)) if args.all_subjects else sorted(set(args.subjects))
    if any(subject < 1 or subject > 40 for subject in subjects):
        raise ValueError("subjects must be between 1 and 40")

    seed_everything(args.train_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device} subjects={subjects} label_shuffle_seed={args.label_shuffle_seed}", flush=True)

    training_results = []
    status_path = args.output_dir / "run_status.json"
    write_json_atomic(
        status_path,
        {
            "state": "running",
            "device": str(device),
            "label_shuffle_seed": args.label_shuffle_seed,
            "requested_subjects": subjects,
            "completed_subjects": [],
            "current_subject": None,
        },
    )
    for subject in subjects:
        summary, grouped, _, mean, std = subject_protocol(
            subject, args, write_window_manifest=not args.skip_window_manifests
        )
        shuffled_grouped = shuffle_train_window_labels(grouped, args.label_shuffle_seed + subject)
        signature = training_signature(args, summary)
        result = None
        if args.resume:
            result = load_completed_subject_result(subject, args, summary)
        if result is None:
            write_json_atomic(
                status_path,
                {
                    "state": "running",
                    "device": str(device),
                    "label_shuffle_seed": args.label_shuffle_seed,
                    "requested_subjects": subjects,
                    "completed_subjects": [item["subject"] for item in training_results],
                    "current_subject": subject,
                },
            )
            result = train_subject(
                subject,
                args.data_root / f"S{subject}_data.npy",
                shuffled_grouped,
                mean,
                std,
                args,
                device,
                signature,
            )
            result["label_shuffle_seed"] = args.label_shuffle_seed + subject
            write_json_atomic(args.output_dir / "subjects" / f"S{subject:02d}_result.json", result)
        training_results.append(result)
        write_json_atomic(
            args.output_dir / "protocol_summary.json",
            {
                "state": "running",
                "config": json_ready_config(args),
                "device": str(device),
                "label_shuffle_seed": args.label_shuffle_seed,
                "training_results": training_results,
            },
        )

    write_json_atomic(
        status_path,
        {
            "state": "completed",
            "device": str(device),
            "label_shuffle_seed": args.label_shuffle_seed,
            "requested_subjects": subjects,
            "completed_subjects": [result["subject"] for result in training_results],
            "current_subject": None,
        },
    )
    write_json_atomic(
        args.output_dir / "protocol_summary.json",
        {
            "state": "completed",
            "config": json_ready_config(args),
            "device": str(device),
            "label_shuffle_seed": args.label_shuffle_seed,
            "training_results": training_results,
        },
    )
    print(f"completed output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
