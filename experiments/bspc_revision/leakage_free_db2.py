"""BSPC revision pipeline for leakage-free DB2 training and audit splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import LSTMModel, build_model  # noqa: E402


SPLITS = ("train", "validation", "audit")
NUM_CLASSES = 18
# Backward-compatible default for DB2 callers; non-default datasets derive this per run.
ACTIVE_CLASSES = tuple(range(1, NUM_CLASSES))


@dataclass(frozen=True)
class Segment:
    label: int
    start: int
    end: int


@dataclass(frozen=True)
class TrialRecord:
    subject: int
    split: str
    active_class: int
    repetition_index: int
    segment_index: int
    active_start: int
    active_end: int
    raw_start: int
    raw_end: int


@dataclass(frozen=True)
class WindowRecord:
    subject: int
    split: str
    active_class: int
    repetition_index: int
    segment_index: int
    raw_start: int
    raw_end: int
    modal_label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help="Total label count including rest (DB2=18, E2=24).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "run",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints_bspc_v2" / "run",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--window-ms", type=float, default=300.0)
    parser.add_argument("--step-ms", type=float, default=50.0)
    parser.add_argument("--fs", type=int, default=2000)
    parser.add_argument("--purge-ms", type=float, default=300.0)
    parser.add_argument("--split-seed", type=int, default=20260815)
    parser.add_argument("--train-seed", type=int, default=20260815)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    # R015 加强 LSTM 配方（默认值与 R005b 完全一致，纯增量）
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam 权重衰减（R015 建议 1e-4）")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="交叉熵标签平滑（R015 建议 0.05）")
    parser.add_argument(
        "--lr-scheduler",
        choices=("plateau", "cosine"),
        default="plateau",
        help="plateau=ReduceLROnPlateau（R005b 默认）；cosine=余弦退火（R015 建议）",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="梯度裁剪 max_norm（R005b 实际使用 1.0；设为 0 表示不裁剪）",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "macro_f1", "balanced_accuracy"),
        default="accuracy",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.0,
        help="类别频数负幂权重；0 关闭，0.5 表示平方根逆频率",
    )
    parser.add_argument(
        "--arch",
        choices=("lstm", "tcn", "transformer", "resnet1d"),
        default="lstm",
        help="模型架构：lstm（默认）、tcn、transformer、resnet1d",
    )
    parser.add_argument(
        "--pooling",
        choices=("last", "mean", "attention"),
        default="last",
        help="LSTM 时间聚合方式：last=最后时间步（R005b/R015 默认）；mean=时间维平均；attention=可学习注意力",
    )
    parser.add_argument(
        "--per-subject-seed",
        action="store_true",
        help="每名受试者训练前重置独立且可续跑复现的随机种子",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-validation-windows", type=int, default=0)
    parser.add_argument("--max-audit-windows", type=int, default=0)
    parser.add_argument("--skip-window-manifests", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已原子保存且检查点与配置校验通过的受试者",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def portable_project_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved_path)


def json_ready_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def training_signature(args: argparse.Namespace, summary: dict[str, object]) -> str:
    signature_keys = (
        "window_ms",
        "step_ms",
        "fs",
        "purge_ms",
        "split_seed",
        "train_seed",
        "epochs",
        "patience",
        "batch_size",
        "learning_rate",
        "max_train_windows",
        "max_validation_windows",
        "max_audit_windows",
    )
    enhanced_training = (
        args.min_epochs != 0
        or args.selection_metric != "accuracy"
        or args.class_weight_power != 0.0
        or args.per_subject_seed
        or args.pooling != "last"
        or args.arch != "lstm"
        or args.num_classes != NUM_CLASSES
    )
    pipeline = "leakage_free_db2_lstm_v1"
    if enhanced_training:
        signature_keys += (
            "min_epochs",
            "selection_metric",
            "class_weight_power",
            "per_subject_seed",
            "pooling",
            "arch",
        )
        pipeline = "leakage_free_db2_lstm_v2_balanced"
    if args.num_classes != NUM_CLASSES:
        signature_keys += ("num_classes",)
    payload = {
        "pipeline": pipeline,
        "subject": summary["subject"],
        "data_sha256": summary["data_sha256"],
        "label_sha256": summary["label_sha256"],
        "config": {key: getattr(args, key) for key in signature_keys},
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_completed_subject_result(
    subject: int,
    args: argparse.Namespace,
    summary: dict[str, object],
) -> dict[str, object] | None:
    result_path = args.output_dir / "subjects" / f"S{subject:02d}_result.json"
    if not args.resume or not result_path.exists():
        return None
    try:
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        checkpoint_path = resolve_project_path(result["checkpoint"])
        if result.get("subject") != subject:
            raise ValueError("subject mismatch")
        if result.get("training_signature") != training_signature(args, summary):
            raise ValueError("training signature mismatch")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint {checkpoint_path}")
        if sha256_file(checkpoint_path) != result.get("checkpoint_sha256"):
            raise ValueError("checkpoint hash mismatch")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"S{subject:02d} resume record rejected: {error}", flush=True)
        return None
    print(f"S{subject:02d} resume record accepted", flush=True)
    return result


def find_segments(labels: np.ndarray) -> list[Segment]:
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], labels.size]
    return [
        Segment(label=int(labels[start]), start=int(start), end=int(end))
        for start, end in zip(starts, ends)
    ]


def split_repetitions(active_class: int, split_seed: int) -> dict[int, str]:
    rng = np.random.default_rng(split_seed + active_class * 1009)
    order = rng.permutation(6).tolist()
    assignment: dict[int, str] = {}
    for repetition_index in order[:4]:
        assignment[int(repetition_index)] = "train"
    assignment[int(order[4])] = "validation"
    assignment[int(order[5])] = "audit"
    return assignment


def build_trial_records(
    labels: np.ndarray,
    subject: int,
    split_seed: int,
    purge_samples: int,
    num_classes: int = NUM_CLASSES,
) -> list[TrialRecord]:
    segments = find_segments(labels)
    by_class: dict[int, list[tuple[int, Segment]]] = defaultdict(list)
    for segment_index, segment in enumerate(segments):
        if segment.label != 0:
            by_class[segment.label].append((segment_index, segment))

    active_classes = tuple(range(1, num_classes))
    missing = [label for label in active_classes if len(by_class[label]) != 6]
    if missing:
        counts = {label: len(by_class[label]) for label in active_classes}
        raise ValueError(f"S{subject}: expected 6 repetitions per active class, got {counts}")

    left_purge = purge_samples // 2
    right_purge = purge_samples - left_purge
    trials: list[TrialRecord] = []

    for active_class in active_classes:
        assignment = split_repetitions(active_class, split_seed)
        for repetition_index, (segment_index, active_segment) in enumerate(by_class[active_class]):
            previous_segment = segments[segment_index - 1] if segment_index > 0 else None
            next_segment = segments[segment_index + 1] if segment_index + 1 < len(segments) else None
            if previous_segment is None or previous_segment.label != 0:
                raise ValueError(f"S{subject}: active segment {segment_index} lacks preceding rest")
            if next_segment is None or next_segment.label != 0:
                raise ValueError(f"S{subject}: active segment {segment_index} lacks following rest")

            left_boundary = (previous_segment.start + previous_segment.end) // 2
            right_boundary = (next_segment.start + next_segment.end) // 2
            raw_start = (
                0
                if previous_segment.start == 0
                else left_boundary + left_purge
            )
            raw_end = (
                int(labels.size)
                if next_segment.end == labels.size
                else right_boundary - right_purge
            )
            if raw_start >= active_segment.start or raw_end <= active_segment.end:
                raise ValueError(
                    f"S{subject}: purge removed active samples for class {active_class}, "
                    f"repetition {repetition_index}"
                )

            trials.append(
                TrialRecord(
                    subject=subject,
                    split=assignment[repetition_index],
                    active_class=active_class,
                    repetition_index=repetition_index,
                    segment_index=segment_index,
                    active_start=active_segment.start,
                    active_end=active_segment.end,
                    raw_start=raw_start,
                    raw_end=raw_end,
                )
            )

    return sorted(trials, key=lambda record: record.raw_start)


def window_modal_labels(
    labels: np.ndarray,
    starts: np.ndarray,
    window_len: int,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    if starts.size == 0:
        return np.empty(0, dtype=np.int64)
    prefix = np.empty((num_classes, labels.size + 1), dtype=np.int32)
    prefix[:, 0] = 0
    for class_index in range(num_classes):
        np.cumsum(labels == class_index, dtype=np.int32, out=prefix[class_index, 1:])
    ends = starts + window_len
    counts = prefix[:, ends] - prefix[:, starts]
    return np.argmax(counts, axis=0).astype(np.int64)


def build_window_records(
    labels: np.ndarray,
    trials: Sequence[TrialRecord],
    window_len: int,
    stride: int,
    num_classes: int = NUM_CLASSES,
) -> list[WindowRecord]:
    trial_starts: list[np.ndarray] = []
    trial_refs: list[TrialRecord] = []
    for trial in trials:
        last_start = trial.raw_end - window_len
        if last_start < trial.raw_start:
            raise ValueError(f"S{trial.subject}: trial is shorter than one window: {trial}")
        starts = np.arange(trial.raw_start, last_start + 1, stride, dtype=np.int64)
        trial_starts.append(starts)
        trial_refs.extend([trial] * len(starts))

    all_starts = np.concatenate(trial_starts) if trial_starts else np.empty(0, dtype=np.int64)
    modal_labels = window_modal_labels(labels, all_starts, window_len, num_classes)
    records = []
    for trial, raw_start, modal_label in zip(trial_refs, all_starts.tolist(), modal_labels.tolist()):
        records.append(
            WindowRecord(
                subject=trial.subject,
                split=trial.split,
                active_class=trial.active_class,
                repetition_index=trial.repetition_index,
                segment_index=trial.segment_index,
                raw_start=int(raw_start),
                raw_end=int(raw_start + window_len),
                modal_label=int(modal_label),
            )
        )
    return records


def records_by_split(records: Iterable[WindowRecord]) -> dict[str, list[WindowRecord]]:
    grouped = {split: [] for split in SPLITS}
    for record in records:
        grouped[record.split].append(record)
    return grouped


def raw_coverage(records: Sequence[WindowRecord], total_samples: int) -> np.ndarray:
    difference = np.zeros(total_samples + 1, dtype=np.int32)
    for record in records:
        difference[record.raw_start] += 1
        difference[record.raw_end] -= 1
    return np.cumsum(difference[:-1]) > 0


def verify_split_isolation(
    grouped: dict[str, list[WindowRecord]],
    total_samples: int,
) -> dict[str, int]:
    coverage = {split: raw_coverage(grouped[split], total_samples) for split in SPLITS}
    overlaps = {
        "train_validation": int(np.count_nonzero(coverage["train"] & coverage["validation"])),
        "train_audit": int(np.count_nonzero(coverage["train"] & coverage["audit"])),
        "validation_audit": int(
            np.count_nonzero(coverage["validation"] & coverage["audit"])
        ),
    }
    if any(overlaps.values()):
        raise AssertionError(f"raw-sample leakage detected: {overlaps}")
    return overlaps


def compute_training_statistics(
    data: np.ndarray,
    train_trials: Sequence[TrialRecord],
) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros(data.shape[1], dtype=np.float64)
    total_square = np.zeros(data.shape[1], dtype=np.float64)
    count = 0
    for trial in train_trials:
        block = np.asarray(data[trial.raw_start : trial.raw_end], dtype=np.float64)
        total += block.sum(axis=0)
        total_square += np.square(block).sum(axis=0)
        count += block.shape[0]
    if count == 0:
        raise ValueError("training split contains no raw samples")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32), count


def stratified_limit(
    records: Sequence[WindowRecord],
    limit: int,
    seed: int,
    num_classes: int = NUM_CLASSES,
) -> list[WindowRecord]:
    if limit <= 0 or len(records) <= limit:
        return list(records)
    if limit < num_classes:
        raise ValueError(f"window limit {limit} is too small to retain all {num_classes} classes")

    rng = np.random.default_rng(seed)
    by_label: dict[int, list[WindowRecord]] = defaultdict(list)
    for record in records:
        by_label[record.modal_label].append(record)
    missing = [label for label in range(num_classes) if not by_label[label]]
    if missing:
        raise ValueError(f"cannot stratify: missing modal labels {missing}")

    base = limit // num_classes
    selected: list[WindowRecord] = []
    leftovers: list[WindowRecord] = []
    for label in range(num_classes):
        indices = rng.permutation(len(by_label[label]))
        take = min(base, len(indices))
        selected.extend(by_label[label][int(index)] for index in indices[:take])
        leftovers.extend(by_label[label][int(index)] for index in indices[take:])

    remaining = limit - len(selected)
    if remaining > 0:
        extra_indices = rng.permutation(len(leftovers))[:remaining]
        selected.extend(leftovers[int(index)] for index in extra_indices)
    return sorted(selected, key=lambda record: record.raw_start)


def compute_class_weights(
    records: Sequence[WindowRecord],
    power: float,
    num_classes: int = NUM_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(
        np.asarray([record.modal_label for record in records], dtype=np.int64),
        minlength=num_classes,
    )
    if np.any(counts == 0):
        raise ValueError(f"cannot weight classes with zero training count: {counts.tolist()}")
    if power == 0.0:
        weights = np.ones(num_classes, dtype=np.float32)
    else:
        weights = np.power(counts.astype(np.float64), -power)
        weights /= weights.mean()
        weights = weights.astype(np.float32)
    return counts, weights


def subject_training_seed(train_seed: int, subject: int) -> int:
    return int(train_seed + subject * 1009)


def selection_score(metrics: dict[str, object], metric_name: str) -> float:
    try:
        return float(metrics[metric_name])
    except KeyError as error:
        raise ValueError(f"unsupported selection metric: {metric_name}") from error


class WindowDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        records: Sequence[WindowRecord],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.data_path = data_path
        self.records = list(records)
        self.mean = mean
        self.std = std
        self._data: np.ndarray | None = None

    def _get_data(self) -> np.ndarray:
        if self._data is None:
            self._data = np.load(self.data_path, mmap_mode="r")
        return self._data

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        raw = np.asarray(
            self._get_data()[record.raw_start : record.raw_end], dtype=np.float32
        )
        normalized = (raw - self.mean) / self.std
        return torch.from_numpy(normalized), torch.tensor(record.modal_label, dtype=torch.long)


def confusion_metrics(confusion: np.ndarray) -> dict[str, object]:
    total = int(confusion.sum())
    true_positive = np.diag(confusion).astype(np.float64)
    true_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(true_positive, true_count, out=np.zeros_like(true_positive), where=true_count > 0)
    precision = np.divide(
        true_positive, pred_count, out=np.zeros_like(true_positive), where=pred_count > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    present = true_count > 0
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1[present].mean()) if np.any(present) else 0.0,
        "balanced_accuracy": float(recall[present].mean()) if np.any(present) else 0.0,
        "per_class_recall": {str(index): float(value) for index, value in enumerate(recall)},
        "class_support": {str(index): int(value) for index, value in enumerate(true_count)},
        "confusion_matrix": confusion.tolist(),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> dict[str, object]:
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss_sum += float(criterion(logits, targets).item())
            predictions = logits.argmax(dim=1)
            for target, prediction in zip(targets.cpu().numpy(), predictions.cpu().numpy()):
                confusion[int(target), int(prediction)] += 1
            sample_count += int(targets.numel())
    metrics = confusion_metrics(confusion)
    metrics["loss"] = loss_sum / sample_count if sample_count else math.nan
    return metrics


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def train_subject(
    subject: int,
    data_path: Path,
    grouped: dict[str, list[WindowRecord]],
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    signature: str,
) -> dict[str, object]:
    effective_seed = args.train_seed
    if args.per_subject_seed:
        effective_seed = subject_training_seed(args.train_seed, subject)
        seed_everything(effective_seed)

    limits = {
        "train": args.max_train_windows,
        "validation": args.max_validation_windows,
        "audit": args.max_audit_windows,
    }
    selected = {
        split: stratified_limit(
            grouped[split],
            limits[split],
            (
                effective_seed + split_index
                if args.per_subject_seed
                else args.train_seed + subject * 100 + split_index
            ),
            args.num_classes,
        )
        for split_index, split in enumerate(SPLITS)
    }

    datasets = {
        split: WindowDataset(data_path, selected[split], mean, std) for split in SPLITS
    }
    loaders = {
        "train": make_loader(
            datasets["train"],
            args.batch_size,
            True,
            effective_seed if args.per_subject_seed else args.train_seed + subject,
            args.num_workers,
            device,
        ),
        "validation": make_loader(
            datasets["validation"],
            args.batch_size,
            False,
            effective_seed + 1 if args.per_subject_seed else args.train_seed,
            args.num_workers,
            device,
        ),
        "audit": make_loader(
            datasets["audit"],
            args.batch_size,
            False,
            effective_seed + 2 if args.per_subject_seed else args.train_seed,
            args.num_workers,
            device,
        ),
    }

    if args.arch == "lstm":
        model = LSTMModel(
            input_size=12,
            hidden_size=256,
            num_layers=3,
            num_classes=args.num_classes,
            dropout=0.3,
            pooling=args.pooling,
        )
    else:
        model = build_model(args.arch, input_size=12, num_classes=args.num_classes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate / 100
        )
        scheduler_step = lambda: scheduler.step()
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=4, factor=0.5
        )
        scheduler_step = lambda: scheduler.step(float(validation_metrics["loss"]))
    training_class_counts, class_weights = compute_class_weights(
        selected["train"], args.class_weight_power, args.num_classes
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"S{subject:02d}_best.pth"
    best_validation_score = -1.0
    best_epoch = 0
    patience_count = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for inputs, targets in loaders["train"]:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * int(targets.numel())
            train_samples += int(targets.numel())

        validation_metrics = evaluate(model, loaders["validation"], device, args.num_classes)
        scheduler_step()
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_samples,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "selection_metric": args.selection_metric,
            "selection_score": selection_score(validation_metrics, args.selection_metric),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        print(
            f"S{subject:02d} epoch {epoch:02d}/{args.epochs} "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"val_loss={epoch_record['validation_loss']:.4f} "
            f"val_acc={epoch_record['validation_accuracy']:.4f} "
            f"val_macro_f1={epoch_record['validation_macro_f1']:.4f} "
            f"select={epoch_record['selection_score']:.4f}",
            flush=True,
        )

        current_validation_score = selection_score(validation_metrics, args.selection_metric)
        if current_validation_score > best_validation_score:
            best_validation_score = current_validation_score
            best_epoch = epoch
            patience_count = 0
            checkpoint_temporary_path = checkpoint_path.with_suffix(".pth.tmp")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "subject": subject,
                    "best_epoch": best_epoch,
                    "best_validation_score": best_validation_score,
                    "selection_metric": args.selection_metric,
                    "subject_training_seed": effective_seed,
                    "training_class_counts": training_class_counts,
                    "class_weights": class_weights,
                    "training_mean": mean,
                    "training_std": std,
                    "config": vars(args),
                },
                checkpoint_temporary_path,
            )
            os.replace(checkpoint_temporary_path, checkpoint_path)
        else:
            patience_count += 1
            if epoch >= args.min_epochs and patience_count >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics = evaluate(model, loaders["validation"], device, args.num_classes)
    audit_metrics = evaluate(model, loaders["audit"], device, args.num_classes)
    elapsed = time.time() - start_time
    result = {
        "subject": subject,
        "device": str(device),
        "subject_training_seed": effective_seed,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_validation_score": best_validation_score,
        "training_class_counts": training_class_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "checkpoint": portable_project_path(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_signature": signature,
        "elapsed_seconds": elapsed,
        "selected_window_counts": {split: len(records) for split, records in selected.items()},
        "selected_class_counts": {
            split: dict(sorted(Counter(record.modal_label for record in records).items()))
            for split, records in selected.items()
        },
        "validation_metrics": validation_metrics,
        "audit_metrics": audit_metrics,
        "history": history,
    }
    return result


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def subject_protocol(
    subject: int,
    args: argparse.Namespace,
    write_window_manifest: bool,
) -> tuple[dict[str, object], dict[str, list[WindowRecord]], list[TrialRecord], np.ndarray, np.ndarray]:
    data_path = args.data_root / f"S{subject}_data.npy"
    label_path = args.data_root / f"S{subject}_label.npy"
    if not data_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"missing data for S{subject}: {data_path} or {label_path}")

    data = np.load(data_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    invalid_labels = np.unique(labels[(labels < 0) | (labels >= args.num_classes)])
    if invalid_labels.size:
        raise ValueError(
            f"S{subject}: labels outside [0, {args.num_classes - 1}]: {invalid_labels.tolist()}"
        )
    window_len = int(round(args.window_ms * args.fs / 1000.0))
    stride = int(round(args.step_ms * args.fs / 1000.0))
    purge_samples = int(round(args.purge_ms * args.fs / 1000.0))
    trials = build_trial_records(
        labels, subject, args.split_seed, purge_samples, args.num_classes
    )
    windows = build_window_records(labels, trials, window_len, stride, args.num_classes)
    grouped = records_by_split(windows)
    overlaps = verify_split_isolation(grouped, len(labels))
    train_trials = [trial for trial in trials if trial.split == "train"]
    mean, std, normalization_samples = compute_training_statistics(data, train_trials)

    class_counts = {
        split: dict(sorted(Counter(record.modal_label for record in grouped[split]).items()))
        for split in SPLITS
    }
    missing_classes = {
        split: [label for label in range(args.num_classes) if class_counts[split].get(label, 0) == 0]
        for split in SPLITS
    }
    if any(missing_classes.values()):
        raise AssertionError(f"S{subject}: class coverage failure: {missing_classes}")

    trial_counts = {
        split: dict(
            sorted(Counter(trial.active_class for trial in trials if trial.split == split).items())
        )
        for split in SPLITS
    }
    summary = {
        "subject": subject,
        "raw_sample_count": int(len(labels)),
        "window_length_samples": window_len,
        "stride_samples": stride,
        "purge_samples": purge_samples,
        "trial_counts": trial_counts,
        "window_counts": {split: len(grouped[split]) for split in SPLITS},
        "class_counts": class_counts,
        "missing_classes": missing_classes,
        "raw_overlap_samples": overlaps,
        "normalization_source": "training trial raw samples only",
        "normalization_sample_count": normalization_samples,
        "training_mean": mean.tolist(),
        "training_std": std.tolist(),
        "data_sha256": sha256_file(data_path),
        "label_sha256": sha256_file(label_path),
    }

    if write_window_manifest:
        manifest_path = args.output_dir / "window_manifests" / f"S{subject:02d}.csv"
        write_csv(
            manifest_path,
            [field.name for field in WindowRecord.__dataclass_fields__.values()],
            (asdict(record) for record in windows),
        )
    return summary, grouped, trials, mean, std


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.patience < 1:
        raise ValueError("patience must be positive")
    if args.min_epochs < 0 or args.min_epochs > args.epochs:
        raise ValueError("min_epochs must be between 0 and epochs")
    if not 0.0 <= args.class_weight_power <= 1.0:
        raise ValueError("class_weight_power must be between 0 and 1")
    if args.num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    subjects = list(range(1, 41)) if args.all_subjects else sorted(set(args.subjects))
    if any(subject < 1 or subject > 40 for subject in subjects):
        raise ValueError("subjects must be between 1 and 40")

    seed_everything(args.train_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device} subjects={subjects} manifest_only={args.manifest_only}", flush=True)

    summaries = []
    all_trials: list[TrialRecord] = []
    training_results = []
    status_path = args.output_dir / "run_status.json"
    write_json_atomic(
        status_path,
        {
            "state": "running",
            "device": str(device),
            "requested_subjects": subjects,
            "completed_subjects": [],
            "current_subject": None,
        },
    )
    for subject in subjects:
        print(f"preparing S{subject:02d}", flush=True)
        summary, grouped, trials, mean, std = subject_protocol(
            subject, args, write_window_manifest=not args.skip_window_manifests
        )
        summaries.append(summary)
        all_trials.extend(trials)
        if not args.manifest_only:
            signature = training_signature(args, summary)
            result = load_completed_subject_result(subject, args, summary)
            if result is None:
                write_json_atomic(
                    status_path,
                    {
                        "state": "running",
                        "device": str(device),
                        "requested_subjects": subjects,
                        "completed_subjects": [item["subject"] for item in training_results],
                        "current_subject": subject,
                    },
                )
                result = train_subject(
                    subject,
                    args.data_root / f"S{subject}_data.npy",
                    grouped,
                    mean,
                    std,
                    args,
                    device,
                    signature,
                )
                write_json_atomic(
                    args.output_dir / "subjects" / f"S{subject:02d}_result.json",
                    result,
                )
            training_results.append(result)
            write_json_atomic(
                args.output_dir / "protocol_summary.json",
                {
                    "state": "running",
                    "config": json_ready_config(args),
                    "device": str(device),
                    "subjects": summaries,
                    "training_results": training_results,
                },
            )

    write_json_atomic(
        args.output_dir / "protocol_summary.json",
        {
            "state": "completed",
            "config": json_ready_config(args),
            "device": str(device),
            "subjects": summaries,
            "training_results": training_results,
        },
    )
    write_csv(
        args.output_dir / "split_trials.csv",
        [field.name for field in TrialRecord.__dataclass_fields__.values()],
        (asdict(record) for record in all_trials),
    )
    class_rows = []
    for summary in summaries:
        for split in SPLITS:
            for label in range(args.num_classes):
                class_rows.append(
                    {
                        "subject": summary["subject"],
                        "split": split,
                        "label": label,
                        "window_count": summary["class_counts"][split].get(label, 0),
                    }
                )
    write_csv(
        args.output_dir / "class_window_counts.csv",
        ["subject", "split", "label", "window_count"],
        class_rows,
    )
    write_json_atomic(
        status_path,
        {
            "state": "completed",
            "device": str(device),
            "requested_subjects": subjects,
            "completed_subjects": [result["subject"] for result in training_results],
            "current_subject": None,
        },
    )
    print(f"completed output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
