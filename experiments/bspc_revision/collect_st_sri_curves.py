"""R006：ST-SRI 滞后剖面曲线采集（训练/零模型变体，CPU 可运行）。

对 onset 协议的独立审计窗口执行遮挡扫描，输出 `st_sri_detector.py`
可直接消费的 npz 曲线文件。支持三类模型变体：

- `trained`：加载既有检查点（默认 R005 未平衡基线，R005b 完成后切换）；
- `reinit`：完全重初始化模型（零分布对照，无需训练）；
- `param_scramble`：张量内参数置乱模型（零分布对照，无需训练）。

标签打乱训练模型需要 GPU 训练，不属于本脚本；后续作为独立训练任务补充。

默认强制 CPU 并限制线程数，避免与正在进行的 GPU 训练争用资源。
背景窗只取训练分区中众数标签为静息的窗口，与审计分区无原始采样重叠。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import pathlib
import types
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import LSTMModel, ST_SRI_Interpreter, build_model  # noqa: E402
from experiments.bspc_revision.leakage_free_db2 import (  # noqa: E402
    NUM_CLASSES,
    build_trial_records,
    build_window_records,
    sha256_file,
)
from experiments.bspc_revision.onset_protocol import build_onset_audit_records  # noqa: E402
from experiments.bspc_revision.transition_windows import PhaseWindowRecord  # noqa: E402

# Compatibility for checkpoints saved on Windows/Python 3.13 (pathlib._local.WindowsPath).
if not hasattr(pathlib, "_local"):
    _fake_pathlib_local = types.ModuleType("pathlib._local")
    _fake_pathlib_local.WindowsPath = pathlib.PureWindowsPath
    sys.modules.setdefault("pathlib._local", _fake_pathlib_local)

MODEL_KWARGS = dict(input_size=12, hidden_size=256, num_layers=3, num_classes=18, dropout=0.3)
SPLIT_SEED = 20260815
PURGE_SAMPLES = 600
WINDOW_SAMPLES = 600
TRAIN_WINDOW_STRIDE = 100
VARIANTS = ("trained", "reinit", "param_scramble")


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
        "--arch",
        choices=("lstm", "tcn", "transformer", "resnet1d"),
        default="lstm",
        help="Architecture used to create the checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints_bspc_v2" / "r005_full_s1_s40",
    )
    parser.add_argument(
        "--ensemble-checkpoint-dirs",
        type=str,
        default=None,
        help="逗号分隔的多个 checkpoint 目录，启用多模型概率平均（3 种子集成）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r006_curves" / "r005_baseline",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--target-mode",
        choices=("argmax", "true"),
        default="argmax",
        help="argmax=模型实际决策（主指标）；true=固定真实类别（信息可用性诊断）",
    )
    parser.add_argument("--phases-ms", nargs="+", type=float, default=[0, 50, 100, 150])
    parser.add_argument(
        "--records-csv",
        type=Path,
        default=None,
        help="R009 统一相位窗口清单（phase_windows.csv）；提供后按清单采集，忽略 --phases-ms",
    )
    parser.add_argument(
        "--window-kinds",
        nargs="+",
        choices=("onset", "offset", "steady"),
        default=["onset", "offset", "steady"],
        help="从 --records-csv 中筛选的窗口类型",
    )
    parser.add_argument("--max-lag-ms", type=float, default=150.0)
    parser.add_argument("--stride-samples", type=int, default=1)
    parser.add_argument("--block-samples", type=int, default=10)
    parser.add_argument("--background-count", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=0, help="0 表示全部审计记录")
    parser.add_argument("--null-seed", type=int, default=20260816)
    parser.add_argument("--threads", type=int, default=2, help="CPU 线程上限，避免拖慢 GPU 训练")
    parser.add_argument(
        "--pooling",
        choices=("last", "mean", "attention"),
        default="last",
        help="LSTM 时间聚合方式，需与训练检查点一致",
    )
    parser.add_argument("--device", default="cpu", help="默认 cpu；GPU 空闲后可指定 cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scramble_state_dict(state_dict: dict, seed: int) -> dict:
    """张量内参数置乱：逐张量随机重排数值，保留每张量的数值多重集。"""
    rng = np.random.default_rng(seed)
    scrambled = {}
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu()
        values = tensor.numpy().ravel()
        permuted = values[rng.permutation(values.size)]
        scrambled[key] = torch.from_numpy(permuted.reshape(tensor.shape).astype(tensor.numpy().dtype))
    return scrambled


def build_analysis_model(arch: str, num_classes: int, pooling: str) -> torch.nn.Module:
    if arch == "lstm":
        return LSTMModel(**{**MODEL_KWARGS, "num_classes": num_classes, "pooling": pooling})
    return build_model(arch, input_size=12, num_classes=num_classes)


def build_variant_model(
    variant: str,
    checkpoint: dict,
    seed: int,
    pooling: str = "last",
    num_classes: int = NUM_CLASSES,
    arch: str = "lstm",
) -> torch.nn.Module:
    """按变体构造模型：trained 加载检查点，reinit 种子化重建，param_scramble 置乱。"""
    if variant == "reinit":
        torch.manual_seed(seed)
        model = build_analysis_model(arch, num_classes, pooling)
    else:
        model = build_analysis_model(arch, num_classes, pooling)
        if variant == "trained":
            state_dict = checkpoint["model_state_dict"]
        elif variant == "param_scramble":
            state_dict = scramble_state_dict(checkpoint["model_state_dict"], seed)
        else:
            raise ValueError(f"unknown variant: {variant}")
        model.load_state_dict(state_dict)
    return model.eval()


def variant_seed(null_seed: int, subject: int, variant: str) -> int:
    return null_seed + subject * 1000 + VARIANTS.index(variant)


def load_phase_window_records(
    csv_path: Path,
    subject: int,
    window_kinds: tuple[str, ...] = ("onset", "offset", "steady"),
) -> list[PhaseWindowRecord]:
    """从 R009 统一相位窗口清单加载单受试者记录。"""
    int_fields = {
        "subject", "active_class", "repetition_index", "segment_index", "phase_samples",
        "reference_sample", "current_endpoint_sample", "current_block_start_sample",
        "current_block_end_sample", "input_start_sample", "input_end_sample",
        "reference_in_window", "current_endpoint_in_window",
        "current_block_start_in_window", "current_block_end_in_window",
    }
    records = []
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject"]) != subject or row["window_kind"] not in window_kinds:
                continue
            values = {
                key: (int(row[key]) if key in int_fields else float(row[key]) if key == "phase_ms" else row[key])
                for key in row
            }
            records.append(PhaseWindowRecord(**values))
    if not records:
        raise ValueError(f"no phase window records for S{subject} in {csv_path}")
    return records


def sample_background_windows(
    data: np.ndarray,
    labels: np.ndarray,
    trials,
    mean: np.ndarray,
    std: np.ndarray,
    count: int,
    seed: int,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """从训练分区抽取众数标签为静息的窗口作为遮挡基线背景。"""
    train_trials = [trial for trial in trials if trial.split == "train"]
    windows = build_window_records(
        labels, train_trials, WINDOW_SAMPLES, TRAIN_WINDOW_STRIDE, num_classes
    )
    rest_windows = [record for record in windows if record.modal_label == 0]
    if not rest_windows:
        raise ValueError("no rest-modal training windows available for background")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(rest_windows), size=min(count, len(rest_windows)), replace=False)
    stacked = np.stack(
        [
            (np.asarray(data[rest_windows[i].raw_start : rest_windows[i].raw_end], dtype=np.float32) - mean) / std
            for i in chosen
        ]
    )
    return torch.from_numpy(stacked)


def normalized_record_window(data: np.ndarray, record, mean, std) -> torch.Tensor:
    raw = np.asarray(data[record.input_start_sample : record.input_end_sample], dtype=np.float32)
    return torch.from_numpy((raw - mean) / std)


def scan_subject_variant(
    subject: int,
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """对单受试者单变体扫描全部审计窗口，返回可保存为 npz 的数组字典。"""
    data = np.load(args.data_root / f"S{subject}_data.npy", mmap_mode="r")
    labels = np.load(args.data_root / f"S{subject}_label.npy", mmap_mode="r")
    invalid_labels = np.unique(labels[(labels < 0) | (labels >= args.num_classes)])
    if invalid_labels.size:
        raise ValueError(
            f"S{subject}: labels outside [0, {args.num_classes - 1}]: "
            f"{invalid_labels.tolist()}"
        )
    trials = build_trial_records(
        labels, subject, SPLIT_SEED, PURGE_SAMPLES, args.num_classes
    )
    if args.records_csv is not None:
        records: list = load_phase_window_records(
            args.records_csv, subject, tuple(args.window_kinds)
        )
        records_source = f"phase_windows_csv:{sha256_file(args.records_csv)}"
    else:
        records = build_onset_audit_records(
            trials,
            labels,
            split="audit",
            phases_ms=args.phases_ms,
            fs=2000,
            window_samples=WINDOW_SAMPLES,
            block_samples=args.block_samples,
        )
        records_source = "onset_protocol"
    if args.max_records > 0:
        records = records[: args.max_records]

    ensemble_dirs = []
    if args.ensemble_checkpoint_dirs:
        ensemble_dirs = [Path(p.strip()) for p in args.ensemble_checkpoint_dirs.split(",") if p.strip()]
    use_ensemble = bool(ensemble_dirs) and variant == "trained"
    seed = variant_seed(args.null_seed, subject, variant)

    if use_ensemble:
        checkpoint_paths = [d / f"S{subject:02d}_best.pth" for d in ensemble_dirs]
        checkpoints = [
            torch.load(p, map_location="cpu", weights_only=False) for p in checkpoint_paths
        ]
        mean = np.asarray(checkpoints[0]["training_mean"], dtype=np.float32)
        std = np.asarray(checkpoints[0]["training_std"], dtype=np.float32)
        models = []
        for ckpt in checkpoints:
            model = build_analysis_model(args.arch, args.num_classes, args.pooling).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            models.append(model)
        model = models[0]

        def predict_proba(x_batch):
            with torch.no_grad():
                probs = [torch.softmax(m(x_batch), dim=1) for m in models]
                return torch.stack(probs).mean(dim=0)
    else:
        checkpoint_path = args.checkpoint_dir / f"S{subject:02d}_best.pth"
        if ensemble_dirs:
            checkpoint_path = ensemble_dirs[0] / f"S{subject:02d}_best.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        mean = np.asarray(checkpoint["training_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["training_std"], dtype=np.float32)
        model = build_variant_model(
            variant,
            checkpoint,
            seed,
            pooling=args.pooling,
            num_classes=args.num_classes,
            arch=args.arch,
        )

        def predict_proba(x_batch):
            with torch.no_grad():
                return torch.softmax(model(x_batch), dim=1)

    background = sample_background_windows(
        data, labels, trials, mean, std, args.background_count, seed, args.num_classes
    )
    interpreter = ST_SRI_Interpreter(model, background, device=device, predict_proba=predict_proba)

    ids, synergy_rows, interaction_rows = [], [], []
    active_classes, phase_list, repetition_list, segment_list = [], [], [], []
    kind_list = []
    predicted_classes, predicted_probs, target_probs, correct_flags = [], [], [], []

    for record in records:
        x = normalized_record_window(data, record, mean, std).to(device)
        probs = predict_proba(x.unsqueeze(0))[0]
        predicted_class = int(torch.argmax(probs).item())
        target_cls = None if args.target_mode == "argmax" else record.active_class

        lags_ms, synergy, redundancy = interpreter.scan_fast(
            x,
            max_lag_ms=args.max_lag_ms,
            stride=args.stride_samples,
            block_size=args.block_samples,
            current_endpoint=WINDOW_SAMPLES - 1,
            target_cls=target_cls,
        )
        synergy = np.asarray(synergy, dtype=np.float32)
        interactions = synergy + np.asarray(redundancy, dtype=np.float32)

        window_kind = getattr(record, "window_kind", "onset")
        ids.append(f"S{subject:02d}_{window_kind}_seg{record.segment_index:03d}_ph{record.phase_ms:g}")
        synergy_rows.append(synergy)
        interaction_rows.append(interactions)
        active_classes.append(record.active_class)
        phase_list.append(record.phase_ms)
        repetition_list.append(record.repetition_index)
        segment_list.append(record.segment_index)
        kind_list.append(window_kind)
        predicted_classes.append(predicted_class)
        predicted_probs.append(float(probs[predicted_class].item()))
        target_probs.append(float(probs[record.active_class].item()))
        correct_flags.append(int(predicted_class == record.active_class))

    lag_axis = np.asarray(lags_ms, dtype=np.float32)
    meta = {
        "subject": subject,
        "variant": variant,
        "arch": args.arch,
        "num_classes": args.num_classes,
        "target_mode": args.target_mode,
        "records_source": records_source,
        "phases_ms": [float(phase) for phase in args.phases_ms],
        "max_lag_ms": args.max_lag_ms,
        "stride_samples": args.stride_samples,
        "block_samples": args.block_samples,
        "background_count": int(background.shape[0]),
        "background_source": "training-partition rest-modal windows only",
        "checkpoint": ";".join(str(p) for p in (checkpoint_paths if use_ensemble else [checkpoint_path])),
        "checkpoint_sha256": ";".join(sha256_file(p) for p in (checkpoint_paths if use_ensemble else [checkpoint_path])),
        "null_seed": args.null_seed,
        "variant_seed": seed,
        "device": str(device),
    }
    return {
        "lags_ms": lag_axis,
        "curves": np.stack(synergy_rows),
        "interactions": np.stack(interaction_rows),
        "ids": np.asarray(ids),
        "window_kind": np.asarray(kind_list),
        "active_class": np.asarray(active_classes, dtype=np.int64),
        "phase_ms": np.asarray(phase_list, dtype=np.float32),
        "repetition_index": np.asarray(repetition_list, dtype=np.int64),
        "segment_index": np.asarray(segment_list, dtype=np.int64),
        "predicted_class": np.asarray(predicted_classes, dtype=np.int64),
        "predicted_prob": np.asarray(predicted_probs, dtype=np.float32),
        "target_prob": np.asarray(target_probs, dtype=np.float32),
        "correct": np.asarray(correct_flags, dtype=np.int64),
        "meta_json": np.asarray(json.dumps(meta, ensure_ascii=False)),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    subjects = list(range(1, 41)) if args.all_subjects else sorted(set(args.subjects))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        for variant in args.variants:
            output_path = args.output_dir / f"S{subject:02d}_{variant}.npz"
            if output_path.exists() and not args.overwrite:
                print(f"skip existing {output_path}", flush=True)
                continue
            start = time.time()
            payload = scan_subject_variant(subject, variant, args, device)
            temporary_path = output_path.with_name(output_path.stem + ".tmp.npz")
            np.savez(temporary_path, **payload)
            os.replace(temporary_path, output_path)
            print(
                f"S{subject:02d} {variant}: {payload['curves'].shape[0]} curves "
                f"in {time.time() - start:.1f}s -> {output_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
