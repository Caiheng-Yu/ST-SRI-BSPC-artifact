"""R010：基线与输出表征扫描（ST-SRI 滞后曲线，无泄漏审计）。

对 onset 协议审计窗口，扫描 5 种基线 × 6 种表征的 ST-SRI 滞后曲线：

基线（把被遮挡的历史/当前块换成什么）：
- ``rest_mean``      训练分区静息窗的均值窗口（与既有 ST_SRI_Interpreter
  默认基线一致，作为锚点）；
- ``rest_sampled``   从训练分区静息窗采样 K 个（``--sample-baselines``）真实窗口，
  预测层面求期望：每个采样窗口分别填充并计算四项遮挡分数，再对 K 次
  interaction 曲线取平均；
- ``background_set`` 从完整背景集合（``--background-count``）中采 K 个
  （``--background-samples``）窗口做同样的预测层面期望平均；
- ``interpolation``  被遮挡区间用端点线性插值填充（每通道独立：区间 [a,b) 的
  每个位置 t 用 x[a-1] 与 x[b] 按位置线性插值；a==0 时用 x[b] 复制，
  右边界 b==T 时用 x[a-1] 复制）；
- ``positional_mean`` 训练分区全部窗口的逐位置均值模板（每通道每时间位置平均，
  不限静息类）。

表征 = 分数模式 × 曲线模式：
- 分数模式：``prob``（目标类 softmax 概率）、``logit``（目标类原始 logit）；
- 曲线模式：``signed``（原始 interaction）、``positive``（synergy=max(i,0)）、
  ``abs``（|interaction|）。
共 6 组合，npz 数组命名 ``curve_{score}_{mode}``。

背景窗只取训练分区（split=="train"）众数标签为静息（modal_label==0）的窗口，
并在代码中断言背景窗全部来自训练分区且与审计/验证分区无原始采样重叠，回应
"背景与审计无原始采样重叠"的审稿要求。

默认强制 CPU 并限制线程数，避免与正在进行的 GPU 训练争用资源。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import pathlib
import types
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import FS, LSTMModel  # noqa: E402
from experiments.bspc_revision.leakage_free_db2 import (  # noqa: E402
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

BASELINES = ("rest_mean", "rest_sampled", "background_set", "interpolation", "positional_mean")
SCORE_MODES = ("prob", "logit")
CURVE_MODES = ("signed", "positive", "abs")
REPRESENTATION_NAMES = tuple(
    f"curve_{score}_{mode}" for score in SCORE_MODES for mode in CURVE_MODES
)

_TAG_SALT = {"background_pool": 1, "rest_sampled": 2, "background_set": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "DB2",
        help="DB2 data directory; override for another dataset layout.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints_bspc_v2" / "r005b_balanced_full_seed20260815",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r010_baseline_repr",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=BASELINES,
        default=list(BASELINES),
        help="要扫描的基线；默认全部 5 种",
    )
    parser.add_argument(
        "--target-mode",
        choices=("argmax", "true"),
        default="argmax",
        help="argmax=模型实际决策类；true=固定真实类别 record.active_class",
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
    parser.add_argument(
        "--sample-baselines",
        type=int,
        default=8,
        help="rest_sampled 基线采样 K 个真实静息窗的数量",
    )
    parser.add_argument(
        "--background-samples",
        type=int,
        default=16,
        help="background_set 基线从完整背景集合中采样 K 个窗口的数量",
    )
    parser.add_argument("--max-records", type=int, default=0, help="0 表示全部审计记录")
    parser.add_argument("--threads", type=int, default=2, help="CPU 线程上限，避免拖慢 GPU 训练")
    parser.add_argument("--device", default="cpu", help="默认 cpu；GPU 空闲后可指定 cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def baseline_seed(subject: int, tag: str) -> int:
    """确定性、可复现的每受试者采样种子。"""
    if tag not in _TAG_SALT:
        raise ValueError(f"unknown seed tag: {tag}")
    return SPLIT_SEED + subject * 1009 + _TAG_SALT[tag] * 101


def load_phase_window_records(
    csv_path: Path,
    subject: int,
    window_kinds: tuple[str, ...] = ("onset", "offset", "steady"),
) -> list[PhaseWindowRecord]:
    """从 R009 统一相位窗口清单加载单受试者记录（同 collect_st_sri_curves）。"""
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


def training_rest_windows(labels: np.ndarray, trials) -> list:
    """训练分区中众数标签为静息（modal_label==0）的全部窗口记录。"""
    train_trials = [trial for trial in trials if trial.split == "train"]
    windows = build_window_records(labels, train_trials, WINDOW_SAMPLES, TRAIN_WINDOW_STRIDE)
    return [record for record in windows if record.modal_label == 0]


def sample_background_records(labels: np.ndarray, trials, count: int, seed: int) -> list:
    """从训练分区静息窗采样 ``count`` 个窗口记录（无放回）。"""
    rest_windows = training_rest_windows(labels, trials)
    if not rest_windows:
        raise ValueError("no rest-modal training windows available for background")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(rest_windows), size=min(count, len(rest_windows)), replace=False)
    return [rest_windows[int(i)] for i in chosen]


def stack_background(data: np.ndarray, records, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """把窗口记录堆叠为归一化 (N, T, C) 张量。"""
    stacked = np.stack(
        [
            (np.asarray(data[record.raw_start : record.raw_end], dtype=np.float32) - mean) / std
            for record in records
        ]
    )
    return torch.from_numpy(stacked)


def sample_background_windows(
    data: np.ndarray,
    labels: np.ndarray,
    trials,
    mean: np.ndarray,
    std: np.ndarray,
    count: int,
    seed: int,
) -> torch.Tensor:
    """从训练分区抽取众数标签为静息的窗口作为遮挡基线背景（同 collect_st_sri_curves）。"""
    records = sample_background_records(labels, trials, count, seed)
    return stack_background(data, records, mean, std)


def assert_train_partition_only(records, trials) -> None:
    """断言背景窗全部来自训练分区，且不与任何非训练 trial 的原始采样重叠。"""
    train_spans = [(trial.raw_start, trial.raw_end) for trial in trials if trial.split == "train"]
    other_spans = [(trial.raw_start, trial.raw_end) for trial in trials if trial.split != "train"]
    for record in records:
        assert record.split == "train", f"background window not from training partition: {record}"
        assert any(
            start <= record.raw_start and record.raw_end <= end for start, end in train_spans
        ), f"background window not contained in any training trial: {record}"
        for start, end in other_spans:
            assert record.raw_end <= start or record.raw_start >= end, (
                f"background window overlaps a non-training trial: {record}"
            )


def build_positional_mean(
    data: np.ndarray,
    labels: np.ndarray,
    trials,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor:
    """训练分区全部窗口（不限静息类）的逐位置均值模板 (T, C)。"""
    train_trials = [trial for trial in trials if trial.split == "train"]
    windows = build_window_records(labels, train_trials, WINDOW_SAMPLES, TRAIN_WINDOW_STRIDE)
    if not windows:
        raise ValueError("no training windows available for positional mean")
    total = np.zeros((WINDOW_SAMPLES, data.shape[1]), dtype=np.float64)
    for record in windows:
        raw = np.asarray(data[record.raw_start : record.raw_end], dtype=np.float32)
        total += (raw - mean) / std
    total /= len(windows)
    return torch.from_numpy(total.astype(np.float32))


def sample_rows(pool: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """从窗口池张量中无放回采样 ``count`` 行。"""
    n = pool.shape[0]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=min(count, n), replace=False)
    return pool[torch.as_tensor(chosen, dtype=torch.long)]


def normalized_record_window(data: np.ndarray, record, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    raw = np.asarray(data[record.input_start_sample : record.input_end_sample], dtype=np.float32)
    return torch.from_numpy((raw - mean) / std)


def template_fill(template: torch.Tensor):
    """返回一个把遮挡区间覆盖为固定模板窗口的填充函数。"""
    def fill(masked_tensor, t_start, t_end, row_indices):
        masked_tensor[row_indices, t_start:t_end, :] = template[t_start:t_end, :]
    return fill


def interpolation_fill(x: torch.Tensor):
    """端点线性插值填充（每通道独立）。

    区间 [a,b)：a==0 用 x[b] 复制；b>=len(x)（右边界）用 x[a-1] 复制；否则
    用 x[a-1] 与 x[b] 按位置线性插值。右边界规则是左边界 a==0 规则的对称情形
    （窗口末端之后不存在 x[b] 采样点，故用块前最后一个采样点恒定延拓）。
    """
    T, C = x.shape
    dtype = x.dtype
    device = x.device

    def fill(masked_tensor, t_start, t_end, row_indices):
        a, b = int(t_start), int(t_end)
        length = b - a
        if a == 0:
            fill_vals = x[b].unsqueeze(0).expand(length, C)
        elif b >= T:
            fill_vals = x[a - 1].unsqueeze(0).expand(length, C)
        else:
            left = x[a - 1]
            right = x[b]
            denom = float(b - (a - 1))
            weights = torch.arange(1, length + 1, dtype=dtype, device=device) / denom
            fill_vals = left.unsqueeze(0) + (right - left).unsqueeze(0) * weights.unsqueeze(1)
        masked_tensor[row_indices, a:b, :] = fill_vals

    return fill


def _extract_scores(logits: torch.Tensor, target_cls: int) -> tuple[np.ndarray, np.ndarray]:
    """返回目标类的 (softmax 概率, 原始 logit) 分数，形状 (B,)。"""
    probs = torch.softmax(logits, dim=1)
    return probs[:, target_cls].cpu().numpy(), logits[:, target_cls].cpu().numpy()


def scan_with_fill(
    model: torch.nn.Module,
    x: torch.Tensor,
    lags: list[int],
    block_size: int,
    current_endpoint: int,
    target_cls: int,
    fill,
) -> tuple[np.ndarray, np.ndarray]:
    """超级批遮挡扫描（仿 scan_fast），填充策略参数化为 ``fill``。

    返回 (interaction_prob, interaction_logit)，各形状 (N_lags,)。
    """
    N_lags = len(lags)
    if N_lags == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty

    T, C = x.shape
    x_base = x.unsqueeze(0).expand(N_lags, T, C).contiguous()
    x_lag = x_base.clone()
    x_curr = x_base.clone()
    x_none = x_base.clone()

    curr_t = int(current_endpoint)
    t_end = curr_t + 1
    t_start = max(0, t_end - block_size)
    all_rows = list(range(N_lags))

    # 当前块遮挡：影响 x_lag 与 x_none 的全部行。
    fill(x_lag, t_start, t_end, all_rows)
    fill(x_none, t_start, t_end, all_rows)

    # 历史块遮挡：影响 x_curr 与 x_none 的对应行。
    for i, tau in enumerate(lags):
        t_prev_end = curr_t - tau + 1
        t_prev_start = t_prev_end - block_size
        fill(x_curr, t_prev_start, t_prev_end, [i])
        fill(x_none, t_prev_start, t_prev_end, [i])

    with torch.no_grad():
        logits_both = model(x.unsqueeze(0))
        logits_lag = model(x_lag)
        logits_curr = model(x_curr)
        logits_none = model(x_none)

    f_both_p, f_both_l = _extract_scores(logits_both, target_cls)
    s_lag_p, s_lag_l = _extract_scores(logits_lag, target_cls)
    s_curr_p, s_curr_l = _extract_scores(logits_curr, target_cls)
    s_none_p, s_none_l = _extract_scores(logits_none, target_cls)

    interaction_prob = f_both_p[0] - s_lag_p - s_curr_p + s_none_p
    interaction_logit = f_both_l[0] - s_lag_l - s_curr_l + s_none_l
    return interaction_prob, interaction_logit


def scan_record(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_cls: int,
    lags: list[int],
    block_size: int,
    current_endpoint: int,
    fills,
) -> tuple[np.ndarray, np.ndarray]:
    """对一条记录按 ``fills``（单填充或 K 个采样填充）扫描并返回期望 interaction。"""
    if len(fills) == 1:
        return scan_with_fill(model, x, lags, block_size, current_endpoint, target_cls, fills[0])
    acc_p = None
    acc_l = None
    for fill in fills:
        ip, il = scan_with_fill(model, x, lags, block_size, current_endpoint, target_cls, fill)
        if acc_p is None:
            acc_p = ip.astype(np.float64)
            acc_l = il.astype(np.float64)
        else:
            acc_p += ip
            acc_l += il
    return (acc_p / len(fills)).astype(np.float32), (acc_l / len(fills)).astype(np.float32)


def compute_lags(max_lag_ms: float, stride: int, block_size: int, current_endpoint: int) -> list[int]:
    max_lag_points = int(max_lag_ms * (FS / 1000))
    curr_t = int(current_endpoint)
    max_lag_points = min(max_lag_points, curr_t - block_size + 1)
    return list(range(stride, max_lag_points + 1, stride))


def build_fixed_fills(ctx: dict, baseline: str, args: argparse.Namespace, device: torch.device):
    """构造不依赖单条记录 x 的填充函数列表（插值除外）。"""
    if baseline == "rest_mean":
        return [template_fill(ctx["rest_mean_template"])]
    if baseline == "positional_mean":
        template = build_positional_mean(
            ctx["data"], ctx["labels"], ctx["trials"], ctx["mean"], ctx["std"]
        ).to(device)
        return [template_fill(template)]
    if baseline == "rest_sampled":
        records = sample_background_records(
            ctx["labels"], ctx["trials"], args.sample_baselines,
            baseline_seed(ctx["subject"], "rest_sampled"),
        )
        pool = stack_background(ctx["data"], records, ctx["mean"], ctx["std"]).to(device)
        return [template_fill(window) for window in pool]
    if baseline == "background_set":
        pool = sample_rows(
            ctx["background_pool"], args.background_samples,
            baseline_seed(ctx["subject"], "background_set"),
        )
        return [template_fill(window) for window in pool]
    raise ValueError(f"unknown baseline: {baseline}")


def load_subject_context(subject: int, args: argparse.Namespace, device: torch.device) -> dict:
    """加载单受试者的数据、检查点、模型与背景材料。"""
    data = np.load(args.data_root / f"S{subject}_data.npy", mmap_mode="r")
    labels = np.load(args.data_root / f"S{subject}_label.npy", mmap_mode="r")
    trials = build_trial_records(labels, subject, SPLIT_SEED, PURGE_SAMPLES)
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

    checkpoint_path = args.checkpoint_dir / f"S{subject:02d}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["training_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["training_std"], dtype=np.float32)

    model = LSTMModel(**MODEL_KWARGS)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    # 背景窗只来自训练分区静息窗，并断言与审计/验证无原始采样重叠。
    background_records = sample_background_records(
        labels, trials, args.background_count, baseline_seed(subject, "background_pool")
    )
    assert_train_partition_only(background_records, trials)
    background_pool = stack_background(data, background_records, mean, std).to(device)

    current_endpoint = WINDOW_SAMPLES - 1
    lags = compute_lags(args.max_lag_ms, args.stride_samples, args.block_samples, current_endpoint)

    return {
        "subject": subject,
        "data": data,
        "labels": labels,
        "trials": trials,
        "records": records,
        "records_source": records_source,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "mean": mean,
        "std": std,
        "model": model,
        "device": device,
        "background_records": background_records,
        "background_pool": background_pool,
        "rest_mean_template": background_pool.mean(dim=0),
        "lags": lags,
        "current_endpoint": current_endpoint,
    }


def scan_subject_baseline(ctx: dict, baseline: str, args: argparse.Namespace) -> dict:
    """对单受试者单基线扫描全部审计窗口，返回可保存为 npz 的数组字典。"""
    records = ctx["records"]
    model = ctx["model"]
    device = ctx["device"]
    lags = ctx["lags"]
    current_endpoint = ctx["current_endpoint"]
    block_size = args.block_samples
    per_record_fill = baseline == "interpolation"
    fixed_fills = None if per_record_fill else build_fixed_fills(ctx, baseline, args, device)

    ids, kind_list = [], []
    active_classes, phase_list, repetition_list, segment_list = [], [], [], []
    predicted_classes, predicted_probs, target_probs, correct_flags = [], [], [], []
    interactions_prob, interactions_logit = [], []

    for record in records:
        x = normalized_record_window(ctx["data"], record, ctx["mean"], ctx["std"]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x.unsqueeze(0)), dim=1)[0]
        predicted_class = int(torch.argmax(probs).item())
        target_cls = predicted_class if args.target_mode == "argmax" else record.active_class

        fills = [interpolation_fill(x)] if per_record_fill else fixed_fills
        interaction_prob, interaction_logit = scan_record(
            model, x, target_cls, lags, block_size, current_endpoint, fills
        )

        window_kind = getattr(record, "window_kind", "onset")
        ids.append(f"S{ctx['subject']:02d}_{window_kind}_seg{record.segment_index:03d}_ph{record.phase_ms:g}")
        kind_list.append(window_kind)
        active_classes.append(record.active_class)
        phase_list.append(record.phase_ms)
        repetition_list.append(record.repetition_index)
        segment_list.append(record.segment_index)
        predicted_classes.append(predicted_class)
        predicted_probs.append(float(probs[predicted_class].item()))
        target_probs.append(float(probs[record.active_class].item()))
        correct_flags.append(int(predicted_class == record.active_class))
        interactions_prob.append(interaction_prob.astype(np.float32))
        interactions_logit.append(interaction_logit.astype(np.float32))

    ip_arr = np.stack(interactions_prob)
    il_arr = np.stack(interactions_logit)

    lag_axis = np.asarray([lag * (1000 / FS) for lag in lags], dtype=np.float32)
    curves = {
        "curve_prob_signed": ip_arr,
        "curve_prob_positive": np.maximum(ip_arr, 0.0),
        "curve_prob_abs": np.abs(ip_arr),
        "curve_logit_signed": il_arr,
        "curve_logit_positive": np.maximum(il_arr, 0.0),
        "curve_logit_abs": np.abs(il_arr),
    }

    meta = {
        "subject": ctx["subject"],
        "baseline": baseline,
        "score_modes": list(SCORE_MODES),
        "curve_modes": list(CURVE_MODES),
        "representations": list(REPRESENTATION_NAMES),
        "score_mode_description": (
            "prob=target-class softmax probability, logit=target-class raw logit; "
            "signed=raw interaction, positive=max(interaction,0)=synergy, abs=|interaction|"
        ),
        "target_mode": args.target_mode,
        "records_source": ctx["records_source"],
        "record_count": len(records),
        "phases_ms": [float(phase) for phase in args.phases_ms],
        "max_lag_ms": args.max_lag_ms,
        "stride_samples": args.stride_samples,
        "stride": args.stride_samples,
        "block_samples": args.block_samples,
        "background_source": "training-partition rest-modal windows only",
        "background_assertion": (
            "all background windows contained in training trials and disjoint "
            "from non-training trials (no raw-sample overlap with audit)"
        ),
        "background_count": args.background_count,
        "background_sampled_count": int(ctx["background_pool"].shape[0]),
        "sample_baselines": args.sample_baselines,
        "background_samples": args.background_samples,
        "checkpoint": str(ctx["checkpoint_path"]),
        "checkpoint_sha256": ctx["checkpoint_sha256"],
        "device": str(device),
    }

    payload = {
        "lags_ms": lag_axis,
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
    payload.update(curves)
    return payload


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    subjects = list(range(1, 41)) if args.all_subjects else sorted(set(args.subjects))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        ctx = load_subject_context(subject, args, device)
        for baseline in args.baselines:
            output_path = args.output_dir / f"S{subject:02d}_{baseline}.npz"
            if output_path.exists() and not args.overwrite:
                print(f"skip existing {output_path}", flush=True)
                continue
            start = time.time()
            payload = scan_subject_baseline(ctx, baseline, args)
            temporary_path = output_path.with_name(output_path.stem + ".tmp.npz")
            np.savez(temporary_path, **payload)
            os.replace(temporary_path, output_path)
            print(
                f"S{subject:02d} {baseline}: {payload['curve_prob_signed'].shape[0]} records "
                f"x {payload['curve_prob_signed'].shape[1]} lags "
                f"in {time.time() - start:.1f}s -> {output_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
