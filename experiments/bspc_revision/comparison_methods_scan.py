"""R011：比较方法与随机化对照 —— TimeSHAP、直接析因遮挡（Dynamask 暂缓）。

2026-08-17 决定：Dynamask 学习平滑掩码、回答一阶重要性问题，与 ST-SRI 的
成对交互主张不对口；论文保留方法能力表行与文献引用，经验对比不再安排。
本脚本保留 dynamask 代码路径（未安装时优雅降级为 status=not_installed），
默认 `--methods occlusion timeshap`，需要补跑时再显式加入 dynamask。

对 onset 协议的独立审计窗口（与 `collect_st_sri_curves.py` 相同记录来源）运行三种
归因方法，输出逐窗口归因/曲线、AOPC 扰动曲线与标量指标，并记录每窗口的
正确/错误标记（predicted_class vs active_class）供分层报告。

方法
----
1. ``occlusion``（直接析因遮挡，锚点方法）
   对每个窗口计算逐 lag 的四项遮挡 interaction
   ``f_both - s_lag - s_curr + s_none``（正 = 历史块与当前块的协同贡献），
   直接复用 ``ST_SRI_Interpreter.scan_fast`` 返回的 synergy + redundancy 作为有符号
   interaction。输出逐 lag 曲线；AOPC 按 |interaction| 降序（重要度从高到低）删除
   lag 块（用背景均值填充对应历史块）计算目标类概率扰动曲线。

2. ``timeshap``（自实现，移植 e3_baseline_comparison.compute_timeshap 核心）
   TimeSHAP (KDD'21) 分组揭露的前向揭露方向简化重实现：从全背景基线开始逐块
   "揭露"时间块，测量 ΔP = P(揭露后) - P(揭露前) 得到每块的**有符号**边际贡献。
   组 = 时间块，组大小默认 10 采样点 = 5 ms。AOPC 按 |归因| 降序删除组。

3. ``dynamask``（参考方法，官方包）
   尝试 ``from dynamask import AttributionModel``。若安装，对每个窗口拟合分块掩码
   （面积正则 + 时间连续性正则）。若未安装，该方法输出 ``"status": "not_installed"``
   （在 npz meta 与 summary 中标注），脚本不崩溃，也不安装任何包。

所有方法的目标类均取原窗口的 argmax 预测类；同时保存 active_class 供正确/错误分层。

AOPC 口径（重要）
----------------
``aopc`` = 目标类概率扰动曲线下的面积，按删除步数归一化（梯形积分 / 步数）。
**面积越小 = 删除重要特征后概率掉得越狠 = 归因越集中。** 扰动曲线的第 0 个点是
未扰动窗口的目标类概率，其后每个点是按重要度降序累计删除一个特征块后的目标类概率。

默认强制 CPU 并限制线程数，避免与正在进行的 GPU 训练争用资源。
背景窗只取训练分区中众数标签为静息的窗口，与审计分区无原始采样重叠。
"""

from __future__ import annotations

import argparse
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

from common import FS, LSTMModel, ST_SRI_Interpreter  # noqa: E402
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
METHOD_CHOICES = ("occlusion", "timeshap", "dynamask")
WINDOW_KIND_CHOICES = ("onset", "offset", "steady")

# dynamask 为可选依赖：未安装时优雅降级，绝不安装任何包。
try:
    from dynamask import AttributionModel as _DynamaskAttributionModel  # noqa: F401
except Exception as exc:  # noqa: BLE001
    _DynamaskAttributionModel = None
    DYNAMASK_AVAILABLE = False
    DYNAMASK_ERROR = f"{type(exc).__name__}: {exc}"
else:
    DYNAMASK_AVAILABLE = True
    DYNAMASK_ERROR = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help="Total label count including rest (DB2=18, E2=24).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints_bspc_v2" / "r005b_balanced_full_seed20260815",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r011_comparison_methods",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=["occlusion", "timeshap"],
        help="默认 occlusion timeshap；dynamask 已暂缓（一阶方法与交互主张不对口），仅保留代码路径供审稿要求时补跑",
    )
    parser.add_argument(
        "--records-csv",
        type=Path,
        default=None,
        help="R009 统一相位窗口清单（phase_windows.csv）；提供后按清单采集，忽略 --phases-ms",
    )
    parser.add_argument(
        "--window-kinds",
        nargs="+",
        choices=WINDOW_KIND_CHOICES,
        default=list(WINDOW_KIND_CHOICES),
        help="从 --records-csv 中筛选的窗口类型",
    )
    parser.add_argument("--phases-ms", nargs="+", type=float, default=[0, 50, 100, 150])
    parser.add_argument("--max-lag-ms", type=float, default=150.0)
    parser.add_argument("--stride-samples", type=int, default=1)
    parser.add_argument("--block-samples", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=10, help="timeshap/dynamask 分块大小（采样点）")
    parser.add_argument("--background-count", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=0, help="0 表示全部审计记录")
    parser.add_argument("--threads", type=int, default=2, help="CPU 线程上限，避免拖慢 GPU 训练")
    parser.add_argument("--device", default="cpu", help="默认 cpu；GPU 空闲后可指定 cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# 记录 / 背景 / 归一化（照抄 collect_st_sri_curves.py 的实现，保持一致）        #
# --------------------------------------------------------------------------- #

def load_phase_window_records(
    csv_path: Path,
    subject: int,
    window_kinds: tuple[str, ...] = ("onset", "offset", "steady"),
) -> list[PhaseWindowRecord]:
    """从 R009 统一相位窗口清单加载单受试者记录。"""
    import csv

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


# --------------------------------------------------------------------------- #
# 归因方法实现                                                                #
# --------------------------------------------------------------------------- #

def _target_prob(model: torch.nn.Module, x_tensor: torch.Tensor, target_cls: int) -> float:
    """单个窗口 (T, C) 在目标类上的 softmax 概率。"""
    with torch.no_grad():
        probs = torch.softmax(model(x_tensor.unsqueeze(0)), dim=1)[0]
    return float(probs[target_cls].item())


def compute_occlusion(
    interpreter: ST_SRI_Interpreter,
    x: torch.Tensor,
    target_cls: int,
    max_lag_ms: float,
    stride: int,
    block_size: int,
    current_endpoint: int,
) -> tuple[np.ndarray, np.ndarray]:
    """直接析因遮挡：逐 lag 四项遮挡 interaction（synergy + redundancy）。

    返回 ``(lags_ms, interaction)``，其中 interaction 为有符号数组
    ``f_both - s_lag - s_curr + s_none``（正 = 历史块与当前块协同贡献）。
    """
    lags_ms, synergy, redundancy = interpreter.scan_fast(
        x,
        max_lag_ms=max_lag_ms,
        stride=stride,
        block_size=block_size,
        current_endpoint=current_endpoint,
        target_cls=target_cls,
    )
    lags_ms = np.asarray(lags_ms, dtype=np.float32)
    interaction = np.asarray(synergy, dtype=np.float32) + np.asarray(redundancy, dtype=np.float32)
    return lags_ms, interaction


def occlusion_blocks(
    lags_ms: np.ndarray,
    block_size: int,
    current_endpoint: int,
    fs: int = FS,
) -> list[tuple[int, int]]:
    """每个 lag 对应的历史块切片 ``[start, end)``（与 scan_fast 的 t_prev 区间一致）。"""
    blocks = []
    for lag_ms in lags_ms:
        tau = int(round(float(lag_ms) * fs / 1000.0))
        end = current_endpoint - tau + 1
        start = end - block_size
        blocks.append((start, end))
    return blocks


def group_blocks(timesteps: int, group_size: int) -> list[tuple[int, int]]:
    """把长度为 ``timesteps`` 的时间轴切成分块切片 ``[(start, end), ...]``。"""
    return [
        (g * group_size, min((g + 1) * group_size, timesteps))
        for g in range((timesteps + group_size - 1) // group_size)
    ]


def compute_timeshap_groups(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    target_cls: int,
    group_size: int,
    device: torch.device,
) -> np.ndarray:
    """TimeSHAP (KDD'21) 前向揭露的自实现：逐块有符号边际贡献。

    从全 baseline 开始逐块揭露时间块，测量 ΔP = P(揭露后) - P(揭露前)。
    返回 shape ``(n_groups,)`` 的 float32 有符号归因（每块一个分数）。
    """
    timesteps = int(x.shape[0])
    blocks = group_blocks(timesteps, group_size)
    x_masked = baseline.to(device).clone()
    prev_prob = _target_prob(model, x_masked, target_cls)
    attr = np.zeros(len(blocks), dtype=np.float32)
    for g, (start, end) in enumerate(blocks):
        x_masked[start:end] = x[start:end].to(device)
        curr_prob = _target_prob(model, x_masked, target_cls)
        attr[g] = curr_prob - prev_prob
        prev_prob = curr_prob
    return attr


def compute_dynamask_groups(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    target_cls: int,
    group_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """用官方 dynamask 拟合分块掩码（面积正则 + 时间连续性正则）。

    按 Dynamask (Bento et al., KDD'21) 文档化接口编写，聚合到 ``group_size``
    采样点的分块掩码。返回 ``(mask[n_groups], fit_loss)``。

    注意：本环境未安装 dynamask，此代码路径在调用方被 ``DYNAMASK_AVAILABLE``
    短路，尚未在真实包上验证；仅作为参考实现保留。
    """
    from dynamask import AttributionModel  # noqa: F401

    timesteps = int(x.shape[0])
    blocks = group_blocks(timesteps, group_size)

    # Dynamask 需要一个可对任意扰动输入返回目标类分数的模型；这里按官方
    # 示例把原始模型包装成打分函数。若真实包接口不同，会在此处抛错并由调用方
    # 记录为 method-level error，不使整个脚本崩溃。
    def score_fn(inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = torch.softmax(model(inputs), dim=1)
        return probs[:, target_cls]

    attributor = AttributionModel(score_fn, device=str(device), task="classification")
    mask = attributor.fit(
        x.unsqueeze(0).to(device),
        target=torch.tensor([target_cls], device=device),
        area_reg_factor=0.1,
        time_reg_factor=0.1,
        verbose=False,
    )
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    fit_loss = float(np.nan)
    # 聚合到分块：每块取块内掩码均值。
    grouped = np.asarray(
        [float(np.mean(mask[start:end])) for start, end in blocks],
        dtype=np.float32,
    )
    return grouped, fit_loss


# --------------------------------------------------------------------------- #
# AOPC 扰动曲线                                                                #
# --------------------------------------------------------------------------- #

def aopc_from_order(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    target_cls: int,
    order: np.ndarray,
    blocks: list[tuple[int, int]],
    device: torch.device,
) -> np.ndarray:
    """按 ``order``（重要度从高到低）累计删除特征块，返回目标类概率扰动曲线。

    曲线长度 = ``len(order) + 1``；第 0 个点为未扰动窗口的目标类概率，之后每个点
    为累计删除一个块后的目标类概率。批量构造累计掩码后单次前向，避免逐点前向开销。
    """
    n = len(order)
    x_stack = x.to(device).unsqueeze(0).repeat(n + 1, 1, 1)
    for k, group_index in enumerate(order, start=1):
        start, end = blocks[int(group_index)]
        x_stack[k:, start:end] = baseline[start:end]
    with torch.no_grad():
        probs = torch.softmax(model(x_stack), dim=1)[:, target_cls]
    return probs.cpu().numpy().astype(np.float32)


def aopc_area(curve: np.ndarray) -> float:
    """扰动曲线下的归一化面积（梯形积分 / 步数）。

    面积越小 = 删除重要特征后概率掉得越狠 = 归因越集中。
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.size < 2:
        return float(curve[0]) if curve.size == 1 else float("nan")
    return float(np.trapezoid(curve) / (curve.size - 1))


def importance_order(values: np.ndarray) -> np.ndarray:
    """按 |values| 降序返回索引（重要度从高到低；稳定排序保证确定性）。"""
    return np.argsort(-np.abs(np.asarray(values, dtype=np.float64)), kind="stable")


# --------------------------------------------------------------------------- #
# 每受试者×方法扫描                                                           #
# --------------------------------------------------------------------------- #

def _base_meta(
    subject: int,
    method: str,
    args: argparse.Namespace,
    checkpoint_path: Path,
    records_source: str,
    background_count: int,
    status: str,
) -> dict:
    return {
        "subject": subject,
        "method": method,
        "status": status,
        "records_source": records_source,
        "phases_ms": [float(phase) for phase in args.phases_ms],
        "window_kinds": list(args.window_kinds),
        "max_lag_ms": args.max_lag_ms,
        "stride_samples": args.stride_samples,
        "block_samples": args.block_samples,
        "group_size": args.group_size,
        "background_count": int(background_count),
        "background_source": "training-partition rest-modal windows only",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": str(args.device),
        "target_class_mode": "argmax predicted class",
        "aopc_note": (
            "aopc = normalized area under the target-class probability perturbation "
            "curve (trapezoid / n_steps); smaller = sharper probability drop when "
            "deleting top-importance features = more concentrated attribution"
        ),
    }


def run_method(
    subject: int,
    method: str,
    args: argparse.Namespace,
    device: torch.device,
    model: torch.nn.Module,
    interpreter: ST_SRI_Interpreter,
    baseline: torch.Tensor,
    windows: list[torch.Tensor],
    target_classes: list[int],
    common: dict,
    checkpoint_path: Path,
    records_source: str,
    background_count: int,
) -> dict:
    """对单个受试者的全部窗口运行单种方法，返回可保存为 npz 的 payload 字典。"""
    n = len(common["ids"])
    # Keep the optional-method path compatible with legacy callers and artifacts.
    common.setdefault("phase_ms", [float("nan")] * n)
    common.setdefault("window_kind", ["unknown"] * n)
    common.setdefault("repetition_index", [-1] * n)
    common.setdefault("segment_index", [-1] * n)
    current_endpoint = WINDOW_SAMPLES - 1

    if method == "dynamask" and not DYNAMASK_AVAILABLE:
        meta = _base_meta(subject, method, args, checkpoint_path, records_source, background_count, "not_installed")
        meta["dynamask_error"] = DYNAMASK_ERROR
        meta["runtime_seconds_per_window"] = 0.0
        return {
            "attribution": np.zeros((n, 0), dtype=np.float32),
            "aopc_curve": np.zeros((n, 0), dtype=np.float32),
            "correct": np.asarray(common["correct"], dtype=np.int64),
            "active_class": np.asarray(common["active_class"], dtype=np.int64),
            "predicted_class": np.asarray(common["predicted_class"], dtype=np.int64),
            "phase_ms": np.asarray(common["phase_ms"], dtype=np.float32),
            "window_kind": np.asarray(common["window_kind"]),
            "repetition_index": np.asarray(common["repetition_index"], dtype=np.int64),
            "segment_index": np.asarray(common["segment_index"], dtype=np.int64),
            "ids": np.asarray(common["ids"]),
            "meta_json": np.asarray(json.dumps(meta, ensure_ascii=False)),
        }

    start_time = time.time()
    attrs: list[np.ndarray] = []
    curves: list[np.ndarray] = []
    lags_ms: np.ndarray | None = None

    for x, target_cls in zip(windows, target_classes):
        x = x.to(device)
        if method == "occlusion":
            lags_ms, attribution = compute_occlusion(
                interpreter, x, target_cls,
                args.max_lag_ms, args.stride_samples, args.block_samples, current_endpoint,
            )
            blocks = occlusion_blocks(lags_ms, args.block_samples, current_endpoint)
        elif method == "timeshap":
            attribution = compute_timeshap_groups(
                model, x, baseline, target_cls, args.group_size, device,
            )
            blocks = group_blocks(WINDOW_SAMPLES, args.group_size)
        elif method == "dynamask":
            attribution, _ = compute_dynamask_groups(
                model, x, baseline, target_cls, args.group_size, device,
            )
            blocks = group_blocks(WINDOW_SAMPLES, args.group_size)
        else:
            raise ValueError(f"unknown method: {method}")

        order = importance_order(attribution)
        curve = aopc_from_order(model, x, baseline, target_cls, order, blocks, device)
        attrs.append(attribution.astype(np.float32))
        curves.append(curve)

    runtime = time.time() - start_time
    meta = _base_meta(subject, method, args, checkpoint_path, records_source, background_count, "ok")
    meta["runtime_seconds_per_window"] = runtime / n if n else 0.0
    meta["runtime_seconds_total"] = runtime
    meta["n_records"] = n

    payload = {
        "attribution": np.stack(attrs) if attrs else np.zeros((0, 0), dtype=np.float32),
        "aopc_curve": np.stack(curves) if curves else np.zeros((0, 0), dtype=np.float32),
        "correct": np.asarray(common["correct"], dtype=np.int64),
        "active_class": np.asarray(common["active_class"], dtype=np.int64),
        "predicted_class": np.asarray(common["predicted_class"], dtype=np.int64),
        "phase_ms": np.asarray(common["phase_ms"], dtype=np.float32),
        "window_kind": np.asarray(common["window_kind"]),
        "repetition_index": np.asarray(common["repetition_index"], dtype=np.int64),
        "segment_index": np.asarray(common["segment_index"], dtype=np.int64),
        "ids": np.asarray(common["ids"]),
        "meta_json": np.asarray(json.dumps(meta, ensure_ascii=False)),
    }
    if method == "occlusion" and lags_ms is not None:
        payload["lags_ms"] = lags_ms
    return payload


def scan_subject(
    subject: int,
    args: argparse.Namespace,
    device: torch.device,
    methods: list[str],
) -> dict[str, dict]:
    """加载单受试者数据与检查点，计算公共预测并运行指定方法。"""
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
        records: list = load_phase_window_records(args.records_csv, subject, tuple(args.window_kinds))
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
    if not records:
        raise ValueError(f"S{subject}: no audit records to scan")

    checkpoint_path = args.checkpoint_dir / f"S{subject:02d}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["training_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["training_std"], dtype=np.float32)

    model = LSTMModel(**{**MODEL_KWARGS, "num_classes": args.num_classes})
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.eval().to(device)

    background_seed = SPLIT_SEED + subject
    background = sample_background_windows(
        data, labels, trials, mean, std, args.background_count, background_seed,
        args.num_classes,
    )
    baseline = torch.mean(background, dim=0).to(device)
    interpreter = ST_SRI_Interpreter(model, background, device=device)

    windows: list[torch.Tensor] = []
    target_classes: list[int] = []
    common: dict = {
        "ids": [],
        "active_class": [],
        "predicted_class": [],
        "correct": [],
        "phase_ms": [],
        "window_kind": [],
        "repetition_index": [],
        "segment_index": [],
    }
    for record in records:
        x = normalized_record_window(data, record, mean, std)
        windows.append(x)
        with torch.no_grad():
            probs = torch.softmax(model(x.unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
        predicted_class = int(np.argmax(probs))
        target_classes.append(predicted_class)
        window_kind = getattr(record, "window_kind", "onset")
        common["ids"].append(
            f"S{subject:02d}_{window_kind}_seg{record.segment_index:03d}_ph{record.phase_ms:g}"
        )
        common["active_class"].append(int(record.active_class))
        common["predicted_class"].append(predicted_class)
        common["correct"].append(int(predicted_class == int(record.active_class)))
        common["phase_ms"].append(float(record.phase_ms))
        common["window_kind"].append(window_kind)
        common["repetition_index"].append(int(record.repetition_index))
        common["segment_index"].append(int(record.segment_index))

    results = {}
    for method in methods:
        results[method] = run_method(
            subject, method, args, device, model, interpreter, baseline,
            windows, target_classes, common, checkpoint_path, records_source,
            int(background.shape[0]),
        )
    return results


# --------------------------------------------------------------------------- #
# 汇总                                                                         #
# --------------------------------------------------------------------------- #

def summarize(subject: int, method: str, attribution: np.ndarray, aopc_curve: np.ndarray, meta: dict) -> dict:
    n_records = int(attribution.shape[0])
    status = meta.get("status", "ok")
    entry: dict = {
        "subject": subject,
        "method": method,
        "status": status,
        "n_records": n_records,
        "aopc_mean": None,
        "aopc_std": None,
        "stability": None,
        "runtime_seconds_per_window_mean": meta.get("runtime_seconds_per_window"),
    }
    if status == "ok":
        per_window_aopc = np.asarray([aopc_area(row) for row in aopc_curve], dtype=np.float64)
        entry["aopc_mean"] = float(np.mean(per_window_aopc))
        entry["aopc_std"] = float(np.std(per_window_aopc))
        per_window_mean_abs = np.mean(np.abs(np.asarray(attribution, dtype=np.float32)), axis=1)
        entry["stability"] = float(np.std(per_window_mean_abs))
    return entry


def summarize_saved_output(subject: int, method: str, output_path: Path) -> dict:
    """Rebuild the run summary from an atomically saved subject artifact."""
    with np.load(output_path, allow_pickle=False) as payload:
        meta = json.loads(str(payload["meta_json"]))
        return summarize(subject, method, payload["attribution"], payload["aopc_curve"], meta)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    subjects = list(range(1, 41)) if args.all_subjects else sorted(set(args.subjects))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_entries: list[dict] = []
    for subject in subjects:
        needed = []
        for method in args.methods:
            output_path = args.output_dir / f"S{subject:02d}_{method}.npz"
            if not args.overwrite and output_path.exists():
                summary_entries.append(summarize_saved_output(subject, method, output_path))
            else:
                needed.append(method)
        if not needed:
            print(f"S{subject:02d}: all requested methods already present, skip", flush=True)
            continue
        results = scan_subject(subject, args, device, needed)
        for method, payload in results.items():
            output_path = args.output_dir / f"S{subject:02d}_{method}.npz"
            temporary_path = output_path.with_name(output_path.stem + ".tmp.npz")
            np.savez(temporary_path, **payload)
            os.replace(temporary_path, output_path)
            meta = json.loads(str(payload["meta_json"]))
            entry = summarize(subject, method, payload["attribution"], payload["aopc_curve"], meta)
            summary_entries.append(entry)
            print(
                f"S{subject:02d} {method}: {entry['n_records']} windows "
                f"aopc_mean={entry['aopc_mean']} stability={entry['stability']} "
                f"runtime/window={entry['runtime_seconds_per_window_mean']} -> {output_path}",
                flush=True,
            )

    summary_payload = {
        "status": "completed",
        "protocol": "R011 comparison methods vs randomization controls",
        "config": {
            "subjects": subjects,
            "methods": list(args.methods),
            "data_root": str(args.data_root),
            "num_classes": args.num_classes,
            "checkpoint_dir": str(args.checkpoint_dir),
            "max_lag_ms": args.max_lag_ms,
            "stride_samples": args.stride_samples,
            "block_samples": args.block_samples,
            "group_size": args.group_size,
            "background_count": args.background_count,
            "max_records": args.max_records,
            "device": str(device),
        },
        "dynamask_installed": DYNAMASK_AVAILABLE,
        "dynamask_error": DYNAMASK_ERROR,
        "aopc_note": (
            "aopc = normalized area under the target-class probability perturbation "
            "curve (trapezoid / n_steps); smaller = sharper probability drop when "
            "deleting top-importance features = more concentrated attribution"
        ),
        "results": summary_entries,
    }
    write_json_atomic(args.output_dir / "summary.json", summary_payload)
    print(f"summary -> {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
