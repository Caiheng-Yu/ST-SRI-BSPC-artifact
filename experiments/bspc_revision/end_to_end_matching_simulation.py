"""A1: 端到端匹配模拟验证（Reviewer 1 硬性要求）。

在真实多通道 sEMG 信号上，用与真实审计完全相同的代码路径验证 ST-SRI
检测链：已知交互模型必须在真实滞后处被检出，无交互模型必须只产生
α 水平的假阳性。

管线逐环节匹配真实流程（collect_st_sri_curves.py + st_sri_detector.py）：
  1. 信号：NinaPro DB2 E1 真实数据（12 通道，2000 Hz）活动段窗口，保留真实
     自相关/互相关结构；归一化用全局统计量（合成模拟中不涉及数据泄漏）；
  2. 背景：静息段窗口均值（对应真实流程的 rest-modal 训练窗口基线）；
  3. 扫描：10 样本当前块、1 样本步长重叠滞后扫描、max_lag=150 ms
     （scan_fast：max_lag_ms=150, stride=1, block_size=10, current_endpoint=T-1）；
  4. 构造：背景替换 4 掩码 → I = f(不遮) − f(遮curr) − f(遮lag) + f(遮两者)；
     曲线保留符号（interactions = synergy + redundancy），正值截断发生在
     检测器的 positive_mass（与真实 npz 一致）；
  5. 检测：先剔除 |lag| < 5 ms（10 样本 @2000 Hz）再高斯平滑 σ=2 bins →
     聚合 → 寻峰 → 边界处理（curve_statistics / detect_peak）；
  6. 阈值：仅由无交互模型（null）曲线的 (1−α) 分位标定（calibrate_thresholds）。

统计口径：
  - 检出率：有交互模型曲线被支持 且 峰在 GT 滞后 ±5 ms 内；
  - 假阳性率：无交互模型曲线被支持的比例（应 ≈ α = 5%）；
  - 无支持峰比例：无交互模型 no-support 比例；
  - 峰误差：有交互模型检出峰与 GT 滞后的偏差分布。

模型（合成打分器，模拟已训练分类器的目标类概率）：
  - null（无交互）：        score = a·E_curr + ε
  - inter（已知交互 @τ_gt）：score = a·E_curr + b·E_curr·E_lag(τ_gt) + ε
  E_block = 块内 12 通道平方和；ε ~ N(0, noise_std)；p = sigmoid(score)，
  predict_proba 返回 [1−p, p]，目标类固定为 1。

用法：
  python experiments/bspc_revision/end_to_end_matching_simulation.py \
      --data-root <dir> --smoke            # 冒烟：3 seeds × 20 窗口/条件
  python experiments/bspc_revision/end_to_end_matching_simulation.py \
      --data-root <dir>                    # 完整：20 seeds × 100 窗口/条件
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import FS, ST_SRI_Interpreter  # noqa: E402
from st_sri_detector import (  # noqa: E402
    DetectorConfig,
    NullThresholds,
    calibrate_thresholds,
    curve_statistics,
    detect_peak,
)

WINDOW_SAMPLES = 600          # 300 ms @ 2000 Hz，与真实窗口一致
BLOCK_SAMPLES = 10            # 当前块 10 样本（= 5 ms 排除区）
MAX_LAG_MS = 150.0
SMOOTH_SIGMA_BINS = 2.0
ALPHA = 0.05
N_CHANNELS = 12
DETECTOR_CONFIG = DetectorConfig(
    fs=FS,
    block_samples=BLOCK_SAMPLES,
    max_lag_ms=MAX_LAG_MS,
    stride_samples=1,
    smooth_sigma_bins=SMOOTH_SIGMA_BINS,
    alpha=ALPHA,
)


class SyntheticScoreModel:
    """确定性打分器：score = a·E_curr (+ b·E_curr·E_lag@tau_gt) + 噪声。"""

    def __init__(
        self,
        alpha_coef: float,
        beta: float,
        tau_gt_samples: int | None,
        noise_std: float,
    ) -> None:
        self.alpha_coef = alpha_coef
        self.beta = beta
        self.tau_gt_samples = tau_gt_samples
        self.noise_std = noise_std

    # ST_SRI_Interpreter.__init__ 调用 model.to(device).eval()；合成模型提供桩方法
    def to(self, device):
        return self

    def eval(self):
        return self

    def block_energy(self, x: torch.Tensor, t_end_exclusive: int) -> torch.Tensor:
        """块 [t_end_exclusive-BLOCK, t_end_exclusive) 的 12 通道去均值 AC 能量。

        t_end_exclusive 语义与 scan_fast 的掩码区间一致（掩码 [curr+1-BLOCK, curr+1)），
        保证掩码替换与模型能量计算完全对齐。
        """
        t_start = max(0, t_end_exclusive - BLOCK_SAMPLES)
        block = x[:, t_start:t_end_exclusive, :]
        centered = block - block.mean(dim=1, keepdim=True)
        return centered.pow(2).sum(dim=(1, 2))

    def predict_proba(self, x_batch: torch.Tensor) -> torch.Tensor:
        """返回 (B, 2) 目标类概率；目标类固定为 1。

        概率直接线性构造（裁剪到 [0,1]），避免 sigmoid 饱和区对
        交互的压缩/符号反转，保证概率空间存在显式的已知交互：
        p = clip(a·E_curr + b·E_curr·E_lag@τ_gt + ε, 0, 1)
        """
        curr_t = x_batch.shape[1] - 1
        e_curr = self.block_energy(x_batch, curr_t + 1)
        prob = self.alpha_coef * e_curr
        if self.tau_gt_samples is not None:
            e_lag = self.block_energy(x_batch, curr_t + 1 - self.tau_gt_samples)
            prob = prob + self.beta * e_curr * e_lag
        if self.noise_std > 0:
            prob = prob + torch.randn_like(prob) * self.noise_std
        # 不 clamp：保持完全线性，避免非线性截断在低能量窗口制造 null 重尾；
        # 合成模型允许概率超出 [0,1]（无物理约束），get_score_batch 直接用该值
        return torch.stack([1.0 - prob, prob], dim=1)


def load_subject(data_root: Path, subject: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(data_root / f"S{subject}_data.npy", mmap_mode="r")
    labels = np.load(data_root / f"S{subject}_label.npy", mmap_mode="r")
    return data, labels


_STATS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def subject_stats(data_root: Path, subject: int) -> tuple[np.ndarray, np.ndarray]:
    """每受试者自身 mean/std（对应真实流程的"只用训练分区统计量"）。"""
    if subject not in _STATS_CACHE:
        data, _ = load_subject(data_root, subject)
        arr = np.asarray(data, dtype=np.float64)
        mean = arr.mean(axis=0)
        std = np.sqrt(np.maximum((arr * arr).mean(axis=0) - mean * mean, 1e-12))
        _STATS_CACHE[subject] = (mean, std)
    return _STATS_CACHE[subject]


def sample_windows(
    data_root: Path,
    subjects: list[int],
    n_windows: int,
    seed: int,
    rest_fraction: float = 0.2,
) -> tuple[list[np.ndarray], torch.Tensor]:
    """采样窗口：活动段窗口（inter 条件）与静息段窗口（背景）。

    每受试者用自身 mean/std 归一化（与真实训练分区归一化一致）；
    背景为静息段窗口堆叠（ST_SRI_Interpreter 内部取均值作为遮挡基线）。
    活动窗口要求 600 样本全在活动段（与 onset 审计窗口一致）。
    """
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    rest_windows: list[np.ndarray] = []
    while len(windows) < n_windows:
        subject = int(rng.choice(subjects))
        data, labels = load_subject(data_root, subject)
        mean, std = subject_stats(data_root, subject)
        arr = np.asarray(data, dtype=np.float32)
        lab = np.asarray(labels)
        active = np.flatnonzero(lab > 0)
        if active.size == 0:
            continue
        start = int(rng.choice(active[: max(1, active.size - WINDOW_SAMPLES)]))
        if lab[start : start + WINDOW_SAMPLES].size < WINDOW_SAMPLES:
            continue
        if float((lab[start : start + WINDOW_SAMPLES] > 0).mean()) < 1.0:
            continue
        windows.append((arr[start : start + WINDOW_SAMPLES] - mean) / std)
    n_rest = max(1, int(n_windows * rest_fraction))
    while len(rest_windows) < n_rest:
        subject = int(rng.choice(subjects))
        data, labels = load_subject(data_root, subject)
        mean, std = subject_stats(data_root, subject)
        arr = np.asarray(data, dtype=np.float32)
        lab = np.asarray(labels)
        rest = np.flatnonzero(lab == 0)
        if rest.size == 0:
            continue
        start = int(rng.choice(rest[: max(1, rest.size - WINDOW_SAMPLES)]))
        if lab[start : start + WINDOW_SAMPLES].size < WINDOW_SAMPLES:
            continue
        rest_windows.append((arr[start : start + WINDOW_SAMPLES] - mean) / std)
    baseline = torch.from_numpy(np.stack(rest_windows))  # (n_rest, T, C)；ST_SRI_Interpreter 内部取均值
    return windows, baseline


def scan_curve(
    x_window: np.ndarray,
    model: SyntheticScoreModel,
    baseline: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """单窗口完整 ST-SRI 曲线（与真实采集同代码路径）。"""
    interpreter = ST_SRI_Interpreter(
        model, baseline, device="cpu", predict_proba=model.predict_proba
    )
    x = torch.from_numpy(np.asarray(x_window, dtype=np.float32))
    lags_ms, synergy, redundancy = interpreter.scan_fast(
        x,
        max_lag_ms=MAX_LAG_MS,
        stride=1,
        block_size=BLOCK_SAMPLES,
        current_endpoint=WINDOW_SAMPLES - 1,
        target_cls=1,  # 合成模型只输出 [1-p, p]，目标类固定为 1
    )
    interactions = np.asarray(synergy, dtype=np.float32) + np.asarray(redundancy, dtype=np.float32)
    return np.asarray(lags_ms, dtype=np.float64), interactions


def run_condition(
    windows: list[np.ndarray],
    baseline: torch.Tensor,
    model: SyntheticScoreModel,
    tau_gt_ms: float | None,
) -> dict:
    curves: list[dict] = []
    for x_window in windows:
        lags_ms, interactions = scan_curve(x_window, model, baseline)
        stats = curve_statistics(lags_ms, interactions, DETECTOR_CONFIG)
        curves.append(stats)
    return {"statistics": curves, "tau_gt_ms": tau_gt_ms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="NinaPro DB2 E1 npy 目录")
    parser.add_argument("--smoke", action="store_true", help="冒烟：3 seeds × 20 窗口/条件")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--alpha-coef", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=3e-4)
    parser.add_argument("--tau-gt-ms", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "a1_matching_simulation")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = min(args.seeds, 3)
        args.windows = min(args.windows, 20)

    subjects = list(range(1, 41))
    tau_gt_samples = int(round(args.tau_gt_ms * FS / 1000))
    rng = np.random.default_rng(20260821)

    null_model = SyntheticScoreModel(args.alpha_coef, args.beta, None, args.noise_std)
    inter_model = SyntheticScoreModel(args.alpha_coef, args.beta, tau_gt_samples, args.noise_std)

    def supported_by_thresholds(stats: dict, thresholds: NullThresholds) -> bool:
        return all(
            stats[stat] > thresholds.threshold_for(stat)
            for stat in DETECTOR_CONFIG.gated_statistics
        )

    all_null_stats: list[dict] = []
    all_inter_stats: list[dict] = []
    per_seed: dict[str, list] = {"null_support": [], "inter_support": [], "inter_recover": []}

    t0 = time.time()
    for seed_index in range(args.seeds):
        seed = int(rng.integers(0, 2**31))
        windows, baseline = sample_windows(
            args.data_root, subjects, args.windows, seed
        )
        null_out = run_condition(windows, baseline, null_model, None)
        inter_out = run_condition(windows, baseline, inter_model, args.tau_gt_ms)
        all_null_stats.extend(null_out["statistics"])
        all_inter_stats.extend(inter_out["statistics"])
        if (seed_index + 1) % 5 == 0 or seed_index == args.seeds - 1:
            print(f"seed {seed_index + 1}/{args.seeds} done ({time.time() - t0:.0f}s)", flush=True)

    # null 校准（只允许 null 统计量进入）
    thresholds = calibrate_thresholds(all_null_stats, DETECTOR_CONFIG)

    # 判决：所有门控统计量超过零分布阈值才算支持（与 detect_peak 同判据）
    null_supported = np.array(
        [supported_by_thresholds(s, thresholds) for s in all_null_stats], dtype=bool
    )
    inter_supported = np.array(
        [supported_by_thresholds(s, thresholds) for s in all_inter_stats], dtype=bool
    )
    inter_peaks = np.array([s["peak_lag_ms"] for s in all_inter_stats], dtype=float)
    recovered = np.array(
        [
            (sup and abs(peak - args.tau_gt_ms) <= 5.0)
            for sup, peak in zip(inter_supported, inter_peaks)
        ],
        dtype=bool,
    )

    n = len(all_null_stats)
    fp_rate = float(null_supported.mean())
    det_rate = float(recovered.mean())
    support_inter = float(inter_supported.mean())
    no_support_null = float((~null_supported).mean())
    peak_err = inter_peaks[recovered] - args.tau_gt_ms if recovered.any() else np.array([])

    summary = {
        "run_id": "A1",
        "pipeline_matching": {
            "window_samples": WINDOW_SAMPLES,
            "block_samples": BLOCK_SAMPLES,
            "exclusion_ms": DETECTOR_CONFIG.exclusion_ms,
            "smooth_sigma_bins": SMOOTH_SIGMA_BINS,
            "max_lag_ms": MAX_LAG_MS,
            "stride_ms": 0.5,
            "alpha": ALPHA,
            "threshold_source": "null-model (no-interaction) curves only",
            "code_path": "ST_SRI_Interpreter.scan_fast + st_sri_detector",
        },
        "model_config": {
            "alpha_coef": args.alpha_coef,
            "beta": args.beta,
            "noise_std": args.noise_std,
            "tau_gt_ms": args.tau_gt_ms,
            "tau_gt_samples": tau_gt_samples,
        },
        "signal": {"source": "NinaPro DB2 E1 (12 ch, 2000 Hz, real autocorrelation)", "n_windows": n, "seeds": args.seeds},
        "null_thresholds": thresholds.thresholds,
        "results": {
            "detection_rate_gt_plusminus_5ms": det_rate,
            "n_detected": int(recovered.sum()),
            "inter_support_rate": support_inter,
            "false_positive_rate_null": fp_rate,
            "no_support_fraction_null": no_support_null,
            "n_null": n,
            "n_inter": n,
            "peak_error_ms_mean": float(peak_err.mean()) if peak_err.size else None,
            "peak_error_ms_median": float(np.median(peak_err)) if peak_err.size else None,
            "peak_error_ms_p25_p75": [float(np.percentile(peak_err, 25)), float(np.percentile(peak_err, 75))] if peak_err.size else None,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "a1_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["results"], ensure_ascii=False, indent=2))
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
