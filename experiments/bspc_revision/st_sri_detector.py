"""R006/R007：全流程零分布校准与可拒绝的 ST-SRI 峰值检测器（纯 CPU）。

本模块替代旧 `common.detect_st_sri_peak_ms` 的强制 argmax 选峰：

- 先移除与当前块重叠的滞后分箱（|lag| 小于块宽），再对保留分箱平滑，
  旧实现顺序相反，会把重叠区伪峰泄漏进有资格区域；
- 支持阈值只由独立零分布（零模型曲线或解析置乱代理）预先标定，
  不根据训练模型曲线调节；
- 连续交互质量未超过零分布阈值时明确报告“无支持峰”，
  而不是对每个谱强制分配一个峰值。

输入是已采集的滞后剖面曲线（npz），不依赖 torch 或 GPU；
曲线采集（模型前向扫描）由后续 R006/R008 运行脚本完成。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATISTIC_NAMES = ("positive_mass", "signed_mass", "peak_height", "peak_prominence")
SURROGATE_MODES = ("permute", "shift")


@dataclass(frozen=True)
class DetectorConfig:
    """检测器协议参数；默认值与 R003-R005 使用的扫描协议一致。"""

    fs: int = 2000
    block_samples: int = 10
    max_lag_ms: float = 150.0
    stride_samples: int = 1
    smooth_sigma_bins: float = 2.0
    alpha: float = 0.05
    gated_statistics: tuple[str, ...] = ("positive_mass", "peak_height")

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        if self.block_samples < 1:
            raise ValueError("block_samples must be positive")
        if self.stride_samples < 1:
            raise ValueError("stride_samples must be positive")
        if self.max_lag_ms <= 0:
            raise ValueError("max_lag_ms must be positive")
        if self.smooth_sigma_bins < 0:
            raise ValueError("smooth_sigma_bins must be nonnegative")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        unknown = set(self.gated_statistics) - set(STATISTIC_NAMES)
        if unknown:
            raise ValueError(f"unknown gated statistics: {sorted(unknown)}")
        if not self.gated_statistics:
            raise ValueError("gated_statistics must not be empty")

    @property
    def exclusion_ms(self) -> float:
        """重叠排除阈值：滞后块与当前块不重叠要求 |lag| >= 块宽。"""
        return self.block_samples * 1000.0 / self.fs

    @property
    def bin_width_ms(self) -> float:
        return self.stride_samples * 1000.0 / self.fs


@dataclass(frozen=True)
class NullThresholds:
    """由零分布统计量预先标定的支持阈值。"""

    alpha: float
    n_null: int
    thresholds: dict[str, float]

    def threshold_for(self, statistic: str) -> float:
        if statistic not in self.thresholds:
            raise KeyError(f"no null threshold calibrated for {statistic!r}")
        return self.thresholds[statistic]


@dataclass(frozen=True)
class DetectionResult:
    """单条滞后剖面的检测判决。"""

    supported: bool
    peak_lag_ms: float | None
    statistics: dict[str, float]
    reason: str

    def to_dict(self) -> dict:
        return {
            "supported": self.supported,
            "peak_lag_ms": self.peak_lag_ms,
            "statistics": {k: float(v) for k, v in self.statistics.items()},
            "reason": self.reason,
        }


def _validate_curve(lags_ms: np.ndarray, curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lags = np.asarray(lags_ms, dtype=np.float64).ravel()
    values = np.asarray(curve, dtype=np.float64).ravel()
    if lags.shape != values.shape:
        raise ValueError(f"lags_ms and curve must have the same length, got {lags.shape} and {values.shape}")
    if lags.size and (not np.all(np.isfinite(lags)) or not np.all(np.isfinite(values))):
        raise ValueError("lags_ms and curve must be finite")
    if lags.size > 1 and np.any(np.diff(lags) <= 0):
        raise ValueError("lags_ms must be strictly increasing")
    return lags, values


def eligible_lag_mask(lags_ms: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """有资格滞后区域：|lag| >= 块宽（滞后块与当前块不重叠）。"""
    lags = np.asarray(lags_ms, dtype=np.float64).ravel()
    return np.abs(lags) >= config.exclusion_ms


def exclude_overlaps(
    lags_ms: np.ndarray,
    curve: np.ndarray,
    config: DetectorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """第一步：移除与当前块重叠的滞后分箱，不做任何平滑。"""
    lags, values = _validate_curve(lags_ms, curve)
    mask = eligible_lag_mask(lags, config)
    max_lag_mask = np.abs(lags) <= config.max_lag_ms
    keep = mask & max_lag_mask
    return lags[keep], values[keep]


def smooth_values(values: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """第二步：只对保留分箱做高斯平滑。"""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or config.smooth_sigma_bins <= 0:
        return values
    return gaussian_filter1d(values, sigma=config.smooth_sigma_bins)


def prepare_curve(
    lags_ms: np.ndarray,
    curve: np.ndarray,
    config: DetectorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """R007 规定的处理顺序：先排除重叠滞后，再对保留分箱平滑。"""
    eligible_lags, eligible_values = exclude_overlaps(lags_ms, curve, config)
    return eligible_lags, smooth_values(eligible_values, config)


def curve_statistics(
    lags_ms: np.ndarray,
    curve: np.ndarray,
    config: DetectorConfig,
) -> dict[str, float]:
    """在有资格区域的平滑曲线上计算连续交互质量统计量。

    - positive_mass：正部质量（连续正质量，按分箱宽度积分）；
    - signed_mass：有符号质量（对有符号交互曲线才有判别意义）；
    - peak_height：峰值显著度（平滑曲线最大值）；
    - peak_prominence：峰值相对有资格区域中位数的超出量；
    - peak_lag_ms：峰值所在滞后（搜索方式与零分布完全一致）。
    """
    eligible_lags, prepared = prepare_curve(lags_ms, curve, config)
    if eligible_lags.size == 0:
        raise ValueError("no eligible lag bins after overlap exclusion")
    peak_index = int(np.argmax(prepared))
    positive = np.clip(prepared, 0.0, None)
    return {
        "positive_mass": float(positive.sum() * config.bin_width_ms),
        "signed_mass": float(prepared.sum() * config.bin_width_ms),
        "peak_height": float(prepared[peak_index]),
        "peak_prominence": float(prepared[peak_index] - np.median(prepared)),
        "peak_lag_ms": float(eligible_lags[peak_index]),
    }


def surrogate_statistics(
    lags_ms: np.ndarray,
    curve: np.ndarray,
    config: DetectorConfig,
    n_surrogates: int,
    rng: np.random.Generator,
    mode: str = "permute",
) -> list[dict[str, float]]:
    """对单条曲线生成与真实处理链一致的解析置乱零分布。

    先按真实协议排除重叠滞后，再对原始保留分箱置乱
    （permute 破坏滞后结构、shift 破坏峰值位置），
    然后走与真实曲线完全相同的平滑和 argmax 搜索。
    """
    if mode not in SURROGATE_MODES:
        raise ValueError(f"mode must be one of {SURROGATE_MODES}")
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be positive")
    eligible_lags, eligible_values = exclude_overlaps(lags_ms, curve, config)
    if eligible_lags.size == 0:
        raise ValueError("no eligible lag bins after overlap exclusion")
    statistics = []
    for _ in range(n_surrogates):
        if mode == "permute":
            surrogate = rng.permutation(eligible_values)
        else:
            shift = int(rng.integers(1, eligible_values.size)) if eligible_values.size > 1 else 0
            surrogate = np.roll(eligible_values, shift)
        prepared = smooth_values(surrogate, config)
        peak_index = int(np.argmax(prepared))
        positive = np.clip(prepared, 0.0, None)
        statistics.append(
            {
                "positive_mass": float(positive.sum() * config.bin_width_ms),
                "signed_mass": float(prepared.sum() * config.bin_width_ms),
                "peak_height": float(prepared[peak_index]),
                "peak_prominence": float(prepared[peak_index] - np.median(prepared)),
                "peak_lag_ms": float(eligible_lags[peak_index]),
            }
        )
    return statistics


def calibrate_thresholds(
    null_statistics: Sequence[dict[str, float]],
    config: DetectorConfig,
) -> NullThresholds:
    """从零分布统计量标定各门控统计量的 (1-alpha) 分位数阈值。

    只允许传入零分布（零模型或置乱代理）的统计量；
    训练模型曲线不得进入本函数。
    """
    if len(null_statistics) == 0:
        raise ValueError("null_statistics must not be empty")
    thresholds = {}
    for statistic in config.gated_statistics:
        values = np.array([entry[statistic] for entry in null_statistics], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"null statistic {statistic!r} contains non-finite values")
        thresholds[statistic] = float(np.quantile(values, 1.0 - config.alpha, method="higher"))
    return NullThresholds(alpha=config.alpha, n_null=len(null_statistics), thresholds=thresholds)


def detect_peak(
    lags_ms: np.ndarray,
    curve: np.ndarray,
    config: DetectorConfig,
    thresholds: NullThresholds,
) -> DetectionResult:
    """可拒绝的峰值检测：所有门控统计量超过零分布阈值才报告峰值。"""
    for statistic in config.gated_statistics:
        thresholds.threshold_for(statistic)
    eligible_lags, prepared = prepare_curve(lags_ms, curve, config)
    if eligible_lags.size == 0:
        return DetectionResult(
            supported=False,
            peak_lag_ms=None,
            statistics={name: float("nan") for name in STATISTIC_NAMES} | {"peak_lag_ms": float("nan")},
            reason="no_eligible_bins",
        )
    statistics = curve_statistics(lags_ms, curve, config)
    failing = [
        statistic
        for statistic in config.gated_statistics
        if statistics[statistic] <= thresholds.threshold_for(statistic)
    ]
    if failing:
        return DetectionResult(
            supported=False,
            peak_lag_ms=None,
            statistics=statistics,
            reason="below_null_threshold:" + ",".join(failing),
        )
    return DetectionResult(
        supported=True,
        peak_lag_ms=statistics["peak_lag_ms"],
        statistics=statistics,
        reason="supported",
    )


def summarize_decisions(decisions: Sequence[DetectionResult]) -> dict:
    """汇总支持率和无支持峰比例。"""
    if len(decisions) == 0:
        raise ValueError("decisions must not be empty")
    supported = [decision for decision in decisions if decision.supported]
    reason_counts: dict[str, int] = {}
    for decision in decisions:
        reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
    supported_lags = [decision.peak_lag_ms for decision in supported if decision.peak_lag_ms is not None]
    return {
        "n_curves": len(decisions),
        "n_supported": len(supported),
        "support_rate": len(supported) / len(decisions),
        "no_support_fraction": 1.0 - len(supported) / len(decisions),
        "reason_counts": dict(sorted(reason_counts.items())),
        "supported_peak_lag_ms": {
            "median": float(np.median(supported_lags)) if supported_lags else None,
            "min": float(np.min(supported_lags)) if supported_lags else None,
            "max": float(np.max(supported_lags)) if supported_lags else None,
        },
    }


def _load_curves_npz(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    archive = np.load(path, allow_pickle=False)
    if "lags_ms" not in archive or "curves" not in archive:
        raise ValueError(f"{path} must contain 'lags_ms' and 'curves' arrays")
    lags_ms = np.asarray(archive["lags_ms"], dtype=np.float64)
    curves = np.asarray(archive["curves"], dtype=np.float64)
    if curves.ndim != 2 or curves.shape[1] != lags_ms.size:
        raise ValueError(
            f"{path}: curves must have shape (n_curves, {lags_ms.size}), got {curves.shape}"
        )
    if "ids" in archive:
        ids = [str(entry) for entry in archive["ids"]]
        if len(ids) != curves.shape[0]:
            raise ValueError(f"{path}: ids length does not match curves")
    else:
        ids = [f"curve_{index}" for index in range(curves.shape[0])]
    return lags_ms, curves, ids


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, required=True, help="待检测曲线的 npz（lags_ms + curves [+ ids]）")
    parser.add_argument(
        "--null-curves",
        type=Path,
        default=None,
        help="零模型（标签打乱/重初始化/参数置乱）曲线 npz；R006 主零分布来源",
    )
    parser.add_argument(
        "--surrogate-count",
        type=int,
        default=0,
        help="对每条待检测曲线生成的解析置乱代理数；无零模型曲线时必须大于 0",
    )
    parser.add_argument(
        "--surrogate-mode",
        choices=SURROGATE_MODES,
        default="permute",
        help="解析置乱方式",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--smooth-sigma-bins", type=float, default=2.0)
    parser.add_argument("--block-samples", type=int, default=10)
    parser.add_argument("--max-lag-ms", type=float, default=150.0)
    parser.add_argument("--stride-samples", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r006_detector_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectorConfig(
        block_samples=args.block_samples,
        max_lag_ms=args.max_lag_ms,
        stride_samples=args.stride_samples,
        smooth_sigma_bins=args.smooth_sigma_bins,
        alpha=args.alpha,
    )
    lags_ms, curves, ids = _load_curves_npz(args.curves)

    null_source: str
    if args.null_curves is not None:
        null_lags, null_curves, _ = _load_curves_npz(args.null_curves)
        if not np.array_equal(null_lags, lags_ms):
            raise ValueError("null curves must share the same lag axis as the target curves")
        null_statistics = [curve_statistics(null_lags, null_curves[i], config) for i in range(null_curves.shape[0])]
        null_source = "null_model_curves"
    elif args.surrogate_count > 0:
        rng = np.random.default_rng(args.seed)
        null_statistics = []
        for index in range(curves.shape[0]):
            null_statistics.extend(
                surrogate_statistics(lags_ms, curves[index], config, args.surrogate_count, rng, mode=args.surrogate_mode)
            )
        null_source = f"surrogate_{args.surrogate_mode}_x{args.surrogate_count}"
    else:
        raise SystemExit("必须提供 --null-curves 或设置 --surrogate-count > 0")

    thresholds = calibrate_thresholds(null_statistics, config)
    decisions = [detect_peak(lags_ms, curves[index], config, thresholds) for index in range(curves.shape[0])]
    summary = summarize_decisions(decisions)

    payload = {
        "run_id": "R006/R007",
        "null_source": null_source,
        "null_threshold_note": "阈值只由零分布标定，未使用待检测曲线调节",
        "config": asdict(config),
        "thresholds": asdict(thresholds),
        "summary": summary,
        "decisions": [
            {"id": curve_id, **decision.to_dict()} for curve_id, decision in zip(ids, decisions)
        ],
    }
    write_json_atomic(args.output, payload)
    print(
        f"support_rate={summary['support_rate']:.4f} "
        f"no_support_fraction={summary['no_support_fraction']:.4f} "
        f"n={summary['n_curves']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
