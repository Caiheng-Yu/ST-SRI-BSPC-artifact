"""R016/A1: validate the current ST-SRI detector on synthetic mechanisms.

This is deliberately separate from the historical E1 result.  It keeps the
seven original mechanisms and 40 windows per mechanism, but runs the current
R006/R007 analysis chain: ten-sample complete-channel blocks, overlap
exclusion before smoothing, and thresholds calibrated from independent
threshold-negative control windows.

The output is a machine-readable audit artifact.  It is a synthetic proxy
evaluation, not evidence about DB2 physiology or model generalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import ST_SRI_Interpreter  # noqa: E402
from experiments.bspc_revision.st_sri_detector import (  # noqa: E402
    DetectorConfig,
    calibrate_thresholds,
    curve_statistics,
    detect_peak,
    prepare_curve,
)


FS = 2000
WINDOW_SAMPLES = 600
CHANNELS = 12
DEFAULT_SEED = 20260515
MECHANISM_WINDOWS = 40
CALIBRATION_WINDOWS = 160
CALIBRATION_SEED_OFFSET = 9173


@dataclass(frozen=True)
class MechanismSpec:
    mechanism_id: str
    gt_lags_ms: tuple[float, ...]
    interaction: bool
    noise_std: float = 0.0
    amp_jitter: float = 0.0


MECHANISMS = (
    MechanismSpec("current_only", (50.0,), False),
    MechanismSpec("history_only", (50.0,), False),
    MechanismSpec("additive_current_history", (50.0,), False),
    MechanismSpec("multiplicative_lagged", (50.0,), True),
    MechanismSpec("multi_lag_interaction", (30.0, 90.0), True),
    MechanismSpec("null_no_interaction", (50.0,), False),
    MechanismSpec("noisy_lagged_interaction", (70.0,), True, 0.35, 0.15),
)


def ms_to_samples(ms: float) -> int:
    return int(round(ms * FS / 1000.0))


class SyntheticClassifier(nn.Module):
    def __init__(self, spec: MechanismSpec) -> None:
        super().__init__()
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(x.shape[0], device=x.device)
        if self.spec.mechanism_id == "current_only":
            score = 6.0 * x[:, -1, 0]
        elif self.spec.mechanism_id == "history_only":
            tau = ms_to_samples(self.spec.gt_lags_ms[0])
            score = 6.0 * x[:, -1 - tau, 1]
        elif self.spec.mechanism_id == "additive_current_history":
            tau = ms_to_samples(self.spec.gt_lags_ms[0])
            score = 0.2 * (x[:, -1, 0] + x[:, -1 - tau, 1])
        elif self.spec.mechanism_id == "multiplicative_lagged":
            tau = ms_to_samples(self.spec.gt_lags_ms[0])
            score = 8.0 * x[:, -1, 0] * x[:, -1 - tau, 1]
        elif self.spec.mechanism_id == "multi_lag_interaction":
            tau1, tau2 = (ms_to_samples(value) for value in self.spec.gt_lags_ms)
            score = 1.5 * (
                x[:, -1, 0] * x[:, -1 - tau1, 1]
                + x[:, -1, 2] * x[:, -1 - tau2, 3]
            )
        elif self.spec.mechanism_id == "null_no_interaction":
            score = zero
        elif self.spec.mechanism_id == "noisy_lagged_interaction":
            tau = ms_to_samples(self.spec.gt_lags_ms[0])
            score = 8.0 * x[:, -1, 0] * x[:, -1 - tau, 1]
        else:
            raise ValueError(f"unknown mechanism: {self.spec.mechanism_id}")
        return torch.stack([zero, score], dim=1)


def generate_windows(spec: MechanismSpec, count: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []

    def amplitude() -> float:
        if spec.amp_jitter == 0:
            return 1.0
        return float(max(0.4, rng.normal(1.0, spec.amp_jitter)))

    for _ in range(count):
        x = rng.normal(0.0, spec.noise_std, size=(WINDOW_SAMPLES, CHANNELS)).astype(np.float32)
        if spec.mechanism_id == "multi_lag_interaction":
            tau1, tau2 = (ms_to_samples(value) for value in spec.gt_lags_ms)
            x[-1, 0] += amplitude()
            x[-1 - tau1, 1] += amplitude()
            x[-1, 2] += amplitude()
            x[-1 - tau2, 3] += amplitude()
        else:
            tau = ms_to_samples(spec.gt_lags_ms[0])
            x[-1, 0] += amplitude()
            x[-1 - tau, 1] += amplitude()
        windows.append(x)
    return torch.from_numpy(np.stack(windows, axis=0))


def scan_windows(
    model: nn.Module,
    background: torch.Tensor,
    windows: torch.Tensor,
    config: DetectorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    interpreter = ST_SRI_Interpreter(model, background, device="cpu")
    curves: list[np.ndarray] = []
    lags_ms: np.ndarray | None = None
    for window in windows:
        lags, synergy, _ = interpreter.scan_fast(
            window,
            max_lag_ms=config.max_lag_ms,
            stride=config.stride_samples,
            block_size=config.block_samples,
            current_endpoint=WINDOW_SAMPLES - 1,
        )
        lags_ms = np.asarray(lags, dtype=np.float64)
        curves.append(np.asarray(synergy, dtype=np.float64))
    if lags_ms is None:
        raise ValueError("no synthetic windows were generated")
    return lags_ms, np.stack(curves, axis=0)


def supported_peaks(
    lags_ms: np.ndarray,
    curve: np.ndarray,
    config: DetectorConfig,
    peak_threshold: float,
) -> list[float]:
    eligible_lags, prepared = prepare_curve(lags_ms, curve, config)
    if prepared.size == 0:
        return []
    distance = max(1, int(round(8.0 / config.bin_width_ms)))
    indices, _ = find_peaks(prepared, height=peak_threshold, distance=distance)
    ordered = sorted(indices.tolist(), key=lambda index: prepared[index], reverse=True)
    return [float(eligible_lags[index]) for index in ordered]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--windows-per-mechanism", type=int, default=MECHANISM_WINDOWS)
    parser.add_argument("--calibration-windows", type=int, default=CALIBRATION_WINDOWS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "a1_synthetic_detector_validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(2)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_config = DetectorConfig(
        fs=FS,
        block_samples=10,
        max_lag_ms=150.0,
        stride_samples=1,
        smooth_sigma_bins=2.0,
        alpha=0.05,
        gated_statistics=("positive_mass", "peak_height"),
    )
    background = torch.zeros((32, WINDOW_SAMPLES, CHANNELS), dtype=torch.float32)

    # The negative calibration set is independent of the reported windows.
    calibration_spec = next(
        spec for spec in MECHANISMS if spec.mechanism_id == "additive_current_history"
    )
    calibration_model = SyntheticClassifier(calibration_spec)
    calibration_windows = generate_windows(
        calibration_spec,
        args.calibration_windows,
        args.seed + CALIBRATION_SEED_OFFSET,
    )
    lags_ms, calibration_curves = scan_windows(
        calibration_model,
        background,
        calibration_windows,
        detector_config,
    )
    calibration_statistics = [
        curve_statistics(lags_ms, curve, detector_config) for curve in calibration_curves
    ]
    thresholds = calibrate_thresholds(calibration_statistics, detector_config)

    rows: list[dict[str, object]] = []
    mechanism_summaries: list[dict[str, object]] = []
    for mechanism_index, spec in enumerate(MECHANISMS):
        model = SyntheticClassifier(spec)
        windows = generate_windows(
            spec,
            args.windows_per_mechanism,
            args.seed + mechanism_index * 1009,
        )
        run_lags, curves = scan_windows(model, background, windows, detector_config)
        if not np.array_equal(run_lags, lags_ms):
            raise AssertionError(f"lag grid changed for {spec.mechanism_id}")

        decisions = []
        for window_index, curve in enumerate(curves):
            decision = detect_peak(run_lags, curve, detector_config, thresholds)
            peaks = supported_peaks(
                run_lags,
                curve,
                detector_config,
                thresholds.threshold_for("peak_height"),
            )
            recovered = bool(
                spec.interaction
                and all(any(abs(peak - gt) <= 5.0 for peak in peaks) for gt in spec.gt_lags_ms)
            )
            row = {
                "mechanism_id": spec.mechanism_id,
                "window": window_index,
                "interaction": spec.interaction,
                "supported": decision.supported,
                "peak_lag_ms": decision.peak_lag_ms,
                "recovered_within_5ms": recovered,
                "top_supported_peaks_ms": ";".join(f"{peak:.3f}" for peak in peaks),
                "reason": decision.reason,
                "positive_mass": decision.statistics.get("positive_mass"),
                "peak_height": decision.statistics.get("peak_height"),
            }
            rows.append(row)
            decisions.append(decision)

        n = len(decisions)
        supported_count = sum(int(decision.supported) for decision in decisions)
        recovery_count = sum(int(row["recovered_within_5ms"]) for row in rows if row["mechanism_id"] == spec.mechanism_id)
        mechanism_summaries.append(
            {
                "mechanism_id": spec.mechanism_id,
                "gt_lags_ms": list(spec.gt_lags_ms),
                "true_current_history_interaction": spec.interaction,
                "n_windows": n,
                "n_supported": supported_count,
                "support_rate": supported_count / n,
                "n_recovered_within_5ms": recovery_count if spec.interaction else None,
                "recovery_rate_within_5ms": recovery_count / n if spec.interaction else None,
                "false_positive_rate": supported_count / n if not spec.interaction else None,
                "reason_counts": {
                    reason: sum(int(decision.reason == reason) for decision in decisions)
                    for reason in sorted({decision.reason for decision in decisions})
                },
            }
        )

    csv_path = output_dir / "per_window.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "R016/A1 current-protocol synthetic detector validation",
        "evaluation_type": "synthetic_proxy",
        "config": {
            "seed": args.seed,
            "mechanism_count": len(MECHANISMS),
            "windows_per_mechanism": args.windows_per_mechanism,
            "calibration_windows": args.calibration_windows,
            "window_samples": WINDOW_SAMPLES,
            "channels": CHANNELS,
            "fs_hz": FS,
            "detector": asdict(detector_config),
            "calibration_model": "independent additive threshold-negative controls",
            "calibration_seed": args.seed + CALIBRATION_SEED_OFFSET,
            "recovery_tolerance_ms": 5.0,
            "peak_separation_ms": 8.0,
        },
        "thresholds": asdict(thresholds),
        "mechanisms": mechanism_summaries,
        "aggregate": {
            "interaction_recovery_rate": float(
                np.mean([
                    item["recovery_rate_within_5ms"]
                    for item in mechanism_summaries
                    if item["recovery_rate_within_5ms"] is not None
                ])
            ),
            "negative_control_false_positive_rate": float(
                np.mean([
                    item["false_positive_rate"]
                    for item in mechanism_summaries
                    if item["false_positive_rate"] is not None
                ])
            ),
            "negative_control_windows": sum(
                item["n_windows"]
                for item in mechanism_summaries
                if not item["true_current_history_interaction"]
            ),
        },
        "claim_boundary": "Synthetic proxy only; does not establish DB2 timing or physiology.",
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config_path = output_dir / "resolved_config.json"
    config_path.write_text(json.dumps(summary["config"], indent=2), encoding="utf-8")

    receipt = {
        "schema_version": "bspc-r016-a1-v1",
        "status": "completed",
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(Path(__file__)),
        "outputs": {
            "summary": summary_path.name,
            "config": config_path.name,
            "per_window": csv_path.name,
        },
        "output_sha256": {
            summary_path.name: sha256_file(summary_path),
            config_path.name: sha256_file(config_path),
            csv_path.name: sha256_file(csv_path),
        },
    }
    (output_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print(json.dumps(summary["aggregate"], indent=2))
    print(json.dumps({"output_dir": str(output_dir), "status": "completed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
