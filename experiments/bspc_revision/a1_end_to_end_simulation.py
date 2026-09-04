"""A1: End-to-end matching simulation for ST-SRI.

Simulates the full ST-SRI analysis pipeline on synthetic multichannel signals:
- 10-sample blocks
- overlapping lag scan
- substitution construction (both / lag / current / none)
- positive truncation (synergy = max(interaction, 0))
- remove lags < 5 ms, then Gaussian smoothing
- peak finding with boundary handling
- aggregation and detection metrics

Run CPU smoke:
    python a1_end_to_end_simulation.py --n-null 20 --n-signal 20 --seeds 0 1 --smoke
Full:
    python a1_end_to_end_simulation.py --n-null 500 --n-signal 500 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

FS = 2000
WINDOW_SAMPLES = 600
BLOCK_SAMPLES = 10
MIN_LAG_MS = 5.0
MAX_LAG_MS = 150.0
STRIDE_SAMPLES = 1
SMOOTH_SIGMA = 2.0
N_CHANNELS = 12


def make_ar_signal(n_samples: int, n_channels: int, seed: int) -> np.ndarray:
    """Generate real-like multichannel AR(1) signals with cross-channel correlation."""
    rng = np.random.default_rng(seed)
    # channel covariance
    a = rng.normal(0, 1, size=(n_channels, n_channels))
    cov = a @ a.T + np.eye(n_channels) * 0.1
    L = np.linalg.cholesky(cov)
    x = np.zeros((n_samples, n_channels), dtype=np.float64)
    phi = rng.uniform(0.7, 0.95, size=n_channels)
    innovations = rng.normal(0, 1, size=(n_samples, n_channels)) @ L.T
    for t in range(1, n_samples):
        x[t] = phi * x[t - 1] + innovations[t]
    return x


def _block_energy(x: np.ndarray, start: int, end: int) -> float:
    block = x[start:end]
    return float(np.mean(np.abs(block)))


def make_score_function(with_interaction: bool, lag_samples: int, strength: float = 2.0):
    """Return score(w) -> float in [0,1] based on current and lag block energies."""
    def score(w: np.ndarray) -> float:
        current = _block_energy(w, WINDOW_SAMPLES - BLOCK_SAMPLES, WINDOW_SAMPLES)
        lag_start = WINDOW_SAMPLES - lag_samples - BLOCK_SAMPLES
        lag = _block_energy(w, lag_start, lag_start + BLOCK_SAMPLES)
        # Linear score: ensures the four-term substitution interaction equals
        # strength * current * lag when with_interaction=True, and ~0 when False.
        if with_interaction:
            return 1.0 + current + lag + strength * current * lag
        return 1.0 + current + lag
    return score


def substitution_curve(x: np.ndarray, score, lag_samples: int, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run overlapping lag scan and return lags_ms and synergy curve (positive-truncated, smoothed)."""
    max_lag = int(MAX_LAG_MS * FS / 1000)
    min_lag = max(1, int(MIN_LAG_MS * FS / 1000))
    lags = list(range(min_lag, max_lag + 1, STRIDE_SAMPLES))
    interactions = []
    x = np.asarray(x, dtype=np.float64)
    for tau in lags:
        x_lag = x.copy()
        x_curr = x.copy()
        x_none = x.copy()
        # current block at the end
        x_lag[WINDOW_SAMPLES - BLOCK_SAMPLES:WINDOW_SAMPLES] = baseline[WINDOW_SAMPLES - BLOCK_SAMPLES:WINDOW_SAMPLES]
        x_none[WINDOW_SAMPLES - BLOCK_SAMPLES:WINDOW_SAMPLES] = baseline[WINDOW_SAMPLES - BLOCK_SAMPLES:WINDOW_SAMPLES]
        # lag block
        lag_start = WINDOW_SAMPLES - tau - BLOCK_SAMPLES
        x_curr[lag_start:lag_start + BLOCK_SAMPLES] = baseline[lag_start:lag_start + BLOCK_SAMPLES]
        x_none[lag_start:lag_start + BLOCK_SAMPLES] = baseline[lag_start:lag_start + BLOCK_SAMPLES]

        f_both = score(x)
        f_lag = score(x_lag)
        f_curr = score(x_curr)
        f_none = score(x_none)
        interactions.append(f_both - f_lag - f_curr + f_none)

    interactions = np.asarray(interactions, dtype=np.float64)
    synergy = np.maximum(interactions, 0.0)
    # smoothing after removing <5ms (already excluded by min_lag)
    synergy_smooth = gaussian_filter1d(synergy, sigma=SMOOTH_SIGMA)
    lags_ms = np.asarray(lags, dtype=np.float64) * (1000.0 / FS)
    return lags_ms, synergy_smooth


def summarize_curve(lags_ms: np.ndarray, curve: np.ndarray) -> dict:
    valid = (lags_ms >= MIN_LAG_MS) & (lags_ms <= MAX_LAG_MS)
    if not np.any(valid):
        return {"positive_mass": 0.0, "peak_height": 0.0, "peak_lag_ms": None}
    c = curve[valid]
    positive_mass = float(np.sum(c))
    peak_height = float(np.max(c)) if c.size else 0.0
    peak_idx = int(np.argmax(c)) if c.size else None
    peak_lag = float(lags_ms[valid][peak_idx]) if peak_idx is not None else None
    return {"positive_mass": positive_mass, "peak_height": peak_height, "peak_lag_ms": peak_lag}


def simulate_set(n: int, with_interaction: bool, seeds: list[int], lag_ms: float = 30.0, strength: float = 2.0) -> list[dict]:
    lag_samples = int(round(lag_ms * FS / 1000))
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for i in range(n):
            s = int(rng.integers(0, 10**9))
            x = make_ar_signal(WINDOW_SAMPLES, N_CHANNELS, seed=s)
            # neutral baseline: low-activity mean-like signal (zeros)
            baseline = np.zeros_like(x)
            score = make_score_function(with_interaction, lag_samples, strength=strength)
            lags_ms, curve = substitution_curve(x, score, lag_samples, baseline)
            summary = summarize_curve(lags_ms, curve)
            summary["seed"] = s
            results.append(summary)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--n-signal", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--lag-ms", type=float, default=30.0)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/a1_simulation.json"))
    args = parser.parse_args()

    if args.smoke:
        args.n_null = min(args.n_null, 10)
        args.n_signal = min(args.n_signal, 10)
        args.seeds = args.seeds[:2]

    print("Running A1 simulation...", flush=True)
    null_results = simulate_set(args.n_null, False, args.seeds, args.lag_ms, args.strength)
    signal_results = simulate_set(args.n_signal, True, args.seeds, args.lag_ms, args.strength)

    null_pm = np.array([r["positive_mass"] for r in null_results])
    null_ph = np.array([r["peak_height"] for r in null_results])
    pm_thr = float(np.quantile(null_pm, 0.95)) if null_pm.size else 0.0
    ph_thr = float(np.quantile(null_ph, 0.95)) if null_ph.size else 0.0

    def is_supported(r):
        return r["positive_mass"] > pm_thr or r["peak_height"] > ph_thr

    fpr = float(np.mean([is_supported(r) for r in null_results]))
    tpr = float(np.mean([is_supported(r) for r in signal_results]))
    no_support_signal = float(np.mean([not is_supported(r) for r in signal_results]))

    result = {
        "config": {
            "fs": FS,
            "window_samples": WINDOW_SAMPLES,
            "block_samples": BLOCK_SAMPLES,
            "min_lag_ms": MIN_LAG_MS,
            "max_lag_ms": MAX_LAG_MS,
            "stride_samples": STRIDE_SAMPLES,
            "smooth_sigma": SMOOTH_SIGMA,
            "n_null": len(null_results),
            "n_signal": len(signal_results),
            "seeds": args.seeds,
            "lag_ms": args.lag_ms,
            "strength": args.strength,
            "smoke": args.smoke,
        },
        "thresholds": {"positive_mass": pm_thr, "peak_height": ph_thr},
        "metrics": {
            "false_positive_rate": fpr,
            "detection_rate": tpr,
            "no_support_fraction_signal": no_support_signal,
            "null_positive_mass_median": float(np.median(null_pm)) if null_pm.size else None,
            "signal_positive_mass_median": float(np.median([r["positive_mass"] for r in signal_results])) if signal_results else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
