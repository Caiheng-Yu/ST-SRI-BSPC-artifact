"""R017: current-protocol DB2 fixed-checkpoint replay and origin diagnostic.

This analysis reconstructs the manuscript's DB2 fixed-checkpoint section from
the current BSPC audit curves.  It deliberately does not reuse the historical
839-window numbers.  The current replay contains 40 R005b checkpoints, a
4/1/1 repetition split, 17 audit repetitions per participant, and four causal
onset positions (0/50/100/150 ms), for 2,720 curves.

The circular-origin calculation is a conditional spectral-origin diagnostic:
each participant mean spectrum is rotated over all available origins and the
same lower-bound peak rule is reapplied.  It is not a signal-generation null,
physiological timing evidence, or population inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURVES = PROJECT_ROOT / "results" / "bspc_revision_v2" / "r006_merged" / "trained.npz"
DEFAULT_MERGE_META = DEFAULT_CURVES.with_name("trained.merge_meta.json")
DEFAULT_DETECTOR = PROJECT_ROOT / "results" / "bspc_revision_v2" / "r007_detector_report.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "bspc_revision_v2" / "r017_db2_fixed_replay"

SUBJECT_RE = re.compile(r"^S(\d+)_")
EXPECTED_PHASES = (0.0, 50.0, 100.0, 150.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--merge-meta", type=Path, default=DEFAULT_MERGE_META)
    parser.add_argument("--detector-report", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--origins", type=int, default=300)
    parser.add_argument("--smooth-sigma-bins", type=float, default=2.0)
    parser.add_argument("--lower-bound-ms", type=float, default=5.0)
    parser.add_argument("--short-lag-ms", type=float, default=30.0)
    parser.add_argument("--emd-min-ms", type=float, default=30.0)
    parser.add_argument("--emd-max-ms", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_curves(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    archive = np.load(path, allow_pickle=False)
    required = {
        "lags_ms", "curves", "ids", "phase_ms", "window_kind", "active_class",
        "predicted_class", "predicted_prob", "target_prob", "correct",
        "repetition_index", "segment_index",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError(f"{path} missing arrays: {missing}")
    lags = np.asarray(archive["lags_ms"], dtype=float)
    curves = np.asarray(archive["curves"], dtype=float)
    if curves.ndim != 2 or curves.shape[1] != lags.size:
        raise ValueError(f"curves shape {curves.shape} does not match lags {lags.shape}")
    arrays = {name: np.asarray(archive[name]) for name in required if name != "lags_ms"}
    n = curves.shape[0]
    for name, values in arrays.items():
        if values.shape[0] != n:
            raise ValueError(f"{name} length {values.shape[0]} != curves {n}")
    if not np.all(np.isfinite(curves)):
        raise ValueError("curves contain non-finite values")
    return lags, arrays | {"curves": curves}


def curve_subject(curve_id: str) -> int:
    match = SUBJECT_RE.match(curve_id)
    if not match:
        raise ValueError(f"cannot parse subject from curve id {curve_id!r}")
    return int(match.group(1))


def validate_current_replay(lags: np.ndarray, arrays: dict[str, np.ndarray], merge_meta: dict) -> dict:
    ids = [str(value) for value in arrays["ids"]]
    subjects = np.asarray([curve_subject(value) for value in ids], dtype=int)
    phases = np.asarray(arrays["phase_ms"], dtype=float)
    kinds = np.asarray(arrays["window_kind"]).astype(str)
    if set(subjects.tolist()) != set(range(1, 41)):
        raise AssertionError("current replay must contain exactly subjects S01-S40")
    if set(kinds.tolist()) != {"onset"}:
        raise AssertionError(f"R017 requires onset curves, got window kinds {sorted(set(kinds))}")
    if set(np.round(phases, 6).tolist()) != set(EXPECTED_PHASES):
        raise AssertionError(f"unexpected phase set: {sorted(set(phases.tolist()))}")
    per_subject = {subject: int(np.sum(subjects == subject)) for subject in range(1, 41)}
    if set(per_subject.values()) != {68}:
        raise AssertionError(f"expected 68 curves per subject, got {per_subject}")
    for subject in range(1, 41):
        subject_phases = phases[subjects == subject]
        counts = {phase: int(np.sum(np.isclose(subject_phases, phase))) for phase in EXPECTED_PHASES}
        if set(counts.values()) != {17}:
            raise AssertionError(f"S{subject:02d} phase counts are not 17 each: {counts}")
    if not np.allclose(lags, np.arange(0.5, 150.0 + 0.5, 0.5)):
        raise AssertionError("R017 expects the 0.5-ms to 150-ms lag grid")
    if merge_meta.get("n_records") != int(arrays["curves"].shape[0]):
        raise AssertionError("merge metadata record count does not match curves")
    first_meta = merge_meta.get("first_file_meta", {})
    required_meta = {
        "target_mode": "argmax",
        "records_source": "onset_protocol",
        "block_samples": 10,
        "max_lag_ms": 150.0,
    }
    for key, expected in required_meta.items():
        if first_meta.get(key) != expected:
            raise AssertionError(f"current curve metadata {key}={first_meta.get(key)!r}, expected {expected!r}")
    return {
        "n_curves": int(arrays["curves"].shape[0]),
        "n_subjects": 40,
        "curves_per_subject": per_subject,
        "phases_ms": list(EXPECTED_PHASES),
        "window_kind": "onset",
        "lag_bins": int(lags.size),
        "current_protocol": {
            "checkpoint_family": "R005b_balanced_full_seed20260815",
            "split": "4 train / 1 validation / 1 audit repetition per active class",
            "normalization": "training repetitions only",
            "window_samples": 600,
            "current_block_samples": 10,
            "audit_repetitions_per_subject": 17,
        },
    }


def prepare_peak(lags: np.ndarray, spectrum: np.ndarray, sigma: float, lower_bound: float) -> tuple[float, float, float]:
    positive = np.maximum(np.where(np.isfinite(spectrum), spectrum, 0.0), 0.0)
    smoothed = gaussian_filter1d(positive, sigma=sigma) if sigma > 0 else positive
    eligible = lags >= lower_bound
    if not np.any(eligible):
        return float("nan"), 0.0, 0.0
    values = smoothed[eligible]
    peak_index = int(np.argmax(values))
    peak_lag = float(lags[eligible][peak_index])
    return peak_lag, float(values[peak_index]), float(np.sum(values))


def poisson_binomial_upper_tail(probabilities: np.ndarray, observed: int) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    distribution = np.array([1.0], dtype=float)
    for probability in probabilities:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid Poisson-binomial probability {probability}")
        next_distribution = np.zeros(distribution.size + 1, dtype=float)
        next_distribution[:-1] += distribution * (1.0 - probability)
        next_distribution[1:] += distribution * probability
        distribution = next_distribution
    if observed <= 0:
        return 1.0
    if observed > len(probabilities):
        return 0.0
    return float(np.sum(distribution[observed:]))


def holm_adjust(p_values: dict[str, float], alpha: float) -> dict[str, dict[str, float | bool]]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = float(p_values[key])
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[key] = {
            "raw_p": value,
            "holm_adjusted_p": running,
            "reject_holm_alpha": bool(running <= alpha),
        }
    return adjusted


def protocol_rows(
    lags: np.ndarray,
    arrays: dict[str, np.ndarray],
    detector_decisions: dict[str, dict],
    sigma: float,
    lower_bound: float,
    short_lag: float,
    emd_min: float,
    emd_max: float,
) -> list[dict]:
    ids = np.asarray([str(value) for value in arrays["ids"]])
    subjects = np.asarray([curve_subject(value) for value in ids], dtype=int)
    phases = np.asarray(arrays["phase_ms"], dtype=float)
    rows = []
    for subject in range(1, 41):
        subject_mask = subjects == subject
        for protocol, protocol_mask in (
            ("reference_phase_0", subject_mask & np.isclose(phases, 0.0)),
            ("four_position", subject_mask),
        ):
            indices = np.flatnonzero(protocol_mask)
            spectrum = np.mean(np.asarray(arrays["curves"])[indices], axis=0)
            peak_lag, peak_height, positive_mass = prepare_peak(lags, spectrum, sigma, lower_bound)
            detector_entries = [detector_decisions.get(str(ids[index])) for index in indices]
            detector_entries = [entry for entry in detector_entries if entry is not None]
            n_supported = sum(bool(entry.get("supported")) for entry in detector_entries)
            rows.append(
                {
                    "subject": subject,
                    "protocol": protocol,
                    "n_windows": int(indices.size),
                    "n_supported_curves": int(n_supported),
                    "curve_support_rate": float(n_supported / len(detector_entries)) if detector_entries else None,
                    "peak_lag_ms": peak_lag,
                    "peak_height": peak_height,
                    "positive_mass": positive_mass,
                    "short_lag": bool(np.isfinite(peak_lag) and peak_lag < short_lag),
                    "emd_band": bool(np.isfinite(peak_lag) and emd_min <= peak_lag <= emd_max),
                    "exact_floor": bool(np.isfinite(peak_lag) and np.isclose(peak_lag, lower_bound)),
                    "mean_target_prob": float(np.mean(np.asarray(arrays["target_prob"])[indices])),
                    "mean_correct": float(np.mean(np.asarray(arrays["correct"])[indices])),
                }
            )
    return rows


def origin_diagnostic(
    lags: np.ndarray,
    arrays: dict[str, np.ndarray],
    sigma: float,
    lower_bound: float,
    short_lag: float,
    origins: int,
    alpha: float,
) -> tuple[list[dict], dict]:
    ids = np.asarray([str(value) for value in arrays["ids"]])
    subjects = np.asarray([curve_subject(value) for value in ids], dtype=int)
    phases = np.asarray(arrays["phase_ms"], dtype=float)
    if origins != lags.size:
        raise ValueError(f"origins={origins} must equal the saved spectrum bins={lags.size}")
    detail = []
    probabilities: dict[str, list[float]] = defaultdict(list)
    observed_counts: dict[str, int] = defaultdict(int)
    for subject in range(1, 41):
        subject_mask = subjects == subject
        for protocol, protocol_mask in (
            ("reference_phase_0", subject_mask & np.isclose(phases, 0.0)),
            ("four_position", subject_mask),
        ):
            spectrum = np.mean(np.asarray(arrays["curves"])[np.flatnonzero(protocol_mask)], axis=0)
            indicators = {"short_lag": [], "exact_floor": []}
            for origin in range(origins):
                rotated = np.roll(spectrum, origin)
                peak_lag, _height, _mass = prepare_peak(lags, rotated, sigma, lower_bound)
                indicators["short_lag"].append(bool(np.isfinite(peak_lag) and peak_lag < short_lag))
                indicators["exact_floor"].append(bool(np.isfinite(peak_lag) and np.isclose(peak_lag, lower_bound)))
            for statistic, values in indicators.items():
                key = f"{protocol}_{statistic}_count"
                probability = float(np.mean(values))
                probabilities[key].append(probability)
                observed = int(values[0])
                observed_counts[key] += observed
                detail.append(
                    {
                        "subject": subject,
                        "protocol": protocol,
                        "statistic": statistic,
                        "identity_origin": 0,
                        "identity_positive": bool(observed),
                        "origin_count": origins,
                        "origin_positive_count": int(np.sum(values)),
                        "origin_probability": probability,
                    }
                )
    raw_p = {
        key: poisson_binomial_upper_tail(np.asarray(probabilities[key]), observed_counts[key])
        for key in observed_counts
    }
    adjusted = holm_adjust(raw_p, alpha)
    tests = {}
    for key in sorted(raw_p):
        tests[key] = {
            "observed_count": observed_counts[key],
            "mean_origin_probability": float(np.mean(probabilities[key])),
            "raw_p": raw_p[key],
            **adjusted[key],
        }
    summary = {
        "status": "diagnostic_only",
        "headline_promotion_allowed": False,
        "method": "participant-mean spectra rotated across all saved lag origins; identity origin included",
        "not_a_null_statement": "conditional rotation of saved spectra, not signal-generation null, population inference, or physiological timing",
        "origins_per_subject_protocol": origins,
        "tests": tests,
    }
    return detail, summary


def load_detector(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions", [])
    return {str(entry["id"]): entry for entry in decisions}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists() and not args.overwrite and (args.output_dir / "summary.json").exists():
        print(f"skip existing {args.output_dir / 'summary.json'}; use --overwrite to recompute")
        return
    lags, arrays = load_curves(args.curves)
    merge_meta = json.loads(args.merge_meta.read_text(encoding="utf-8"))
    validation = validate_current_replay(lags, arrays, merge_meta)
    detector_decisions = load_detector(args.detector_report)
    subject_rows = protocol_rows(
        lags, arrays, detector_decisions, args.smooth_sigma_bins, args.lower_bound_ms,
        args.short_lag_ms, args.emd_min_ms, args.emd_max_ms,
    )
    origin_rows, origin_summary = origin_diagnostic(
        lags, arrays, args.smooth_sigma_bins, args.lower_bound_ms, args.short_lag_ms,
        args.origins, args.alpha,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "subject_protocol_summary.csv", subject_rows)
    write_csv(args.output_dir / "conditional_origin_detail.csv", origin_rows)
    json_dump(args.output_dir / "input_receipt.json", {
        "curves": str(args.curves),
        "curves_sha256": sha256_file(args.curves),
        "merge_meta": str(args.merge_meta),
        "merge_meta_sha256": sha256_file(args.merge_meta),
        "detector_report": str(args.detector_report),
        "detector_report_sha256": sha256_file(args.detector_report) if args.detector_report.exists() else None,
        "validation": validation,
    })
    summary = {
        "run_id": "R017",
        "status": "completed",
        "experiment": "current-protocol DB2 fixed-checkpoint replay",
        "historical_reference_boundary": {
            "historical_windows": 839,
            "historical_subjects": 40,
            "historical_short_lag_count": "26/40 (archived only)",
            "historical_median_peak_ms": "14.0 (archived only)",
            "statement": "historical values are not current R017 evidence",
        },
        "config": {
            "smooth_sigma_bins": args.smooth_sigma_bins,
            "lower_bound_ms": args.lower_bound_ms,
            "short_lag_ms": args.short_lag_ms,
            "emd_band_ms": [args.emd_min_ms, args.emd_max_ms],
            "origins": args.origins,
            "alpha": args.alpha,
            "detector_report_available": bool(detector_decisions),
        },
        "validation": validation,
        "protocol_summary": {
            protocol: {
                "n_subjects": 40,
                "n_windows": int(sum(row["n_windows"] for row in subject_rows if row["protocol"] == protocol)),
                "n_short_lag": int(sum(row["short_lag"] for row in subject_rows if row["protocol"] == protocol)),
                "n_emd_band": int(sum(row["emd_band"] for row in subject_rows if row["protocol"] == protocol)),
                "median_peak_lag_ms": float(np.median([row["peak_lag_ms"] for row in subject_rows if row["protocol"] == protocol])),
                "mean_curve_support_rate": float(np.mean([row["curve_support_rate"] for row in subject_rows if row["protocol"] == protocol and row["curve_support_rate"] is not None])) if detector_decisions else None,
            }
            for protocol in ("reference_phase_0", "four_position")
        },
        "conditional_spectral_origin": origin_summary,
    }
    json_dump(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["protocol_summary"], ensure_ascii=False, indent=2))
    print(f"R017 output -> {args.output_dir}")


if __name__ == "__main__":
    main()
