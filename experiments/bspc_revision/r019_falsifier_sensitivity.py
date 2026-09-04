"""R019: current-protocol permutation, detector sensitivity, and stratification.

This script consumes already collected current-protocol curves.  It keeps the
participant as the aggregation unit for peak-count summaries and does not
reinterpret historical manuscript counts as new evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINED = PROJECT_ROOT / "results" / "bspc_revision_v2" / "r006_merged" / "trained.npz"
DEFAULT_PERMUTATION_DIR = PROJECT_ROOT / "results" / "bspc_revision_v2" / "r006_curves"
DEFAULT_DETECTOR = PROJECT_ROOT / "results" / "bspc_revision_v2" / "r007_detector_report.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "bspc_revision_v2" / "r019_falsifier_sensitivity"
SUBJECT_RE = re.compile(r"^S(\d+)_")
REQUIRED_PERMUTATION_SEEDS = (101, 202, 303)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-curves", type=Path, default=DEFAULT_TRAINED)
    parser.add_argument(
        "--permutation-dir", type=Path, action="append", default=None,
        help="目录或单个 npz；目录内读取 S*_param_scramble.npz，可重复提供",
    )
    parser.add_argument(
        "--permutation-seed", type=int, action="append", default=None,
        help="与 --permutation-dir 一一对应；默认现有 R006 null_seed 20260816",
    )
    parser.add_argument("--detector-report", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lower-bounds-ms", nargs="+", type=float, default=[5.0, 10.0, 15.0, 20.0])
    parser.add_argument("--smooth-sigmas", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0])
    parser.add_argument("--short-lag-ms", type=float, default=30.0)
    parser.add_argument("--floor-tolerance-ms", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_subject(curve_id: str) -> int:
    match = SUBJECT_RE.match(curve_id)
    if not match:
        raise ValueError(f"cannot parse subject from {curve_id!r}")
    return int(match.group(1))


def load_npz(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    archive = np.load(path, allow_pickle=False)
    required = {"lags_ms", "curves", "ids", "phase_ms", "window_kind", "correct", "target_prob", "active_class", "predicted_class"}
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError(f"{path} missing arrays: {missing}")
    lags = np.asarray(archive["lags_ms"], dtype=float)
    arrays = {name: np.asarray(archive[name]) for name in required if name != "lags_ms"}
    curves = np.asarray(arrays["curves"], dtype=float)
    if curves.ndim != 2 or curves.shape[1] != lags.size:
        raise ValueError(f"{path}: curves shape {curves.shape} does not match lags {lags.shape}")
    n = curves.shape[0]
    if any(values.shape[0] != n for values in arrays.values()):
        raise ValueError(f"{path}: metadata length mismatch")
    return lags, arrays


def concat_npz(paths: list[Path]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not paths:
        raise ValueError("no npz files found")
    lags, first = load_npz(paths[0])
    chunks = defaultdict(list)
    for name, values in first.items():
        chunks[name].append(values)
    for path in paths[1:]:
        other_lags, arrays = load_npz(path)
        if not np.array_equal(lags, other_lags):
            raise ValueError(f"lag grid mismatch: {paths[0]} vs {path}")
        for name, values in arrays.items():
            chunks[name].append(values)
    return lags, {name: np.concatenate(values) for name, values in chunks.items()}


def resolve_curve_source(source: Path, pattern: str) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(source.glob(pattern))


def participant_groups(arrays: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    subjects = np.asarray([parse_subject(str(value)) for value in arrays["ids"]], dtype=int)
    groups = {}
    for subject in range(1, 41):
        indices = np.flatnonzero(subjects == subject)
        if indices.size == 0:
            raise ValueError(f"missing subject S{subject:02d}")
        groups[subject] = indices
    return groups


def positive_smoothed(curve: np.ndarray, sigma: float) -> np.ndarray:
    values = np.maximum(np.where(np.isfinite(curve), curve, 0.0), 0.0)
    return gaussian_filter1d(values, sigma=sigma) if sigma > 0 else values


def peak_for_curve(lags: np.ndarray, curve: np.ndarray, lower_bound: float, sigma: float) -> tuple[float, float, float]:
    prepared = positive_smoothed(curve, sigma)
    mask = lags >= lower_bound
    if not np.any(mask):
        return float("nan"), 0.0, 0.0
    values = prepared[mask]
    index = int(np.argmax(values))
    return float(lags[mask][index]), float(values[index]), float(np.sum(values))


def participant_peak_rows(
    lags: np.ndarray,
    arrays: dict[str, np.ndarray],
    lower_bounds: list[float],
    sigmas: list[float],
    short_lag: float,
    floor_tolerance: float,
    variant: str,
    seed: int | None,
) -> list[dict]:
    groups = participant_groups(arrays)
    rows = []
    for lower in lower_bounds:
        for sigma in sigmas:
            for subject, indices in groups.items():
                spectrum = np.mean(np.asarray(arrays["curves"])[indices], axis=0)
                peak, height, mass = peak_for_curve(lags, spectrum, lower, sigma)
                rows.append({
                    "variant": variant,
                    "seed": seed,
                    "lower_bound_ms": lower,
                    "smooth_sigma_bins": sigma,
                    "subject": subject,
                    "n_windows": int(indices.size),
                    "peak_lag_ms": peak,
                    "peak_height": height,
                    "positive_mass": mass,
                    "short_lag": bool(np.isfinite(peak) and peak < short_lag),
                    "exact_floor": bool(np.isfinite(peak) and abs(peak - lower) <= floor_tolerance),
                    "mean_target_prob": float(np.mean(np.asarray(arrays["target_prob"])[indices])),
                    "mean_correct": float(np.mean(np.asarray(arrays["correct"])[indices])),
                })
    return rows


def summarize_peak_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["seed"], row["lower_bound_ms"], row["smooth_sigma_bins"])].append(row)
    summary = []
    for (variant, seed, lower, sigma), entries in sorted(grouped.items(), key=str):
        peaks = np.asarray([entry["peak_lag_ms"] for entry in entries], dtype=float)
        masses = np.asarray([entry["positive_mass"] for entry in entries], dtype=float)
        summary.append({
            "variant": variant,
            "seed": seed,
            "lower_bound_ms": lower,
            "smooth_sigma_bins": sigma,
            "n_subjects": len(entries),
            "short_lag_count": int(sum(entry["short_lag"] for entry in entries)),
            "exact_floor_count": int(sum(entry["exact_floor"] for entry in entries)),
            "median_peak_lag_ms": float(np.nanmedian(peaks)),
            "median_positive_mass": float(np.median(masses)),
        })
    return summary


def load_decisions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(entry["id"]): entry for entry in payload.get("decisions", [])}


def stratified_summary(arrays: dict[str, np.ndarray], decisions: dict[str, dict]) -> dict:
    ids = [str(value) for value in arrays["ids"]]
    if not decisions:
        return {"status": "missing_detector_report"}
    quartiles = np.quantile(np.asarray(arrays["target_prob"], dtype=float), [0.25, 0.5, 0.75])
    rows = []
    for index, curve_id in enumerate(ids):
        decision = decisions.get(curve_id)
        if decision is None:
            raise ValueError(f"detector report missing {curve_id}")
        if arrays["window_kind"][index] != "onset":
            continue
        value = float(arrays["target_prob"][index])
        quartile = f"Q{int(np.digitize(value, quartiles, right=True)) + 1}"
        rows.append({
            "id": curve_id,
            "phase": f"{float(arrays['phase_ms'][index]):g}",
            "active_class": str(int(arrays["active_class"][index])),
            "predicted_class": str(int(arrays["predicted_class"][index])),
            "correct": str(int(arrays["correct"][index])),
            "target_prob_quartile": quartile,
            "supported": bool(decision["supported"]),
            "positive_mass": float(decision["statistics"]["positive_mass"]),
            "peak_height": float(decision["statistics"]["peak_height"]),
        })
    dimensions = ("phase", "active_class", "predicted_class", "correct", "target_prob_quartile")
    output = {}
    for dimension in dimensions:
        levels = defaultdict(list)
        for row in rows:
            levels[row[dimension]].append(row)
        output[dimension] = {
            level: {
                "n": len(entries),
                "n_supported": int(sum(entry["supported"] for entry in entries)),
                "support_rate": float(np.mean([entry["supported"] for entry in entries])),
                "positive_mass_median": float(np.median([entry["positive_mass"] for entry in entries])),
                "peak_height_median": float(np.median([entry["peak_height"] for entry in entries])),
            }
            for level, entries in sorted(levels.items())
        }
    return {"status": "completed", "n_curves": len(rows), "dimensions": output}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists() and not args.overwrite and (args.output_dir / "summary.json").exists():
        print(f"skip existing {args.output_dir / 'summary.json'}; use --overwrite to recompute")
        return
    permutation_dirs = args.permutation_dir or [DEFAULT_PERMUTATION_DIR]
    permutation_seeds = args.permutation_seed or [20260816]
    if len(permutation_dirs) != len(permutation_seeds):
        raise ValueError("--permutation-dir and --permutation-seed must have equal lengths")

    trained_lags, trained = load_npz(args.trained_curves)
    trained_rows = participant_peak_rows(
        trained_lags, trained, args.lower_bounds_ms, args.smooth_sigmas,
        args.short_lag_ms, args.floor_tolerance_ms, "trained", None,
    )
    all_rows = list(trained_rows)
    source_receipt = [{"variant": "trained", "path": str(args.trained_curves), "sha256": sha256_file(args.trained_curves)}]
    permutation_sources = []
    for directory, seed in zip(permutation_dirs, permutation_seeds):
        paths = resolve_curve_source(directory, "S*_param_scramble.npz")
        if not paths:
            permutation_sources.append({"seed": seed, "source": str(directory), "status": "missing"})
            continue
        lags, arrays = concat_npz(paths)
        if not np.array_equal(trained_lags, lags):
            raise ValueError(f"permutation lag grid mismatch for {directory}")
        permutation_rows = participant_peak_rows(
            lags, arrays, args.lower_bounds_ms, args.smooth_sigmas,
            args.short_lag_ms, args.floor_tolerance_ms, "param_scramble", seed,
        )
        all_rows.extend(permutation_rows)
        permutation_sources.append({
            "seed": seed,
            "source": str(directory),
            "status": "completed",
            "n_files": len(paths),
            "n_curves": int(arrays["curves"].shape[0]),
            "files": [str(path) for path in paths],
        })

    detector_decisions = load_decisions(args.detector_report)
    stratified = stratified_summary(trained, detector_decisions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "participant_peak_rows.csv", all_rows)
    write_csv(args.output_dir / "participant_peak_summary.csv", summarize_peak_rows(all_rows))
    available_seeds = sorted(
        int(source["seed"])
        for source in permutation_sources
        if source.get("status") == "completed"
    )
    missing_required_seeds = [
        seed for seed in REQUIRED_PERMUTATION_SEEDS if seed not in available_seeds
    ]
    summary = {
        "run_id": "R019",
        "status": "completed" if not missing_required_seeds else "completed_with_available_permutations",
        "claim_boundary": "participant-cluster descriptive falsifier and detector sensitivity; no historical count is reused as current evidence",
        "trained_curve_source": str(args.trained_curves),
        "detector_report": str(args.detector_report),
        "config": {
            "lower_bounds_ms": args.lower_bounds_ms,
            "smooth_sigmas": args.smooth_sigmas,
            "short_lag_ms": args.short_lag_ms,
            "floor_tolerance_ms": args.floor_tolerance_ms,
        },
        "permutation_sources": permutation_sources,
        "available_permutation_seeds": available_seeds,
        "required_permutation_seeds": list(REQUIRED_PERMUTATION_SEEDS),
        "missing_required_seeds": missing_required_seeds,
        "detector_sensitivity": summarize_peak_rows(trained_rows),
        "stratified_detector": stratified,
        "historical_boundary": {
            "old_short_lag_count": "40/40 (archived manuscript falsifier only)",
            "old_detector_sensitivity": "26/40 at 5 ms lower bound (archived manuscript only)",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"permutation_sources": permutation_sources, "missing_required_seeds": summary["missing_required_seeds"]}, ensure_ascii=False, indent=2))
    print(f"R019 output -> {args.output_dir}")


if __name__ == "__main__":
    main()
