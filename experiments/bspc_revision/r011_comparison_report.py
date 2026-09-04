"""Create the prespecified R011 method-comparison and integrity reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def aopc_area(curve: np.ndarray) -> float:
    if curve.size < 2:
        return float("nan")
    return float(np.trapezoid(curve, dx=1.0) / (curve.size - 1))


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_artifact_name(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"S(\d{2})_(occlusion|timeshap)\.npz", path.name)
    if match is None:
        raise ValueError(f"unexpected R011 artifact name: {path.name}")
    return int(match.group(1)), match.group(2)


def load_records(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"attribution", "aopc_curve", "correct", "phase_ms", "window_kind", "meta_json"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        arrays = {name: np.asarray(payload[name]) for name in required - {"meta_json"}}
        meta = json.loads(str(payload["meta_json"]))
    n_records = arrays["attribution"].shape[0]
    if any(array.shape[0] != n_records for array in arrays.values()):
        raise ValueError(f"{path}: inconsistent record counts")
    if meta.get("status") != "ok":
        raise ValueError(f"{path}: method status is {meta.get('status')}")
    if not np.isfinite(arrays["attribution"]).all() or not np.isfinite(arrays["aopc_curve"]).all():
        raise ValueError(f"{path}: non-finite attribution or AOPC values")
    return arrays, meta


def aggregate(records: list[dict[str, object]], method: str, correctness: str, phase_ms: str) -> dict[str, object]:
    selected = [
        row
        for row in records
        if row["method"] == method
        and (correctness == "all" or row["correctness"] == correctness)
        and (phase_ms == "all" or row["phase_ms"] == phase_ms)
    ]
    values = np.asarray([row["aopc"] for row in selected], dtype=np.float64)
    return {
        "method": method,
        "correctness": correctness,
        "phase_ms": phase_ms,
        "n_records": len(selected),
        "n_subjects": len({row["subject"] for row in selected}),
        "aopc_mean": float(values.mean()) if values.size else None,
        "aopc_median": float(np.median(values)) if values.size else None,
        "aopc_std": float(values.std()) if values.size else None,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("S*_*.*"))
    expected = {(subject, method) for subject in range(1, 41) for method in ("occlusion", "timeshap")}
    found: set[tuple[int, str]] = set()
    raw_records: list[dict[str, object]] = []
    integrity: list[dict[str, object]] = []

    for path in paths:
        if path.suffix != ".npz":
            continue
        subject, method = parse_artifact_name(path)
        if (subject, method) in found:
            raise ValueError(f"duplicate artifact for S{subject:02d} {method}")
        found.add((subject, method))
        arrays, meta = load_records(path)
        aopc_values = np.asarray([aopc_area(row) for row in arrays["aopc_curve"]], dtype=np.float64)
        if not np.isfinite(aopc_values).all():
            raise ValueError(f"{path}: invalid AOPC value")
        integrity.append(
            {
                "subject": subject,
                "method": method,
                "n_records": int(aopc_values.size),
                "n_groups": int(arrays["attribution"].shape[1]),
                "n_aopc_steps": int(arrays["aopc_curve"].shape[1]),
                "finite": True,
                "runtime_seconds_per_window": meta.get("runtime_seconds_per_window"),
            }
        )
        for index, aopc in enumerate(aopc_values.tolist()):
            raw_records.append(
                {
                    "subject": subject,
                    "method": method,
                    "correctness": "correct" if int(arrays["correct"][index]) else "error",
                    "phase_ms": f"{float(arrays['phase_ms'][index]):g}",
                    "window_kind": str(arrays["window_kind"][index]),
                    "aopc": aopc,
                }
            )

    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        raise ValueError(f"artifact completeness failure: missing={missing}, extra={extra}")

    phases = sorted({str(row["phase_ms"]) for row in raw_records}, key=float)
    aggregate_rows = [
        aggregate(raw_records, method, correctness, phase)
        for method in ("occlusion", "timeshap")
        for correctness in ("all", "correct", "error")
        for phase in ("all", *phases)
    ]
    write_csv(args.output_dir / "r011_aopc_stratified.csv", aggregate_rows)
    write_csv(args.output_dir / "r011_artifact_integrity.csv", integrity)
    write_json_atomic(
        args.output_dir / "r011_report.json",
        {
            "status": "completed",
            "protocol": "R011 full method comparison with correctness and phase strata",
            "input_dir": str(args.input_dir),
            "expected_artifacts": len(expected),
            "verified_artifacts": len(found),
            "total_records": len(raw_records),
            "phases_ms": phases,
            "strata": aggregate_rows,
            "integrity": {
                "all_finite": True,
                "records_per_artifact": sorted({int(row["n_records"]) for row in integrity}),
                "groups_per_method": {
                    method: sorted({int(row["n_groups"]) for row in integrity if row["method"] == method})
                    for method in ("occlusion", "timeshap")
                },
            },
        },
    )
    print(f"completed output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
