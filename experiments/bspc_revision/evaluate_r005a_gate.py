"""Evaluate the R005a multi-seed diagnostic gate before a full balanced rerun."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAGNOSTIC_ROOT = (
    PROJECT_ROOT / "results" / "bspc_revision_v2" / "r005a_balanced_diagnostic"
)
DEFAULT_BASELINE_ROOT = (
    PROJECT_ROOT / "results" / "bspc_revision_v2" / "r005_full_s1_s40" / "subjects"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-root", type=Path, default=DEFAULT_DIAGNOSTIC_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260815, 20260816, 20260817])
    parser.add_argument("--problem-subjects", nargs="+", type=int, default=[10, 15, 17, 27])
    parser.add_argument("--control-subjects", nargs="+", type=int, default=[1, 40])
    parser.add_argument("--problem-median-f1", type=float, default=0.45)
    parser.add_argument("--collapse-f1", type=float, default=0.20)
    parser.add_argument("--control-f1-tolerance", type=float, default=0.05)
    parser.add_argument("--control-balanced-tolerance", type=float, default=0.05)
    parser.add_argument("--max-subject-f1-range", type=float, default=0.20)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluate_gate(args: argparse.Namespace) -> dict[str, object]:
    expected_subjects = sorted(set(args.problem_subjects + args.control_subjects))
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    baseline: dict[int, dict[str, float]] = {}

    for subject in args.control_subjects:
        path = args.baseline_root / f"S{subject:02d}_result.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        metrics = read_json(path)["audit_metrics"]
        baseline[subject] = {
            "macro_f1": float(metrics["macro_f1"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
        }

    for seed in args.seeds:
        subject_root = args.diagnostic_root / f"seed_{seed}" / "subjects"
        for subject in expected_subjects:
            path = subject_root / f"S{subject:02d}_result.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            result = read_json(path)
            metrics = result["audit_metrics"]
            rows.append(
                {
                    "seed": seed,
                    "subject": subject,
                    "role": "problem" if subject in args.problem_subjects else "control",
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "best_epoch": int(result["best_epoch"]),
                    "selection_metric": result.get("selection_metric"),
                    "subject_training_seed": result.get("subject_training_seed"),
                }
            )

    by_subject: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_subject[int(row["subject"])].append(row)

    subject_summary: dict[str, dict[str, object]] = {}
    conditions: list[dict[str, object]] = []
    for subject in expected_subjects:
        subject_rows = by_subject.get(subject, [])
        f1_values = [float(row["macro_f1"]) for row in subject_rows]
        balanced_values = [float(row["balanced_accuracy"]) for row in subject_rows]
        accuracy_values = [float(row["accuracy"]) for row in subject_rows]
        summary = {
            "role": "problem" if subject in args.problem_subjects else "control",
            "n_seeds": len(subject_rows),
            "accuracy_median": statistics.median(accuracy_values) if accuracy_values else None,
            "macro_f1_median": statistics.median(f1_values) if f1_values else None,
            "macro_f1_min": min(f1_values) if f1_values else None,
            "macro_f1_max": max(f1_values) if f1_values else None,
            "macro_f1_range": max(f1_values) - min(f1_values) if f1_values else None,
            "balanced_accuracy_median": (
                statistics.median(balanced_values) if balanced_values else None
            ),
        }
        subject_summary[str(subject)] = summary

        if subject in args.problem_subjects:
            conditions.append(
                {
                    "name": f"S{subject:02d}_problem_median_macro_f1",
                    "passed": bool(
                        f1_values and statistics.median(f1_values) >= args.problem_median_f1
                    ),
                    "value": summary["macro_f1_median"],
                    "threshold": args.problem_median_f1,
                }
            )
        elif subject in baseline:
            conditions.extend(
                [
                    {
                        "name": f"S{subject:02d}_control_macro_f1_non_degradation",
                        "passed": bool(
                            f1_values
                            and statistics.median(f1_values)
                            >= baseline[subject]["macro_f1"] - args.control_f1_tolerance
                        ),
                        "value": summary["macro_f1_median"],
                        "threshold": baseline[subject]["macro_f1"]
                        - args.control_f1_tolerance,
                    },
                    {
                        "name": f"S{subject:02d}_control_balanced_non_degradation",
                        "passed": bool(
                            balanced_values
                            and statistics.median(balanced_values)
                            >= baseline[subject]["balanced_accuracy"]
                            - args.control_balanced_tolerance
                        ),
                        "value": summary["balanced_accuracy_median"],
                        "threshold": baseline[subject]["balanced_accuracy"]
                        - args.control_balanced_tolerance,
                    },
                ]
            )

        conditions.append(
            {
                "name": f"S{subject:02d}_seed_stability",
                "passed": bool(
                    len(f1_values) == len(args.seeds)
                    and max(f1_values) - min(f1_values) <= args.max_subject_f1_range
                ),
                "value": summary["macro_f1_range"],
                "threshold": args.max_subject_f1_range,
            }
        )

    collapsed_runs = [
        row
        for row in rows
        if int(row["subject"]) in args.problem_subjects
        and float(row["macro_f1"]) < args.collapse_f1
    ]
    conditions.extend(
        [
            {
                "name": "all_expected_results_present",
                "passed": not missing and len(rows) == len(args.seeds) * len(expected_subjects),
                "value": len(rows),
                "threshold": len(args.seeds) * len(expected_subjects),
            },
            {
                "name": "no_problem_subject_collapse",
                "passed": len(collapsed_runs) == 0,
                "value": len(collapsed_runs),
                "threshold": 0,
            },
        ]
    )

    gate_passed = all(bool(condition["passed"]) for condition in conditions)
    return {
        "state": "completed",
        "gate_passed": gate_passed,
        "launch_full_run": gate_passed,
        "seeds": args.seeds,
        "problem_subjects": args.problem_subjects,
        "control_subjects": args.control_subjects,
        "thresholds": {
            "problem_median_f1": args.problem_median_f1,
            "collapse_f1": args.collapse_f1,
            "control_f1_tolerance": args.control_f1_tolerance,
            "control_balanced_tolerance": args.control_balanced_tolerance,
            "max_subject_f1_range": args.max_subject_f1_range,
        },
        "baseline_controls": baseline,
        "missing": missing,
        "collapsed_runs": collapsed_runs,
        "conditions": conditions,
        "subject_summary": subject_summary,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    summary = evaluate_gate(args)

    # 手动暂停标记：若存在，门控照常计算并报告，但不触发 R005b 全量自动启动。
    workspace = args.diagnostic_root.resolve().parents[2]
    hold_marker = workspace / "tmp" / "hold_full_run.flag"
    if hold_marker.is_file():
        summary["manual_hold"] = True
        summary["hold_marker"] = str(hold_marker)
        summary["hold_note"] = (
            "用户在门控后要求暂停，不自动启动 R005b；"
            "删除该标记后重新运行门控或监督器即可恢复自动启动"
        )
        summary["launch_full_run"] = False
    else:
        summary["manual_hold"] = False

    output_path = args.diagnostic_root / "gate_summary.json"
    write_json_atomic(output_path, summary)

    csv_path = args.diagnostic_root / "diagnostic_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "seed",
            "subject",
            "role",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "best_epoch",
            "selection_metric",
            "subject_training_seed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["rows"])

    print(
        f"R005a gate_passed={summary['gate_passed']} "
        f"manual_hold={summary['manual_hold']} "
        f"rows={len(summary['rows'])} output={output_path}",
        flush=True,
    )
    raise SystemExit(0 if summary["gate_passed"] else 2)


if __name__ == "__main__":
    main()
