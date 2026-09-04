"""对 40 人真实标签执行 `restimulus` 零点协议数据级审计。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bspc_revision.leakage_free_db2 import NUM_CLASSES, build_trial_records
from experiments.bspc_revision.onset_protocol import build_onset_audit_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "onset_protocol_integrity.json")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 41)))
    parser.add_argument("--phases-ms", nargs="+", type=float, default=[0, 50, 100, 150])
    return parser.parse_args()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    subject_summaries = []
    coordinate_keys = set()
    split_repetition_keys = defaultdict(set)
    transition_counts = Counter()

    for subject in args.subjects:
        labels = np.load(args.data_root / f"S{subject}_label.npy", mmap_mode="r")
        invalid_labels = np.unique(labels[(labels < 0) | (labels >= args.num_classes)])
        if invalid_labels.size:
            raise ValueError(
                f"S{subject}: labels outside [0, {args.num_classes - 1}]: "
                f"{invalid_labels.tolist()}"
            )
        trials = build_trial_records(
            labels,
            subject,
            split_seed=20260815,
            purge_samples=600,
            num_classes=args.num_classes,
        )
        audit_trials = [trial for trial in trials if trial.split == "audit"]
        records = build_onset_audit_records(
            trials,
            labels,
            split="audit",
            phases_ms=args.phases_ms,
            fs=2000,
            window_samples=600,
            block_samples=10,
        )
        expected_audit_trials = args.num_classes - 1
        expected_records = expected_audit_trials * len(args.phases_ms)
        if len(audit_trials) != expected_audit_trials or len(records) != expected_records:
            raise AssertionError(
                f"S{subject}: expected {expected_audit_trials} audit repetitions and "
                f"{expected_records} records, "
                f"got {len(audit_trials)} and {len(records)}"
            )

        phase_counts = Counter()
        for record in records:
            if labels[record.restimulus_onset_sample - 1] != 0:
                raise AssertionError(f"S{subject}: onset predecessor is not rest")
            if labels[record.restimulus_onset_sample] != record.active_class:
                raise AssertionError(f"S{subject}: onset target label mismatch")
            if not (
                record.input_start_sample >= next(
                    trial.raw_start
                    for trial in audit_trials
                    if trial.segment_index == record.segment_index
                )
                and record.input_end_sample <= next(
                    trial.raw_end
                    for trial in audit_trials
                    if trial.segment_index == record.segment_index
                )
            ):
                raise AssertionError(f"S{subject}: onset window crosses trial support")
            coordinate_key = (
                subject,
                record.segment_index,
                record.phase_samples,
                record.input_start_sample,
                record.input_end_sample,
            )
            if coordinate_key in coordinate_keys:
                raise AssertionError(f"S{subject}: duplicate onset record {coordinate_key}")
            coordinate_keys.add(coordinate_key)
            split_repetition_keys[record.split].add(
                (subject, record.active_class, record.repetition_index)
            )
            phase_counts[str(record.phase_ms)] += 1
            transition_counts[f"{int(labels[record.restimulus_onset_sample - 1])}->{int(labels[record.restimulus_onset_sample])}"] += 1

        subject_summaries.append(
            {
                "subject": subject,
                "audit_repetitions": len(audit_trials),
                "record_count": len(records),
                "phase_counts": dict(sorted(phase_counts.items())),
                "active_classes": sorted({record.active_class for record in records}),
            }
        )

    records_per_subject = (args.num_classes - 1) * len(args.phases_ms)
    expected_total = len(args.subjects) * records_per_subject
    if len(coordinate_keys) != expected_total:
        raise AssertionError(f"expected {expected_total} unique records, got {len(coordinate_keys)}")
    payload = {
        "status": "PASS",
        "protocol_name": "NinaPro restimulus 修正标签起点对齐协议",
        "zero_point_scope": "restimulus active_start；不是 sEMG 生理起点或机械动作起点",
        "subjects": args.subjects,
        "num_classes": args.num_classes,
        "phases_ms": args.phases_ms,
        "records_per_subject": records_per_subject,
        "total_unique_records": len(coordinate_keys),
        "transition_counts": dict(sorted(transition_counts.items())),
        "split_repetition_counts": {
            split: len(keys) for split, keys in sorted(split_repetition_keys.items())
        },
        "invariants": {
            "exact_rest_to_active_transition": True,
            "window_length_samples": 600,
            "current_endpoint_in_window": 599,
            "current_block_length_samples": 10,
            "no_trial_support_crossing": True,
            "unique_subject_trial_phase_coordinates": True,
            "no_synthetic_first_window_onsets": True,
        },
        "subject_summaries": subject_summaries,
    }
    write_json_atomic(args.output, payload)
    print(
        f"PASS subjects={len(args.subjects)} records={len(coordinate_keys)} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
