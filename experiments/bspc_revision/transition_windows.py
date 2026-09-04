"""R009：统一相位窗口协议（onset 转换、offset 反向转换、稳态运动窗）。

在已审计的 onset 协议之上扩展两类时间对照窗口：

- `offset`：运动到静息的反向转换。参考点为活动段最后一个采样点
  （`active_end - 1`），相位定义与 onset 镜像对称；
- `steady`：稳态运动窗口。参考点位于活动段内部的固定相对位置
  （默认中点），不接近任何转换边界。

三类窗口共用统一记录格式和因果输入约束，供 R008/R009 曲线采集使用。
所有窗口仍严格限制在重复段原始支持范围内，不跨训练、验证、审计分区。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bspc_revision.leakage_free_db2 import (  # noqa: E402
    TrialRecord,
    build_trial_records,
    write_csv,
)
from experiments.bspc_revision.onset_protocol import (  # noqa: E402
    build_onset_audit_records,
    milliseconds_to_samples,
)

WINDOW_KINDS = ("onset", "offset", "steady")


@dataclass(frozen=True)
class PhaseWindowRecord:
    subject: int
    split: str
    active_class: int
    repetition_index: int
    segment_index: int
    window_kind: str
    phase_ms: float
    phase_samples: int
    reference_sample: int
    current_endpoint_sample: int
    current_block_start_sample: int
    current_block_end_sample: int
    input_start_sample: int
    input_end_sample: int
    reference_in_window: int
    current_endpoint_in_window: int
    current_block_start_in_window: int
    current_block_end_in_window: int


def _make_record(
    trial: TrialRecord,
    window_kind: str,
    phase_ms: float,
    phase_samples: int,
    reference_sample: int,
    current_endpoint: int,
    window_samples: int,
    block_samples: int,
) -> PhaseWindowRecord:
    current_block_start = current_endpoint - block_samples + 1
    input_end = current_endpoint + 1
    input_start = input_end - window_samples
    record = PhaseWindowRecord(
        subject=trial.subject,
        split=trial.split,
        active_class=trial.active_class,
        repetition_index=trial.repetition_index,
        segment_index=trial.segment_index,
        window_kind=window_kind,
        phase_ms=phase_ms,
        phase_samples=phase_samples,
        reference_sample=reference_sample,
        current_endpoint_sample=current_endpoint,
        current_block_start_sample=current_block_start,
        current_block_end_sample=current_endpoint + 1,
        input_start_sample=input_start,
        input_end_sample=input_end,
        reference_in_window=reference_sample - input_start,
        current_endpoint_in_window=current_endpoint - input_start,
        current_block_start_in_window=current_block_start - input_start,
        current_block_end_in_window=current_endpoint + 1 - input_start,
    )
    verify_phase_window_record(record, trial, window_samples, block_samples)
    return record


def verify_phase_window_record(
    record: PhaseWindowRecord,
    trial: TrialRecord,
    window_samples: int,
    block_samples: int,
) -> None:
    if record.window_kind not in WINDOW_KINDS:
        raise AssertionError(f"unknown window kind: {record.window_kind}")
    if record.input_end_sample - record.input_start_sample != window_samples:
        raise AssertionError("causal input has the wrong length")
    if record.current_endpoint_in_window != window_samples - 1:
        raise AssertionError("causal input must end at the declared current endpoint")
    if record.current_block_end_sample - record.current_block_start_sample != block_samples:
        raise AssertionError("current block has the wrong width")
    if record.input_start_sample < trial.raw_start or record.input_end_sample > trial.raw_end:
        raise ValueError(
            f"S{trial.subject} class {trial.active_class} repetition "
            f"{trial.repetition_index} {record.window_kind} phase {record.phase_ms:g} ms: "
            f"window [{record.input_start_sample}, {record.input_end_sample}) exceeds "
            f"trial support [{trial.raw_start}, {trial.raw_end})"
        )
    expected_offset = record.current_endpoint_sample - record.reference_sample
    if record.window_kind in ("onset", "offset"):
        if expected_offset != record.phase_samples:
            raise AssertionError("current endpoint is not at the declared phase")
    elif expected_offset < 0:
        raise AssertionError("steady reference must not lie after the current endpoint")


def build_onset_phase_windows(
    trials: Iterable[TrialRecord],
    labels: np.ndarray,
    split: str,
    phases_ms: Sequence[float],
    fs: int = 2000,
    window_samples: int = 600,
    block_samples: int = 10,
) -> list[PhaseWindowRecord]:
    """静息到运动转换窗口；委托已审计的 onset 协议构造后转换格式。"""
    onset_records = build_onset_audit_records(
        trials, labels, split, phases_ms, fs, window_samples, block_samples
    )
    trial_by_segment = {trial.segment_index: trial for trial in trials if trial.split == split}
    converted = []
    for record in onset_records:
        trial = trial_by_segment[record.segment_index]
        converted.append(
            _make_record(
                trial,
                "onset",
                record.phase_ms,
                record.phase_samples,
                record.restimulus_onset_sample,
                record.current_endpoint_sample,
                window_samples,
                block_samples,
            )
        )
    return converted


def build_offset_phase_windows(
    trials: Iterable[TrialRecord],
    labels: np.ndarray,
    split: str,
    phases_ms: Sequence[float],
    fs: int = 2000,
    window_samples: int = 600,
    block_samples: int = 10,
) -> list[PhaseWindowRecord]:
    """运动到静息反向转换窗口；参考点为活动段最后一个采样点。"""
    phase_pairs = [(float(p), milliseconds_to_samples(float(p), fs)) for p in phases_ms]
    if any(phase_samples < 0 for _, phase_samples in phase_pairs):
        raise ValueError("offset phases must be nonnegative")
    records = []
    for trial in trials:
        if trial.split != split:
            continue
        last_active = trial.active_end - 1
        if labels[last_active] != trial.active_class or labels[trial.active_end] != 0:
            raise ValueError(
                f"S{trial.subject}: active_end={trial.active_end} is not an "
                f"{trial.active_class}->0 transition"
            )
        for phase_ms, phase_samples in phase_pairs:
            records.append(
                _make_record(
                    trial,
                    "offset",
                    phase_ms,
                    phase_samples,
                    last_active,
                    last_active + phase_samples,
                    window_samples,
                    block_samples,
                )
            )
    return records


def build_steady_phase_windows(
    trials: Iterable[TrialRecord],
    labels: np.ndarray,
    split: str,
    fractions: Sequence[float] = (0.5,),
    fs: int = 2000,
    window_samples: int = 600,
    block_samples: int = 10,
) -> list[PhaseWindowRecord]:
    """稳态运动窗口；参考点位于活动段 [active_start, active_end) 的相对位置。"""
    if any(not 0.0 < fraction < 1.0 for fraction in fractions):
        raise ValueError("steady fractions must be in (0, 1)")
    records = []
    for trial in trials:
        if trial.split != split:
            continue
        segment_length = trial.active_end - trial.active_start
        for fraction in fractions:
            reference = trial.active_start + int(round(fraction * (segment_length - 1)))
            if labels[reference] != trial.active_class:
                raise ValueError(
                    f"S{trial.subject}: steady reference {reference} is not inside "
                    f"class {trial.active_class} segment"
                )
            phase_samples = reference - trial.active_start
            records.append(
                _make_record(
                    trial,
                    "steady",
                    phase_samples * 1000.0 / fs,
                    phase_samples,
                    reference,
                    reference,
                    window_samples,
                    block_samples,
                )
            )
    return records


def build_all_phase_windows(
    trials: list[TrialRecord],
    labels: np.ndarray,
    split: str,
    phases_ms: Sequence[float],
    steady_fractions: Sequence[float],
    fs: int = 2000,
    window_samples: int = 600,
    block_samples: int = 10,
) -> list[PhaseWindowRecord]:
    records: list[PhaseWindowRecord] = []
    records.extend(
        build_onset_phase_windows(trials, labels, split, phases_ms, fs, window_samples, block_samples)
    )
    records.extend(
        build_offset_phase_windows(trials, labels, split, phases_ms, fs, window_samples, block_samples)
    )
    records.extend(
        build_steady_phase_windows(
            trials, labels, split, steady_fractions, fs, window_samples, block_samples
        )
    )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r009_window_manifest",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 41)))
    parser.add_argument("--phases-ms", nargs="+", type=float, default=[0, 50, 100, 150])
    parser.add_argument("--steady-fractions", nargs="+", type=float, default=[0.5])
    return parser.parse_args()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    coordinate_keys = set()
    all_records: list[PhaseWindowRecord] = []
    subject_summaries = []

    for subject in args.subjects:
        labels = np.load(args.data_root / f"S{subject}_label.npy", mmap_mode="r")
        trials = build_trial_records(labels, subject, split_seed=20260815, purge_samples=600)
        audit_trials = [trial for trial in trials if trial.split == "audit"]
        records = build_all_phase_windows(
            trials,
            labels,
            split="audit",
            phases_ms=args.phases_ms,
            steady_fractions=args.steady_fractions,
            fs=2000,
            window_samples=600,
            block_samples=10,
        )
        expected = len(audit_trials) * (2 * len(args.phases_ms) + len(args.steady_fractions))
        if len(records) != expected:
            raise AssertionError(
                f"S{subject}: expected {expected} phase windows, got {len(records)}"
            )

        kind_counts = Counter()
        for record in records:
            coordinate_key = (
                subject,
                record.segment_index,
                record.window_kind,
                record.phase_samples,
                record.input_start_sample,
            )
            if coordinate_key in coordinate_keys:
                raise AssertionError(f"S{subject}: duplicate phase window {coordinate_key}")
            coordinate_keys.add(coordinate_key)
            kind_counts[record.window_kind] += 1

        all_records.extend(records)
        subject_summaries.append(
            {
                "subject": subject,
                "audit_repetitions": len(audit_trials),
                "record_count": len(records),
                "kind_counts": dict(sorted(kind_counts.items())),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "phase_windows.csv",
        [field.name for field in fields(PhaseWindowRecord)],
        (asdict(record) for record in all_records),
    )
    payload = {
        "status": "PASS",
        "protocol_name": "R009 统一相位窗口协议（onset/offset/steady）",
        "scope_note": "参考点为 restimulus 修正标签边界或活动段内部位置，不是生理或机械起点",
        "subjects": args.subjects,
        "phases_ms": args.phases_ms,
        "steady_fractions": args.steady_fractions,
        "total_records": len(all_records),
        "records_per_subject": len(all_records) // max(1, len(args.subjects)),
        "kind_counts_total": dict(
            sorted(Counter(record.window_kind for record in all_records).items())
        ),
        "invariants": {
            "window_length_samples": 600,
            "current_endpoint_in_window": 599,
            "current_block_length_samples": 10,
            "no_trial_support_crossing": True,
            "unique_subject_segment_kind_phase_coordinates": True,
            "offset_reference_is_last_active_sample": True,
            "steady_reference_inside_active_segment": True,
        },
        "subject_summaries": subject_summaries,
    }
    write_json_atomic(args.output_dir / "window_manifest_integrity.json", payload)
    print(
        f"PASS subjects={len(args.subjects)} records={len(all_records)} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
