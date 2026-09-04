import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bspc_revision.leakage_free_db2 import TrialRecord
from experiments.bspc_revision.transition_windows import (
    build_offset_phase_windows,
    build_onset_phase_windows,
    build_steady_phase_windows,
)

FS = 2000
WINDOW_SAMPLES = 600
BLOCK_SAMPLES = 10


def make_trial(
    raw_start: int,
    raw_end: int,
    active_start: int,
    active_end: int,
    split: str = "audit",
    active_class: int = 3,
    segment_index: int = 5,
) -> TrialRecord:
    return TrialRecord(
        subject=1,
        split=split,
        active_class=active_class,
        repetition_index=0,
        segment_index=segment_index,
        active_start=active_start,
        active_end=active_end,
        raw_start=raw_start,
        raw_end=raw_end,
    )


def make_labels(total: int, active_start: int, active_end: int, active_class: int = 3) -> np.ndarray:
    labels = np.zeros(total, dtype=np.int64)
    labels[active_start:active_end] = active_class
    return labels


# 活动段 [5000, 9000)，前后各留 5000/1000 静息；trial 支持覆盖全段
ACTIVE_START = 5000
ACTIVE_END = 9000
TOTAL = 10000
TRIAL = make_trial(0, TOTAL, ACTIVE_START, ACTIVE_END)
LABELS = make_labels(TOTAL, ACTIVE_START, ACTIVE_END)


class OnsetPhaseWindowTest(unittest.TestCase):
    def test_matches_audited_onset_coordinates(self) -> None:
        records = build_onset_phase_windows([TRIAL], LABELS, "audit", [0, 50], FS)
        self.assertEqual(len(records), 2)
        phase0 = records[0]
        self.assertEqual(phase0.window_kind, "onset")
        self.assertEqual(phase0.reference_sample, ACTIVE_START)
        self.assertEqual(phase0.current_endpoint_sample, ACTIVE_START)
        self.assertEqual(phase0.input_end_sample, ACTIVE_START + 1)
        self.assertEqual(phase0.input_end_sample - phase0.input_start_sample, WINDOW_SAMPLES)
        phase50 = records[1]
        self.assertEqual(phase50.current_endpoint_sample, ACTIVE_START + 100)
        self.assertEqual(phase50.reference_in_window, WINDOW_SAMPLES - 100 - 1)


class OffsetPhaseWindowTest(unittest.TestCase):
    def test_reference_is_last_active_sample(self) -> None:
        records = build_offset_phase_windows([TRIAL], LABELS, "audit", [0, 50], FS)
        self.assertEqual(len(records), 2)
        phase0 = records[0]
        self.assertEqual(phase0.window_kind, "offset")
        self.assertEqual(phase0.reference_sample, ACTIVE_END - 1)
        self.assertEqual(phase0.current_endpoint_sample, ACTIVE_END - 1)
        self.assertEqual(LABELS[phase0.reference_sample], 3)
        self.assertEqual(LABELS[phase0.reference_sample + 1], 0)
        phase50 = records[1]
        self.assertEqual(phase50.current_endpoint_sample, ACTIVE_END - 1 + 100)

    def test_invalid_transition_rejected(self) -> None:
        labels = LABELS.copy()
        labels[ACTIVE_END] = 7  # 破坏 active->0 转换
        with self.assertRaisesRegex(ValueError, "transition"):
            build_offset_phase_windows([TRIAL], labels, "audit", [0], FS)

    def test_window_crossing_support_rejected(self) -> None:
        trial = make_trial(0, ACTIVE_END + 50, ACTIVE_START, ACTIVE_END)
        with self.assertRaisesRegex(ValueError, "exceeds trial support"):
            build_offset_phase_windows([trial], LABELS, "audit", [50], FS)

    def test_split_filtering(self) -> None:
        train_trial = make_trial(0, TOTAL, ACTIVE_START, ACTIVE_END, split="train")
        records = build_offset_phase_windows([train_trial], LABELS, "audit", [0], FS)
        self.assertEqual(records, [])


class SteadyPhaseWindowTest(unittest.TestCase):
    def test_midpoint_reference(self) -> None:
        records = build_steady_phase_windows([TRIAL], LABELS, "audit", [0.5], FS)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.window_kind, "steady")
        expected = ACTIVE_START + int(round(0.5 * (ACTIVE_END - ACTIVE_START - 1)))
        self.assertEqual(record.reference_sample, expected)
        self.assertEqual(record.current_endpoint_sample, expected)
        self.assertEqual(LABELS[record.reference_sample], 3)
        self.assertEqual(record.input_end_sample - record.input_start_sample, WINDOW_SAMPLES)
        self.assertGreater(record.phase_samples, 0)

    def test_multiple_fractions(self) -> None:
        records = build_steady_phase_windows([TRIAL], LABELS, "audit", [0.25, 0.75], FS)
        self.assertEqual(len(records), 2)
        self.assertLess(records[0].reference_sample, records[1].reference_sample)

    def test_invalid_fraction_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_steady_phase_windows([TRIAL], LABELS, "audit", [1.0], FS)

    def test_reference_outside_segment_rejected(self) -> None:
        labels = LABELS.copy()
        labels[ACTIVE_START + 2000] = 0  # 在活动段内部制造空洞
        with self.assertRaisesRegex(ValueError, "not inside"):
            build_steady_phase_windows([TRIAL], labels, "audit", [0.5], FS)


if __name__ == "__main__":
    unittest.main()
