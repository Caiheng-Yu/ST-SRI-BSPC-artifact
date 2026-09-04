import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bspc_revision.leakage_free_db2 import (
    ACTIVE_CLASSES,
    build_trial_records,
    build_window_records,
    compute_class_weights,
    portable_project_path,
    records_by_split,
    resolve_project_path,
    selection_score,
    subject_training_seed,
    verify_split_isolation,
)
from experiments.bspc_revision.onset_protocol import build_onset_audit_records


def synthetic_labels() -> np.ndarray:
    parts = [np.zeros(1200, dtype=np.int64)]
    for label in ACTIVE_CLASSES:
        for repetition in range(6):
            parts.append(np.full(1400, label, dtype=np.int64))
            final_recording_segment = label == ACTIVE_CLASSES[-1] and repetition == 5
            rest_length = 30 if final_recording_segment else 1600
            parts.append(np.zeros(rest_length, dtype=np.int64))
    return np.concatenate(parts)


class LeakageFreeProtocolTest(unittest.TestCase):
    def test_project_paths_are_portable(self) -> None:
        relative_path = Path("checkpoints_bspc_v2") / "test" / "S01_best.pth"
        self.assertEqual(resolve_project_path(relative_path), PROJECT_ROOT / relative_path)
        self.assertEqual(portable_project_path(PROJECT_ROOT / relative_path), relative_path.as_posix())

    def test_balanced_training_helpers(self) -> None:
        labels = synthetic_labels()
        trials = build_trial_records(labels, subject=1, split_seed=20260815, purge_samples=200)
        records = build_window_records(labels, trials, window_len=600, stride=100)
        train_records = [record for record in records if record.split == "train"]

        counts, weights = compute_class_weights(train_records, power=0.5)
        self.assertEqual(counts.shape, (18,))
        self.assertEqual(weights.shape, (18,))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLess(weights[0], weights[int(np.argmin(counts[1:])) + 1])
        self.assertEqual(subject_training_seed(20260815, 10), 20270905)
        self.assertEqual(selection_score({"macro_f1": 0.625}, "macro_f1"), 0.625)

    def test_repetition_assignment_and_window_isolation(self) -> None:
        labels = synthetic_labels()
        trials = build_trial_records(labels, subject=1, split_seed=20260815, purge_samples=200)
        windows = build_window_records(labels, trials, window_len=600, stride=100)
        grouped = records_by_split(windows)

        for label in ACTIVE_CLASSES:
            label_trials = [trial for trial in trials if trial.active_class == label]
            self.assertEqual(sum(trial.split == "train" for trial in label_trials), 4)
            self.assertEqual(sum(trial.split == "validation" for trial in label_trials), 1)
            self.assertEqual(sum(trial.split == "audit" for trial in label_trials), 1)

        overlaps = verify_split_isolation(grouped, len(labels))
        self.assertEqual(overlaps, {
            "train_validation": 0,
            "train_audit": 0,
            "validation_audit": 0,
        })
        for split_records in grouped.values():
            present = {record.modal_label for record in split_records}
            self.assertEqual(present, set(range(18)))

    def test_onset_records_use_exact_restimulus_transition(self) -> None:
        labels = synthetic_labels()
        trials = build_trial_records(labels, subject=1, split_seed=20260815, purge_samples=200)
        records = build_onset_audit_records(
            trials,
            labels,
            split="audit",
            phases_ms=[0, 50, 100, 150],
            fs=2000,
            window_samples=600,
            block_samples=10,
        )

        self.assertEqual(len(records), len(ACTIVE_CLASSES) * 4)
        for record in records:
            self.assertEqual(labels[record.restimulus_onset_sample - 1], 0)
            self.assertEqual(labels[record.restimulus_onset_sample], record.active_class)
            self.assertEqual(record.input_end_sample - record.input_start_sample, 600)
            self.assertEqual(record.current_endpoint_in_window, 599)
            self.assertEqual(record.current_block_end_in_window, 600)
            self.assertEqual(record.current_block_end_in_window - record.current_block_start_in_window, 10)
            self.assertEqual(
                record.current_endpoint_sample - record.restimulus_onset_sample,
                record.phase_samples,
            )
            self.assertEqual(record.restimulus_onset_in_window, 599 - record.phase_samples)

    def test_onset_record_rejects_nontransition_active_start(self) -> None:
        labels = synthetic_labels()
        trials = build_trial_records(labels, subject=1, split_seed=20260815, purge_samples=200)
        trial = next(record for record in trials if record.split == "audit")
        labels[trial.active_start - 1] = trial.active_class

        with self.assertRaisesRegex(ValueError, "is not a restimulus"):
            build_onset_audit_records(
                [trial],
                labels,
                split="audit",
                phases_ms=[0],
                fs=2000,
                window_samples=600,
                block_samples=10,
            )


if __name__ == "__main__":
    unittest.main()
