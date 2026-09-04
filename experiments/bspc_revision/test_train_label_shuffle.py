import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bspc_revision.leakage_free_db2 import WindowRecord
from experiments.bspc_revision.train_label_shuffle import shuffle_train_window_labels


def make_grouped(train_labels: list[int]) -> dict[str, list[WindowRecord]]:
    def record(index: int, split: str, label: int) -> WindowRecord:
        return WindowRecord(
            subject=1,
            split=split,
            active_class=label if label else 1,
            repetition_index=index,
            segment_index=index,
            raw_start=index * 100,
            raw_end=index * 100 + 600,
            modal_label=label,
        )

    return {
        "train": [record(i, "train", label) for i, label in enumerate(train_labels)],
        "validation": [record(100, "validation", 2)],
        "audit": [record(200, "audit", 3)],
    }


class ShuffleTrainLabelsTest(unittest.TestCase):
    def test_preserves_class_histogram(self) -> None:
        grouped = make_grouped([0, 0, 0, 1, 1, 2, 2, 3])
        shuffled = shuffle_train_window_labels(grouped, seed=5)
        original_counts = Counter(record.modal_label for record in grouped["train"])
        shuffled_counts = Counter(record.modal_label for record in shuffled["train"])
        self.assertEqual(original_counts, shuffled_counts)

    def test_actually_permuted(self) -> None:
        grouped = make_grouped([0, 1, 2, 3, 4, 5, 6, 7])
        shuffled = shuffle_train_window_labels(grouped, seed=5)
        original = [record.modal_label for record in grouped["train"]]
        permuted = [record.modal_label for record in shuffled["train"]]
        self.assertNotEqual(original, permuted)

    def test_deterministic_per_seed(self) -> None:
        grouped = make_grouped([0, 1, 2, 3, 4, 5, 6, 7])
        first = [r.modal_label for r in shuffle_train_window_labels(grouped, seed=5)["train"]]
        second = [r.modal_label for r in shuffle_train_window_labels(grouped, seed=5)["train"]]
        third = [r.modal_label for r in shuffle_train_window_labels(grouped, seed=6)["train"]]
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_validation_and_audit_untouched(self) -> None:
        grouped = make_grouped([0, 1, 2, 3])
        shuffled = shuffle_train_window_labels(grouped, seed=5)
        self.assertIs(shuffled["validation"], grouped["validation"])
        self.assertIs(shuffled["audit"], grouped["audit"])

    def test_only_modal_label_changes(self) -> None:
        grouped = make_grouped([1, 2, 3, 4])
        shuffled = shuffle_train_window_labels(grouped, seed=5)
        for original, new in zip(grouped["train"], shuffled["train"]):
            self.assertEqual(original.raw_start, new.raw_start)
            self.assertEqual(original.raw_end, new.raw_end)
            self.assertEqual(original.segment_index, new.segment_index)
            self.assertEqual(original.split, new.split)


if __name__ == "__main__":
    unittest.main()
