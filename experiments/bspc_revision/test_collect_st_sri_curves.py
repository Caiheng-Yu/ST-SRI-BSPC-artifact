import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import LSTMModel
from experiments.bspc_revision.collect_st_sri_curves import (
    MODEL_KWARGS,
    build_variant_model,
    sample_background_windows,
    scramble_state_dict,
    variant_seed,
)
from experiments.bspc_revision.leakage_free_db2 import TrialRecord


def make_state_dict(seed: int) -> dict:
    torch.manual_seed(seed)
    model = LSTMModel(**MODEL_KWARGS)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


class ScrambleStateDictTest(unittest.TestCase):
    def test_preserves_shapes_and_value_multisets(self) -> None:
        state_dict = make_state_dict(101)
        scrambled = scramble_state_dict(state_dict, seed=7)
        self.assertEqual(set(scrambled), set(state_dict))
        for key, tensor in state_dict.items():
            self.assertEqual(tuple(scrambled[key].shape), tuple(tensor.shape))
            np.testing.assert_array_equal(
                np.sort(scrambled[key].numpy().ravel()),
                np.sort(tensor.numpy().ravel()),
                err_msg=key,
            )

    def test_actually_permutes_parameters(self) -> None:
        state_dict = make_state_dict(101)
        scrambled = scramble_state_dict(state_dict, seed=7)
        changed = any(
            not torch.equal(scrambled[key], tensor)
            for key, tensor in state_dict.items()
            if tensor.numel() > 1
        )
        self.assertTrue(changed)

    def test_deterministic_per_seed(self) -> None:
        state_dict = make_state_dict(101)
        first = scramble_state_dict(state_dict, seed=7)
        second = scramble_state_dict(state_dict, seed=7)
        other = scramble_state_dict(state_dict, seed=8)
        for key in state_dict:
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertFalse(torch.equal(first["fc.weight"], other["fc.weight"]))

    def test_does_not_modify_input(self) -> None:
        state_dict = make_state_dict(101)
        snapshot = {key: value.clone() for key, value in state_dict.items()}
        scramble_state_dict(state_dict, seed=7)
        for key in state_dict:
            self.assertTrue(torch.equal(state_dict[key], snapshot[key]), key)


class VariantModelTest(unittest.TestCase):
    def test_trained_loads_checkpoint_weights(self) -> None:
        state_dict = make_state_dict(202)
        model = build_variant_model("trained", {"model_state_dict": state_dict}, seed=1)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state_dict[key]), key)

    def test_reinit_deterministic_per_seed(self) -> None:
        first = build_variant_model("reinit", {"model_state_dict": make_state_dict(1)}, seed=42)
        second = build_variant_model("reinit", {"model_state_dict": make_state_dict(1)}, seed=42)
        third = build_variant_model("reinit", {"model_state_dict": make_state_dict(1)}, seed=43)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)
        self.assertFalse(torch.equal(first.state_dict()["fc.weight"], third.state_dict()["fc.weight"]))

    def test_param_scramble_matches_scrambled_weights(self) -> None:
        state_dict = make_state_dict(303)
        model = build_variant_model("param_scramble", {"model_state_dict": state_dict}, seed=9)
        expected = scramble_state_dict(state_dict, seed=9)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]), key)

    def test_variant_seed_is_subject_and_variant_specific(self) -> None:
        seeds = {
            variant_seed(20260816, subject, variant)
            for subject in (1, 2)
            for variant in ("trained", "reinit", "param_scramble")
        }
        self.assertEqual(len(seeds), 6)

    def test_unknown_variant_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_variant_model("bad", {"model_state_dict": make_state_dict(1)}, seed=1)


class BackgroundWindowTest(unittest.TestCase):
    def make_trial(self, raw_start: int, raw_end: int) -> TrialRecord:
        return TrialRecord(
            subject=1,
            split="train",
            active_class=1,
            repetition_index=0,
            segment_index=1,
            active_start=raw_start + 600,
            active_end=raw_start + 700,
            raw_start=raw_start,
            raw_end=raw_end,
        )

    def test_only_rest_modal_windows_are_sampled(self) -> None:
        rng = np.random.default_rng(5)
        total = 600 * 8
        data = rng.standard_normal((total, 12)).astype(np.float32)
        labels = np.zeros(total, dtype=np.int64)
        trial = self.make_trial(0, total)
        mean = np.zeros(12, dtype=np.float32)
        std = np.ones(12, dtype=np.float32)
        background = sample_background_windows(data, labels, [trial], mean, std, count=4, seed=3)
        self.assertEqual(background.shape, (4, 600, 12))
        candidates = [
            np.asarray(data[start : start + 600], dtype=np.float32) for start in range(0, total - 599, 100)
        ]
        for i in range(background.shape[0]):
            match = any(np.allclose(background[i].numpy(), candidate) for candidate in candidates)
            self.assertTrue(match)

    def test_no_rest_windows_rejected(self) -> None:
        total = 600 * 4
        data = np.zeros((total, 12), dtype=np.float32)
        labels = np.ones(total, dtype=np.int64)
        trial = self.make_trial(0, total)
        with self.assertRaises(ValueError):
            sample_background_windows(
                data,
                labels,
                [trial],
                np.zeros(12, dtype=np.float32),
                np.ones(12, dtype=np.float32),
                count=2,
                seed=3,
            )


class PhaseWindowCsvTest(unittest.TestCase):
    def write_csv(self, directory: str) -> Path:
        import csv as csv_module

        path = Path(directory) / "phase_windows.csv"
        header = [
            "subject", "split", "active_class", "repetition_index", "segment_index",
            "window_kind", "phase_ms", "phase_samples", "reference_sample",
            "current_endpoint_sample", "current_block_start_sample", "current_block_end_sample",
            "input_start_sample", "input_end_sample", "reference_in_window",
            "current_endpoint_in_window", "current_block_start_in_window",
            "current_block_end_in_window",
        ]
        rows = []
        for subject, kind, seg in ((1, "onset", 11), (1, "steady", 12), (2, "onset", 21)):
            endpoint = 5000 + seg
            rows.append([
                subject, "audit", 3, 0, seg, kind, 0.0, 0, endpoint,
                endpoint, endpoint - 9, endpoint + 1, endpoint - 599, endpoint + 1,
                599, 599, 590, 600,
            ])
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv_module.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_filters_by_subject_and_kind(self) -> None:
        from experiments.bspc_revision.collect_st_sri_curves import load_phase_window_records

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self.write_csv(tmpdir)
            records = load_phase_window_records(csv_path, 1, ("onset", "offset", "steady"))
            self.assertEqual(len(records), 2)
            self.assertEqual({record.window_kind for record in records}, {"onset", "steady"})
            onset_only = load_phase_window_records(csv_path, 1, ("onset",))
            self.assertEqual(len(onset_only), 1)
            record = onset_only[0]
            self.assertEqual(record.subject, 1)
            self.assertEqual(record.segment_index, 11)
            self.assertIsInstance(record.input_start_sample, int)
            self.assertIsInstance(record.phase_ms, float)
            self.assertEqual(record.input_end_sample - record.input_start_sample, 600)

    def test_missing_subject_rejected(self) -> None:
        from experiments.bspc_revision.collect_st_sri_curves import load_phase_window_records

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = self.write_csv(tmpdir)
            with self.assertRaises(ValueError):
                load_phase_window_records(csv_path, 40, ("onset",))


if __name__ == "__main__":
    unittest.main()
