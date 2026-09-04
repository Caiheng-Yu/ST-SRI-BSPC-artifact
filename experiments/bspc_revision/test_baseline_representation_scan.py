import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bspc_revision.baseline_representation_scan import (  # noqa: E402
    WINDOW_SAMPLES,
    assert_train_partition_only,
    build_positional_mean,
    interpolation_fill,
    sample_background_records,
    stack_background,
)
from experiments.bspc_revision.leakage_free_db2 import (  # noqa: E402
    TrialRecord,
    build_window_records,
)

SCRIPT = PROJECT_ROOT / "experiments" / "bspc_revision" / "baseline_representation_scan.py"
N_CHANNELS = 12


def make_trial(subject: int, split: str, active_class: int, raw_start: int, raw_end: int,
               active_start: int, active_end: int, segment_index: int) -> TrialRecord:
    return TrialRecord(
        subject=subject,
        split=split,
        active_class=active_class,
        repetition_index=0,
        segment_index=segment_index,
        active_start=active_start,
        active_end=active_end,
        raw_start=raw_start,
        raw_end=raw_end,
    )


def make_synthetic(total: int = 4000, active_ranges=((1000, 1500), (3000, 3500))) -> tuple:
    """构造含 train/audit 两个 trial 的合成数据。"""
    data = np.random.default_rng(0).standard_normal((total, N_CHANNELS)).astype(np.float32)
    labels = np.zeros(total, dtype=np.int64)
    for start, end in active_ranges:
        labels[start:end] = 1
    train_trial = make_trial(1, "train", 1, 0, 2000, 1000, 1500, 1)
    audit_trial = make_trial(1, "audit", 1, 2000, 4000, 3000, 3500, 3)
    return data, labels, [train_trial, audit_trial]


class InterpolationFillTest(unittest.TestCase):
    def test_interior_linear_ramp_and_endpoint_continuity(self) -> None:
        T, C = 50, 3
        torch.manual_seed(0)
        x = torch.randn(T, C, dtype=torch.float32)
        fill = interpolation_fill(x)

        a, b = 20, 30
        masked = x.unsqueeze(0).clone()
        fill(masked, a, b, [0])
        denom = float(b - (a - 1))
        step = (x[b] - x[a - 1]) / denom
        expected = np.stack([
            (x[a - 1] + (x[b] - x[a - 1]) * ((j + 1) / denom)).numpy()
            for j in range(b - a)
        ])
        np.testing.assert_allclose(masked[0, a:b, :].numpy(), expected, rtol=1e-5, atol=1e-6)
        # 端点连续：填充首元素紧邻 x[a-1]，末元素紧邻 x[b]，均差一个插值步长。
        np.testing.assert_allclose(masked[0, a, :].numpy(), (x[a - 1] + step).numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(masked[0, b - 1, :].numpy(), (x[b] - step).numpy(), rtol=1e-5, atol=1e-6)
        # 未遮挡区域保持不变。
        np.testing.assert_array_equal(masked[0, :a, :].numpy(), x[:a, :].numpy())
        np.testing.assert_array_equal(masked[0, b:, :].numpy(), x[b:, :].numpy())

    def test_left_boundary_copies_right_anchor(self) -> None:
        T, C = 50, 3
        torch.manual_seed(1)
        x = torch.randn(T, C, dtype=torch.float32)
        fill = interpolation_fill(x)
        masked = x.unsqueeze(0).clone()
        fill(masked, 0, 10, [0])
        expected = x[10].numpy().reshape(1, C).repeat(10, axis=0)
        np.testing.assert_allclose(masked[0, 0:10, :].numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_right_boundary_copies_left_anchor(self) -> None:
        T, C = 50, 3
        torch.manual_seed(2)
        x = torch.randn(T, C, dtype=torch.float32)
        fill = interpolation_fill(x)
        masked = x.unsqueeze(0).clone()
        fill(masked, 40, T, [0])
        expected = x[39].numpy().reshape(1, C).repeat(T - 40, axis=0)
        np.testing.assert_allclose(masked[0, 40:T, :].numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_multiple_rows_filled_identically(self) -> None:
        T, C = 50, 3
        torch.manual_seed(3)
        x = torch.randn(T, C, dtype=torch.float32)
        fill = interpolation_fill(x)
        masked = x.unsqueeze(0).repeat(3, 1, 1).clone()
        fill(masked, 20, 30, [0, 1, 2])
        for row in (0, 1, 2):
            np.testing.assert_array_equal(masked[row].numpy(), masked[0].numpy())


class BaselineTemplateTest(unittest.TestCase):
    def test_rest_mean_and_positional_mean_shapes_and_sources(self) -> None:
        data, labels, trials = make_synthetic()
        mean = np.zeros(N_CHANNELS, dtype=np.float32)
        std = np.ones(N_CHANNELS, dtype=np.float32)

        windows = build_window_records(labels, [t for t in trials if t.split == "train"],
                                       WINDOW_SAMPLES, 100)
        rest_records = [w for w in windows if w.modal_label == 0]
        active_records = [w for w in windows if w.modal_label != 0]
        self.assertTrue(rest_records, "expected rest-modal windows")
        self.assertTrue(active_records, "expected active-modal windows")

        # rest_mean 来源：仅静息窗。
        background_records = sample_background_records(labels, trials, count=len(rest_records), seed=1)
        self.assertEqual(len(background_records), len(rest_records))
        for record in background_records:
            self.assertEqual(record.split, "train")
            self.assertEqual(record.modal_label, 0)
        rest_mean = stack_background(data, background_records, mean, std).mean(dim=0)
        self.assertEqual(tuple(rest_mean.shape), (WINDOW_SAMPLES, N_CHANNELS))
        expected_rest = np.stack([
            (data[w.raw_start:w.raw_end].astype(np.float32) - mean) / std for w in rest_records
        ]).mean(axis=0)
        np.testing.assert_allclose(rest_mean.numpy(), expected_rest, rtol=1e-5, atol=1e-6)

        # positional_mean 来源：训练分区全部窗口（含运动类）。
        positional_mean = build_positional_mean(data, labels, trials, mean, std)
        self.assertEqual(tuple(positional_mean.shape), (WINDOW_SAMPLES, N_CHANNELS))
        expected_pos = np.stack([
            (data[w.raw_start:w.raw_end].astype(np.float32) - mean) / std for w in windows
        ]).mean(axis=0)
        np.testing.assert_allclose(positional_mean.numpy(), expected_pos, rtol=1e-5, atol=1e-6)
        # 两者来源不同，故不应完全相等（positional 含运动窗）。
        self.assertFalse(np.allclose(positional_mean.numpy(), rest_mean.numpy()))


class BackgroundSamplingTest(unittest.TestCase):
    def test_background_sampling_is_train_only(self) -> None:
        data, labels, trials = make_synthetic()
        background_records = sample_background_records(labels, trials, count=8, seed=5)
        self.assertTrue(background_records)
        for record in background_records:
            self.assertEqual(record.split, "train")
        # 断言与 audit trial 无原始采样重叠。
        assert_train_partition_only(background_records, trials)


class SmokeTest(unittest.TestCase):
    def test_rest_mean_smoke_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            cmd = [
                sys.executable, str(SCRIPT),
                "--subjects", "1",
                "--max-records", "2",
                "--baselines", "rest_mean",
                "--device", "cpu",
                "--threads", "2",
                "--output-dir", str(output_dir),
            ]
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            npz_path = output_dir / "S01_rest_mean.npz"
            self.assertTrue(npz_path.exists(), f"missing {npz_path}")
            with np.load(npz_path) as payload:
                for key in ("curve_prob_signed", "curve_prob_positive", "curve_prob_abs",
                            "curve_logit_signed", "curve_logit_positive", "curve_logit_abs"):
                    self.assertIn(key, payload.files)
                lags_ms = payload["lags_ms"]
                n_lags = int(lags_ms.shape[0])
                self.assertGreater(n_lags, 0)
                for key in ("curve_prob_signed", "curve_prob_positive", "curve_prob_abs",
                            "curve_logit_signed", "curve_logit_positive", "curve_logit_abs"):
                    self.assertEqual(tuple(payload[key].shape), (2, n_lags))
                meta = json.loads(str(payload["meta_json"]))
                self.assertEqual(meta["subject"], 1)
                self.assertEqual(meta["baseline"], "rest_mean")
                self.assertIn("checkpoint_sha256", meta)
                self.assertIn("background_source", meta)


if __name__ == "__main__":
    unittest.main()
