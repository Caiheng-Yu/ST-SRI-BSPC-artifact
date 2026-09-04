import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bspc_revision.st_sri_detector import (
    DetectorConfig,
    NullThresholds,
    calibrate_thresholds,
    curve_statistics,
    detect_peak,
    eligible_lag_mask,
    exclude_overlaps,
    prepare_curve,
    summarize_decisions,
    surrogate_statistics,
)

FS = 2000
STRIDE_SAMPLES = 1
MAX_LAG_MS = 150.0


def make_lags(config: DetectorConfig) -> np.ndarray:
    stride_ms = config.stride_samples * 1000.0 / config.fs
    n_lags = int(round(config.max_lag_ms / stride_ms))
    return stride_ms * np.arange(1, n_lags + 1)


def make_config(**overrides) -> DetectorConfig:
    defaults = dict(
        fs=FS,
        block_samples=10,
        max_lag_ms=MAX_LAG_MS,
        stride_samples=STRIDE_SAMPLES,
        smooth_sigma_bins=2.0,
        alpha=0.05,
    )
    defaults.update(overrides)
    return DetectorConfig(**defaults)


class DetectorConfigTest(unittest.TestCase):
    def test_invalid_alpha_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_config(alpha=0.0)
        with self.assertRaises(ValueError):
            make_config(alpha=1.0)

    def test_unknown_gated_statistic_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_config(gated_statistics=("peak_lag_ms",))

    def test_exclusion_equals_block_width(self) -> None:
        config = make_config()
        self.assertAlmostEqual(config.exclusion_ms, 5.0)
        self.assertAlmostEqual(config.bin_width_ms, 0.5)


class OverlapExclusionTest(unittest.TestCase):
    def test_mask_excludes_sub_block_width_lags(self) -> None:
        config = make_config()
        lags = make_lags(config)
        mask = eligible_lag_mask(lags, config)
        self.assertEqual(int(mask.sum()), 291)  # 5.0 ms 到 150.0 ms，步长 0.5 ms
        self.assertFalse(mask[lags < 5.0].any())
        self.assertTrue(mask[lags >= 5.0].all())

    def test_exclusion_precedes_smoothing(self) -> None:
        """重叠区的尖峰不得通过平滑泄漏进有资格区域。"""
        config = make_config()
        lags = make_lags(config)
        curve = np.zeros_like(lags)
        curve[lags < config.exclusion_ms] = 100.0  # 仅在重叠区放置尖峰

        eligible_lags, eligible_values = exclude_overlaps(lags, curve, config)
        self.assertTrue(np.all(eligible_values == 0.0))

        prepared_lags, prepared = prepare_curve(lags, curve, config)
        np.testing.assert_array_equal(prepared_lags, eligible_lags)
        np.testing.assert_allclose(prepared, gaussian_filter1d(eligible_values, sigma=2.0))
        self.assertEqual(float(prepared.max()), 0.0)

    def test_smoothing_only_on_retained_bins(self) -> None:
        config = make_config()
        rng = np.random.default_rng(7)
        lags = make_lags(config)
        curve = rng.standard_normal(lags.size)
        _, eligible_values = exclude_overlaps(lags, curve, config)
        _, prepared = prepare_curve(lags, curve, config)
        np.testing.assert_allclose(prepared, gaussian_filter1d(eligible_values, sigma=2.0))

    def test_non_increasing_lags_rejected(self) -> None:
        config = make_config()
        lags = make_lags(config)[::-1]
        with self.assertRaises(ValueError):
            exclude_overlaps(lags, np.zeros_like(lags), config)

    def test_nan_curve_rejected(self) -> None:
        config = make_config()
        lags = make_lags(config)
        curve = np.zeros_like(lags)
        curve[3] = np.nan
        with self.assertRaises(ValueError):
            exclude_overlaps(lags, curve, config)


class SurrogateNullTest(unittest.TestCase):
    def test_permute_destroys_structure_but_keeps_noise_floor(self) -> None:
        config = make_config()
        rng = np.random.default_rng(11)
        lags = make_lags(config)
        curve = rng.standard_normal(lags.size)
        stats = surrogate_statistics(lags, curve, config, 200, rng, mode="permute")
        self.assertEqual(len(stats), 200)
        # 纯噪声的置乱代理峰值应远低于同曲线加显著峰时的峰值
        noisy = curve_statistics(lags, curve, config)
        bumped = curve.copy()
        bump = 8.0 * np.exp(-0.5 * ((lags - 60.0) / 2.0) ** 2)
        bumped += bump
        bumped_stats = curve_statistics(lags, bumped, config)
        self.assertGreater(bumped_stats["peak_height"], max(s["peak_height"] for s in stats))
        self.assertGreater(noisy["positive_mass"], 0.0)

    def test_shift_destroys_peak_position(self) -> None:
        config = make_config()
        lags = make_lags(config)
        curve = np.zeros_like(lags)
        curve[np.argmin(np.abs(lags - 60.0))] = 5.0
        rng = np.random.default_rng(3)
        stats = surrogate_statistics(lags, curve, config, 50, rng, mode="shift")
        peak_lags = {round(entry["peak_lag_ms"], 3) for entry in stats}
        self.assertGreater(len(peak_lags), 1)

    def test_invalid_mode_and_count_rejected(self) -> None:
        config = make_config()
        lags = make_lags(config)
        curve = np.zeros_like(lags)
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            surrogate_statistics(lags, curve, config, 10, rng, mode="bad")
        with self.assertRaises(ValueError):
            surrogate_statistics(lags, curve, config, 0, rng)


class CalibrationAndDetectionTest(unittest.TestCase):
    def test_empty_null_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_thresholds([], make_config())

    def test_threshold_matches_higher_quantile(self) -> None:
        config = make_config(alpha=0.1)
        rng = np.random.default_rng(5)
        null_stats = [
            {"positive_mass": float(value), "peak_height": float(value / 2)}
            for value in rng.uniform(0, 1, size=100)
        ]
        thresholds = calibrate_thresholds(null_stats, config)
        expected = float(
            np.quantile([entry["positive_mass"] for entry in null_stats], 0.9, method="higher")
        )
        self.assertEqual(thresholds.n_null, 100)
        self.assertEqual(thresholds.threshold_for("positive_mass"), expected)

    def test_missing_gated_threshold_rejected(self) -> None:
        config = make_config()
        lags = make_lags(config)
        thresholds = NullThresholds(alpha=0.05, n_null=10, thresholds={"positive_mass": 0.0})
        with self.assertRaises(KeyError):
            detect_peak(lags, np.zeros_like(lags), config, thresholds)

    def test_no_eligible_bins_is_unsupported(self) -> None:
        config = make_config(max_lag_ms=4.0)  # 所有分箱都在重叠区内
        lags = make_lags(config)
        thresholds = NullThresholds(
            alpha=0.05, n_null=10, thresholds={"positive_mass": 0.0, "peak_height": 0.0}
        )
        result = detect_peak(lags, np.ones_like(lags), config, thresholds)
        self.assertFalse(result.supported)
        self.assertEqual(result.reason, "no_eligible_bins")
        self.assertIsNone(result.peak_lag_ms)

    def test_clear_peak_supported_and_noise_rejected(self) -> None:
        """独立零模型曲线标定阈值：显著峰应通过，纯噪声应大多被拒绝。"""
        config = make_config()
        lags = make_lags(config)
        rng = np.random.default_rng(23)

        null_curves = rng.standard_normal((300, lags.size))
        null_stats = [curve_statistics(lags, null_curves[i], config) for i in range(300)]
        thresholds = calibrate_thresholds(null_stats, config)

        bump = 6.0 * np.exp(-0.5 * ((lags - 60.0) / 2.0) ** 2)
        supported_count = 0
        for _ in range(10):
            curve = rng.standard_normal(lags.size) + bump
            result = detect_peak(lags, curve, config, thresholds)
            self.assertTrue(result.supported)
            self.assertAlmostEqual(result.peak_lag_ms, 60.0, delta=2.0)
            supported_count += 1
        self.assertEqual(supported_count, 10)

        noise_decisions = [
            detect_peak(lags, rng.standard_normal(lags.size), config, thresholds)
            for _ in range(200)
        ]
        noise_support_rate = sum(d.supported for d in noise_decisions) / len(noise_decisions)
        self.assertLessEqual(noise_support_rate, 0.10)

    def test_surrogate_null_controls_false_positives(self) -> None:
        """无零模型曲线时，对每条噪声曲线的置乱代理标定也应压制误报。"""
        config = make_config()
        lags = make_lags(config)
        rng = np.random.default_rng(29)
        supported = 0
        n_curves = 100
        for _ in range(n_curves):
            curve = rng.standard_normal(lags.size)
            stats = surrogate_statistics(lags, curve, config, 200, rng)
            thresholds = calibrate_thresholds(stats, config)
            supported += int(detect_peak(lags, curve, config, thresholds).supported)
        self.assertLessEqual(supported / n_curves, 0.15)


class SummaryTest(unittest.TestCase):
    def test_empty_decisions_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_decisions([])

    def test_summary_counts(self) -> None:
        config = make_config()
        lags = make_lags(config)
        thresholds = NullThresholds(
            alpha=0.05, n_null=10, thresholds={"positive_mass": 0.0, "peak_height": 0.0}
        )
        supported = detect_peak(lags, np.abs(lags), config, thresholds)
        config_short = make_config(max_lag_ms=4.0)
        unsupported = detect_peak(
            make_lags(config_short), np.ones(8), config_short, thresholds
        )
        summary = summarize_decisions([supported, unsupported])
        self.assertEqual(summary["n_curves"], 2)
        self.assertEqual(summary["n_supported"], 1)
        self.assertAlmostEqual(summary["support_rate"], 0.5)
        self.assertAlmostEqual(summary["no_support_fraction"], 0.5)
        self.assertEqual(summary["reason_counts"].get("no_eligible_bins"), 1)


class CliSmokeTest(unittest.TestCase):
    def test_cli_end_to_end(self) -> None:
        config = make_config()
        lags = make_lags(config)
        rng = np.random.default_rng(31)
        curves = rng.standard_normal((12, lags.size))
        bump = 6.0 * np.exp(-0.5 * ((lags - 60.0) / 2.0) ** 2)
        for index in range(3):
            curves[index] += bump
        with tempfile.TemporaryDirectory() as tmpdir:
            curves_path = Path(tmpdir) / "curves.npz"
            output_path = Path(tmpdir) / "report.json"
            np.savez(curves_path, lags_ms=lags, curves=curves)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "st_sri_detector.py"),
                    "--curves",
                    str(curves_path),
                    "--surrogate-count",
                    "50",
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["n_curves"], 12)
            self.assertEqual(len(report["decisions"]), 12)
            self.assertIn("positive_mass", report["thresholds"]["thresholds"])
            self.assertTrue(report["null_source"].startswith("surrogate_permute"))


if __name__ == "__main__":
    unittest.main()
