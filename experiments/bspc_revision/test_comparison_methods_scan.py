import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import LSTMModel, ST_SRI_Interpreter  # noqa: E402
from experiments.bspc_revision import comparison_methods_scan as mod  # noqa: E402
from experiments.bspc_revision.leakage_free_db2 import (  # noqa: E402
    build_trial_records,
)
from experiments.bspc_revision.onset_protocol import build_onset_audit_records  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "DB2"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_bspc_v2" / "r005b_balanced_full_seed20260815"


def _load_s1_window(max_lag_ms: float = 20.0):
    """加载 S1 真实数据、检查点与首个 onset 审计窗口（合成背景以加速一致性测试）。"""
    data = np.load(DATA_ROOT / "S1_data.npy", mmap_mode="r")
    labels = np.load(DATA_ROOT / "S1_label.npy", mmap_mode="r")
    trials = build_trial_records(labels, 1, mod.SPLIT_SEED, mod.PURGE_SAMPLES)
    records = build_onset_audit_records(
        trials, labels, split="audit", phases_ms=[0.0], fs=2000,
        window_samples=600, block_samples=10,
    )
    record = records[0]
    checkpoint = torch.load(
        CHECKPOINT_DIR / "S01_best.pth", map_location="cpu", weights_only=False,
    )
    mean = np.asarray(checkpoint["training_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["training_std"], dtype=np.float32)
    model = LSTMModel(**mod.MODEL_KWARGS).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    x = mod.normalized_record_window(data, record, mean, std)
    background = torch.zeros((4, 600, 12), dtype=torch.float32)
    interpreter = ST_SRI_Interpreter(model, background, device=torch.device("cpu"))
    with torch.no_grad():
        probs = torch.softmax(model(x.unsqueeze(0)), dim=1)[0]
    target_cls = int(torch.argmax(probs).item())
    return interpreter, x, target_cls


class CliParseTest(unittest.TestCase):
    def test_defaults(self) -> None:
        args = mod.parse_args([])
        self.assertEqual(args.subjects, [1])
        # Dynamask 已暂缓：默认只跑 occlusion + timeshap
        self.assertEqual(args.methods, ["occlusion", "timeshap"])
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.max_lag_ms, 150.0)
        self.assertEqual(args.group_size, 10)
        self.assertEqual(args.phases_ms, [0, 50, 100, 150])
        self.assertEqual(args.checkpoint_dir, CHECKPOINT_DIR)

    def test_methods_validation(self) -> None:
        args = mod.parse_args(["--methods", "occlusion", "timeshap"])
        self.assertEqual(args.methods, ["occlusion", "timeshap"])
        with self.assertRaises(SystemExit):
            mod.parse_args(["--methods", "occlusion", "bogus"])

    def test_subjects_and_all_subjects(self) -> None:
        args = mod.parse_args(["--subjects", "3", "5"])
        self.assertEqual(args.subjects, [3, 5])
        self.assertFalse(args.all_subjects)
        args = mod.parse_args(["--all-subjects"])
        self.assertTrue(args.all_subjects)


class OcclusionConsistencyTest(unittest.TestCase):
    def test_matches_scan_fast_on_s1_window(self) -> None:
        interpreter, x, target_cls = _load_s1_window()
        lags_ms, interaction = mod.compute_occlusion(
            interpreter, x, target_cls,
            max_lag_ms=20.0, stride=1, block_size=10, current_endpoint=599,
        )
        lags_ms_ref, synergy, redundancy = interpreter.scan_fast(
            x, max_lag_ms=20.0, stride=1, block_size=10,
            current_endpoint=599, target_cls=target_cls,
        )
        np.testing.assert_allclose(lags_ms, np.asarray(lags_ms_ref, dtype=np.float32))
        expected = np.asarray(synergy, dtype=np.float32) + np.asarray(redundancy, dtype=np.float32)
        self.assertEqual(interaction.shape, expected.shape)
        denom = np.abs(expected) + 1e-12
        rel_error = float(np.max(np.abs(interaction - expected) / denom))
        self.assertLess(rel_error, 1e-4)


class TimeshapFunctionTest(unittest.TestCase):
    def test_returns_correct_shape_on_small_lstm(self) -> None:
        torch.manual_seed(0)
        model = LSTMModel(input_size=12, hidden_size=32, num_layers=1, num_classes=18, dropout=0.0).eval()
        x = torch.randn(600, 12)
        baseline = torch.zeros(600, 12)
        attr = mod.compute_timeshap_groups(
            model, x, baseline, target_cls=0, group_size=10, device=torch.device("cpu"),
        )
        self.assertEqual(attr.shape, (60,))
        self.assertEqual(attr.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(attr)))

    def test_group_blocks_boundaries(self) -> None:
        blocks = mod.group_blocks(600, 10)
        self.assertEqual(len(blocks), 60)
        self.assertEqual(blocks[0], (0, 10))
        self.assertEqual(blocks[-1], (590, 600))
        # 非整除长度也正确
        self.assertEqual(mod.group_blocks(25, 10), [(0, 10), (10, 20), (20, 25)])


class DynamaskGracefulTest(unittest.TestCase):
    def _make_args(self):
        return mod.parse_args(["--device", "cpu"])

    def test_not_installed_path_does_not_raise(self) -> None:
        if mod.DYNAMASK_AVAILABLE:
            self.skipTest("dynamask installed; graceful path not exercised")
        self.assertFalse(mod.DYNAMASK_AVAILABLE)
        self.assertIsNotNone(mod.DYNAMASK_ERROR)
        args = self._make_args()
        common = {
            "ids": ["S01_onset_seg000_ph0", "S01_onset_seg001_ph0"],
            "active_class": [3, 5],
            "predicted_class": [3, 5],
            "correct": [1, 1],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "S01_best.pth"
            checkpoint_path.write_bytes(b"dummy")
            payload = mod.run_method(
                subject=1,
                method="dynamask",
                args=args,
                device=torch.device("cpu"),
                model=None,
                interpreter=None,
                baseline=None,
                windows=[],
                target_classes=[],
                common=common,
                checkpoint_path=checkpoint_path,
                records_source="onset_protocol",
                background_count=64,
            )
        meta = json.loads(str(payload["meta_json"]))
        self.assertEqual(meta["status"], "not_installed")
        self.assertEqual(payload["attribution"].shape, (2, 0))
        self.assertEqual(payload["aopc_curve"].shape, (2, 0))
        np.testing.assert_array_equal(payload["correct"], np.asarray([1, 1]))
        # 汇总路径也应产出 not_installed 标量占位
        summary = mod.summarize(1, "dynamask", payload["attribution"], payload["aopc_curve"], meta)
        self.assertEqual(summary["status"], "not_installed")
        self.assertIsNone(summary["aopc_mean"])

    @unittest.skipUnless(mod.DYNAMASK_AVAILABLE, "dynamask not installed")
    def test_installed_smoke_small_window(self) -> None:
        torch.manual_seed(1)
        model = LSTMModel(input_size=12, hidden_size=16, num_layers=1, num_classes=18, dropout=0.0).eval()
        x = torch.randn(600, 12)
        baseline = torch.zeros(600, 12)
        mask, loss = mod.compute_dynamask_groups(
            model, x, baseline, target_cls=0, group_size=10, device=torch.device("cpu"),
        )
        self.assertEqual(mask.shape, (60,))


class EndToEndSmokeTest(unittest.TestCase):
    def test_occlusion_smoke_generates_npz_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            argv = [
                "--subjects", "1",
                "--data-root", str(DATA_ROOT),
                "--max-records", "2",
                "--methods", "occlusion",
                "--device", "cpu",
                "--threads", "2",
                "--output-dir", str(output_dir),
            ]
            mod.main(argv)

            npz_path = output_dir / "S01_occlusion.npz"
            summary_path = output_dir / "summary.json"
            self.assertTrue(npz_path.exists())
            self.assertTrue(summary_path.exists())

            with np.load(npz_path) as npz:
                required = {"attribution", "aopc_curve", "correct", "active_class",
                            "predicted_class", "ids", "meta_json"}
                self.assertTrue(required.issubset(set(npz.files)))
                self.assertEqual(npz["attribution"].shape, (2, 300))
                self.assertEqual(npz["aopc_curve"].shape, (2, 301))
                self.assertEqual(npz["correct"].shape, (2,))
                self.assertEqual(npz["active_class"].shape, (2,))
                self.assertEqual(npz["predicted_class"].shape, (2,))
                self.assertEqual(npz["ids"].shape, (2,))
                self.assertEqual(npz["lags_ms"].shape, (300,))
                meta = json.loads(str(npz["meta_json"]))
                self.assertEqual(meta["status"], "ok")
                self.assertEqual(meta["method"], "occlusion")

            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(summary["dynamask_installed"])
            results = summary["results"]
            self.assertEqual(len(results), 1)
            entry = results[0]
            self.assertEqual(entry["subject"], 1)
            self.assertEqual(entry["method"], "occlusion")
            self.assertEqual(entry["status"], "ok")
            self.assertIsInstance(entry["aopc_mean"], float)
            self.assertIsInstance(entry["stability"], float)
            self.assertIsInstance(entry["runtime_seconds_per_window_mean"], float)


if __name__ == "__main__":
    unittest.main()
