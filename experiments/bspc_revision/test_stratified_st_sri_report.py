import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bspc_revision.stratified_st_sri_report import (
    assign_quartile_bins,
    load_curve_metadata,
    load_decisions,
    stratify,
)


def make_npz(path: Path, n: int, subject: int = 1) -> list[str]:
    ids = [f"S{subject:02d}_onset_seg{i:03d}_ph0" for i in range(n)]
    rng = np.random.default_rng(1)
    np.savez(
        path,
        lags_ms=np.linspace(0.5, 150.0, 10),
        curves=rng.standard_normal((n, 10)),
        ids=np.asarray(ids),
        window_kind=np.asarray(["onset"] * (n - 1) + ["steady"]),
        phase_ms=np.zeros(n),
        active_class=np.asarray([1] * (n // 2) + [2] * (n - n // 2)),
        predicted_class=np.asarray([1] * (n // 2) + [2] * (n - n // 2)),
        correct=np.asarray([1] * (n // 2) + [0] * (n - n // 2)),
        target_prob=np.linspace(0.05, 0.95, n),
        meta_json=np.asarray(json.dumps({"subject": subject})),
    )
    return ids


def make_report(path: Path, ids: list[str], supported_ids: set[str]) -> None:
    decisions = [
        {
            "id": curve_id,
            "supported": curve_id in supported_ids,
            "peak_lag_ms": 60.0 if curve_id in supported_ids else None,
            "statistics": {
                "positive_mass": 1.0 if curve_id in supported_ids else 0.1,
                "signed_mass": 0.5,
                "peak_height": 0.8 if curve_id in supported_ids else 0.05,
                "peak_prominence": 0.3,
                "peak_lag_ms": 60.0 if curve_id in supported_ids else 10.0,
            },
            "reason": "supported" if curve_id in supported_ids else "below_null_threshold:positive_mass",
        }
        for curve_id in ids
    ]
    path.write_text(json.dumps({"decisions": decisions}, ensure_ascii=False), encoding="utf-8")


class StratifiedReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self.tmpdir.name)
        self.ids = make_npz(tmp / "curves.npz", n=8)
        self.supported = {self.ids[0], self.ids[1], self.ids[2]}  # 全部 correct=1 的前 4 条中支持 3 条
        make_report(tmp / "report.json", self.ids, self.supported)
        self.rows = load_curve_metadata([tmp / "curves.npz"])
        self.decisions = load_decisions(tmp / "report.json")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_metadata_rows(self) -> None:
        self.assertEqual(len(self.rows), 8)
        self.assertEqual(self.rows[0]["subject"], 1)
        self.assertEqual(self.rows[-1]["window_kind"], "steady")

    def test_correctness_strata(self) -> None:
        report = stratify(self.rows, self.decisions)
        correct = report["correct"]
        self.assertEqual(correct["1"]["n"], 4)
        self.assertEqual(correct["1"]["n_supported"], 3)
        self.assertAlmostEqual(correct["1"]["support_rate"], 0.75)
        self.assertEqual(correct["0"]["support_rate"], 0.0)
        self.assertAlmostEqual(correct["1"]["supported_peak_lag_ms_median"], 60.0)

    def test_class_and_kind_strata(self) -> None:
        report = stratify(self.rows, self.decisions)
        self.assertEqual(report["active_class"]["1"]["n"], 4)
        self.assertEqual(report["active_class"]["2"]["n"], 4)
        self.assertEqual(report["window_kind"]["onset"]["n"], 7)
        self.assertEqual(report["window_kind"]["steady"]["n"], 1)

    def test_quartile_bins_cover_all_rows(self) -> None:
        report = stratify(self.rows, self.decisions)
        total = sum(level["n"] for level in report["target_prob_quartile"].values())
        self.assertEqual(total, 8)
        bins = assign_quartile_bins(np.array([0.5, 0.5, 0.5]))
        self.assertEqual(bins, ["Q1", "Q1", "Q1"])

    def test_id_mismatch_rejected(self) -> None:
        bad = dict(self.decisions)
        bad.pop(self.ids[0])
        bad["alien"] = bad[self.ids[1]]
        with self.assertRaises(ValueError):
            stratify(self.rows, bad)


if __name__ == "__main__":
    unittest.main()
