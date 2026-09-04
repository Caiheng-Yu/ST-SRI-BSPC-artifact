"""R015b 支持率归因：97.5% (mean pooling) vs 18.7% (last pooling) 差异来源判定。

交叉判决 + 量级统计。两套审计的 DetectorConfig 相同（fs=2000, block=10,
max_lag=150, sigma=2），判决统计量可直接互判。

Usage (from repo root):
    python 00_research/docs/audits/r015b_support_attribution.py
"""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results" / "bspc_revision_v2"  # 05_revision_experiments/results


def main() -> None:
    r7 = json.loads((ROOT / "r007_detector_report.json").read_text(encoding="utf-8"))
    r15 = json.loads((ROOT / "r015b_full_detector.json").read_text(encoding="utf-8"))
    t7 = r7["thresholds"]["thresholds"]
    t15 = r15["thresholds"]["thresholds"]

    def stats(report):
        pm = np.array([d["statistics"]["positive_mass"] for d in report["decisions"]])
        ph = np.array([d["statistics"]["peak_height"] for d in report["decisions"]])
        return pm, ph

    pm7, ph7 = stats(r7)     # last pooling (r005b) curves
    pm15, ph15 = stats(r15)  # mean pooling (r015b) curves

    def judge(pm, ph, th):
        return (pm >= th["positive_mass"]) & (ph >= th["peak_height"])

    print("thresholds r007 :", t7)
    print("thresholds r015b:", t15)
    print(f"last curves x last thresholds : {judge(pm7, ph7, t7).mean():.1%}  (r007 正式审计)")
    print(f"last curves x mean thresholds : {judge(pm7, ph7, t15).mean():.1%}  (交叉判决)")
    print(f"mean curves x last thresholds : {judge(pm15, ph15, t7).mean():.1%}  (交叉判决)")
    print(f"mean curves x mean thresholds : {judge(pm15, ph15, t15).mean():.1%}  (r015b 正式审计)")
    print(f"\npositive_mass 中位: last={np.median(pm7):.4g}  mean={np.median(pm15):.4g}  "
          f"last-null 阈值/mean-null 阈值 比值={t7['positive_mass']/t15['positive_mass']:.0f}")

    # last-pooling label-shuffle null 量级（r006_curves_shuffled trained）
    masses = []
    for f in sorted(glob.glob(str(ROOT / "r006_curves_shuffled" / "S*_trained.npz"))):
        d = np.load(f, allow_pickle=False)
        masses.append(np.abs(np.asarray(d["interactions"], dtype=np.float64)).sum(axis=1))
    m = np.concatenate(masses)
    print(f"\nlast-pooling label-shuffle null (n={len(m)}): 中位 {np.median(m):.4g}  "
          f"p95 {np.percentile(m, 95):.4g}  (mean-null 阈值低 {np.median(m)/t15['positive_mass']:.0f} 倍)")


if __name__ == "__main__":
    main()
