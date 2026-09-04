"""R010 分析：基线 × 表征的稳定性汇总。

对 r010_baseline_repr 的 5 种基线分别合并 40 人曲线，用与 R008 相同的零分布
阈值做检测器判决（阈值只由零模型标定），汇总：
- 各基线的支持率与无支持峰比例（换基线结论是否稳健）；
- 6 种表征（prob/logit × signed/positive/abs）的原始统计量（换读法是否稳健）；
- 各基线相对 rest_mean 锚点的逐记录正质量秩相关（同窗口一致性）。

输出 r010_analysis/baseline_comparison.json 与 baseline_comparison.csv。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bspc_revision.merge_curve_npz import collect_files  # noqa: E402

BASELINES = ("rest_mean", "rest_sampled", "background_set", "interpolation", "positional_mean")
SCORE_MODES = ("prob", "logit")
CURVE_MODES = ("signed", "positive", "abs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r010_baseline_repr")
    parser.add_argument("--null-curves", type=Path, default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r006_merged" / "null_models.npz")
    parser.add_argument("--detector-script", type=Path, default=PROJECT_ROOT / "experiments" / "bspc_revision" / "st_sri_detector.py")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r010_analysis")
    parser.add_argument("--python", default="python")
    return parser.parse_args()


def merge_baseline(data_dir: Path, baseline: str, output: Path) -> Path:
    files = collect_files([data_dir], [baseline])
    parts: dict[str, list[np.ndarray]] = {}
    lags = None
    for path in files:
        archive = np.load(path, allow_pickle=False)
        if lags is None:
            lags = np.asarray(archive["lags_ms"], dtype=np.float64)
        else:
            if not np.array_equal(lags, np.asarray(archive["lags_ms"], dtype=np.float64)):
                raise ValueError(f"{path}: lags_ms mismatch")
        for key in archive.files:
            values = np.asarray(archive[key])
            if values.ndim > 0 and key != "lags_ms":
                parts.setdefault(key, []).append(values)
    payload = {"lags_ms": lags}
    for key, arrays in parts.items():
        payload[key] = np.concatenate(arrays, axis=0)
    # 检测器只读 "curves" 键；与主审计同口径：prob 分数下的正值交互（synergy）
    payload["curves"] = payload["curve_prob_positive"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    return output


def run_detector(args: argparse.Namespace, curves: Path, output: Path) -> dict:
    subprocess.run(
        [args.python, "-u", str(args.detector_script), "--curves", str(curves),
         "--null-curves", str(args.null_curves), "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    anchor_mass = None

    for baseline in BASELINES:
        merged = args.output_dir / f"merged_{baseline}.npz"
        merge_baseline(args.data_dir, baseline, merged)
        detector_json = args.output_dir / f"{baseline}_detector.json"
        report = run_detector(args, merged, detector_json)
        summary = report["summary"]

        archive = np.load(merged, allow_pickle=False)
        row: dict = {
            "baseline": baseline,
            "n_records": int(summary["n_curves"]),
            "support_rate": float(summary["support_rate"]),
            "no_support_fraction": float(summary["no_support_fraction"]),
        }
        mass_vectors = {}
        for score in SCORE_MODES:
            for mode in CURVE_MODES:
                curves = np.asarray(archive[f"curve_{score}_{mode}"], dtype=np.float64)
                positive_mass = np.maximum(curves, 0).sum(axis=1)
                row[f"{score}_{mode}_mean_positive_mass"] = float(positive_mass.mean())
                row[f"{score}_{mode}_mean_signed_sum"] = float(curves.sum(axis=1).mean())
                row[f"{score}_{mode}_mean_abs_sum"] = float(np.abs(curves).sum(axis=1).mean())
                row[f"{score}_{mode}_mean_peak"] = float(np.abs(curves).max(axis=1).mean())
                if score == "prob" and mode == "positive":
                    mass_vectors[baseline] = positive_mass
                    if baseline == "rest_mean":
                        anchor_mass = positive_mass
        if anchor_mass is not None and baseline != "rest_mean":
            rho, _ = stats.spearmanr(anchor_mass, mass_vectors[baseline])
            row["spearman_positive_mass_vs_rest_mean"] = float(rho)
        else:
            row["spearman_positive_mass_vs_rest_mean"] = 1.0 if baseline == "rest_mean" else None
        rows.append(row)
        print(f"{baseline}: support={row['support_rate']:.3f}", flush=True)

    with open(args.output_dir / "baseline_comparison.csv", "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "baseline_comparison.json").write_text(
        json.dumps({"baselines": rows, "note": "阈值与 R008 相同（只由零模型标定）；支持率口径与主审计一致。"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output_dir / 'baseline_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
