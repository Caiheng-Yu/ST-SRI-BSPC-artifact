"""R009：ST-SRI 分层统计报告（纯 CPU）。

把 `st_sri_detector.py` 的逐曲线判决与 `collect_st_sri_curves.py` 的
曲线元数据按 id 连接，按计划 B3 分层报告：

- 窗口类型 × 相位（onset/offset/steady）；
- 真实类别、预测类别；
- 正确 / 错误决策；
- 目标概率四分位（置信度分层）。

每层报告样本数、支持率、连续正质量和峰值显著度中位数，
避免把异质窗口混合成单一受试者级谱。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STRATIFY_DIMS = ("window_kind", "phase", "active_class", "predicted_class", "correct", "target_prob_quartile")


def load_decisions(report_path: Path) -> dict[str, dict]:
    """读取检测器报告，返回 id -> 判决字典。"""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    decisions = {}
    for entry in report["decisions"]:
        if entry["id"] in decisions:
            raise ValueError(f"duplicate decision id {entry['id']} in {report_path}")
        decisions[entry["id"]] = entry
    if not decisions:
        raise ValueError(f"no decisions in {report_path}")
    return decisions


def load_curve_metadata(npz_paths: list[Path]) -> list[dict]:
    """读取曲线 npz 的元数据数组，每行一条记录。"""
    rows = []
    for path in npz_paths:
        archive = np.load(path, allow_pickle=False)
        required = ("ids", "window_kind", "phase_ms", "active_class", "predicted_class", "correct", "target_prob")
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"{path} missing arrays: {missing}")
        meta = json.loads(str(archive["meta_json"])) if "meta_json" in archive else {}
        subject = meta.get("subject")
        for index, curve_id in enumerate(archive["ids"]):
            rows.append(
                {
                    "id": str(curve_id),
                    "subject": int(subject) if subject is not None else None,
                    "window_kind": str(archive["window_kind"][index]),
                    "phase": f"{float(archive['phase_ms'][index]):g}",
                    "active_class": int(archive["active_class"][index]),
                    "predicted_class": int(archive["predicted_class"][index]),
                    "correct": int(archive["correct"][index]),
                    "target_prob": float(archive["target_prob"][index]),
                    "source_npz": path.name,
                }
            )
    if not rows:
        raise ValueError("no curve metadata rows loaded")
    return rows


def assign_quartile_bins(values: np.ndarray) -> list[str]:
    """按分位数把连续值分到 Q1..Qk；取值不足 4 个时自动减少分箱。"""
    values = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
    indices = np.digitize(values, edges, right=True)
    return [f"Q{index + 1}" for index in indices]


def stratify(rows: list[dict], decisions: dict[str, dict]) -> dict:
    """按各分层维度汇总支持率和连续质量统计量。"""
    row_ids = {row["id"] for row in rows}
    missing = row_ids - set(decisions)
    extra = set(decisions) - row_ids
    if missing or extra:
        raise ValueError(f"decision/metadata id mismatch: missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}")

    quartiles = assign_quartile_bins(np.array([row["target_prob"] for row in rows]))
    for row, quartile in zip(rows, quartiles):
        row["target_prob_quartile"] = quartile

    report: dict[str, dict[str, dict]] = {}
    for dim in STRATIFY_DIMS:
        levels: dict[str, list[dict]] = {}
        for row in rows:
            levels.setdefault(str(row[dim]), []).append(row)
        dim_report = {}
        for level, level_rows in sorted(levels.items()):
            level_decisions = [decisions[row["id"]] for row in level_rows]
            supported = [entry for entry in level_decisions if entry["supported"]]
            supported_lags = [
                entry["statistics"]["peak_lag_ms"] for entry in supported if entry["statistics"].get("peak_lag_ms") is not None
            ]
            dim_report[level] = {
                "n": len(level_rows),
                "n_supported": len(supported),
                "support_rate": len(supported) / len(level_rows),
                "positive_mass_median": float(np.median([entry["statistics"]["positive_mass"] for entry in level_decisions])),
                "peak_height_median": float(np.median([entry["statistics"]["peak_height"] for entry in level_decisions])),
                "supported_peak_lag_ms_median": float(np.median(supported_lags)) if supported_lags else None,
            }
        report[dim] = dim_report
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", nargs="+", type=Path, required=True, help="曲线 npz（可多个受试者/变体合并）")
    parser.add_argument("--report", type=Path, required=True, help="st_sri_detector.py 输出的判决 JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r009_stratified_report.json",
    )
    return parser.parse_args()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    rows = load_curve_metadata(args.curves)
    decisions = load_decisions(args.report)
    report = stratify(rows, decisions)
    payload = {
        "run_id": "R009",
        "stratify_dims": list(STRATIFY_DIMS),
        "n_curves": len(rows),
        "curves": [str(path) for path in args.curves],
        "detector_report": str(args.report),
        "strata": report,
    }
    write_json_atomic(args.output, payload)
    print(f"strata={list(report)} n={len(rows)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
