"""合并按受试者分片的曲线 npz 为单一聚合 npz（供 st_sri_detector.py / stratified_st_sri_report.py 使用）。

用法：
    python experiments/bspc_revision/merge_curve_npz.py \
        --input-dirs results/bspc_revision_v2/r006_curves results/bspc_revision_v2/r006_curves_shuffled \
        --variants trained reinit param_scramble \
        --output results/bspc_revision_v2/r006_merged/merged.npz

每个输入目录中按 `S*_{variant}.npz` 匹配；所有文件的 `lags_ms` 必须一致；
逐记录数组（ids、curves、interactions、active_class、correct 等）沿第 0 轴拼接；
`meta_json` 取第一个文件并记录来源文件列表。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True, help="文件名后缀，如 trained")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_files(input_dirs: list[Path], variants: list[str]) -> list[Path]:
    files: list[Path] = []
    for directory in input_dirs:
        for variant in variants:
            matches = sorted(directory.glob(f"S*_{variant}.npz"))
            if not matches:
                raise FileNotFoundError(f"no S*_{variant}.npz in {directory}")
            files.extend(matches)
    return files


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"skip existing {args.output}", flush=True)
        return

    files = collect_files(args.input_dirs, args.variants)
    parts: dict[str, list[np.ndarray]] = {}
    reference_lags = None
    meta = None
    for path in files:
        archive = np.load(path, allow_pickle=False)
        if reference_lags is None:
            reference_lags = np.asarray(archive["lags_ms"], dtype=np.float64)
            if "meta_json" in archive:
                meta_arr = archive["meta_json"]
                raw = meta_arr.item() if meta_arr.ndim == 0 else meta_arr[0]
                meta = json.loads(str(raw))
        else:
            if not np.array_equal(reference_lags, np.asarray(archive["lags_ms"], dtype=np.float64)):
                raise ValueError(f"{path}: lags_ms 与其他文件不一致")
        for key in archive.files:
            if key in ("lags_ms", "meta_json"):
                continue
            values = np.asarray(archive[key])
            if values.ndim == 0:
                continue
            parts.setdefault(key, []).append(values)

    payload: dict[str, np.ndarray] = {"lags_ms": reference_lags}
    for key, arrays in sorted(parts.items()):
        payload[key] = np.concatenate(arrays, axis=0)

    merged_meta = {
        "merge_source_files": [str(path) for path in files],
        "n_source_files": len(files),
        "n_records": int(payload["curves"].shape[0]),
        "first_file_meta": meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # meta_json 不写入聚合 npz（object 数组与 allow_pickle=False 不兼容），改存旁车 json
    sidecar = args.output.with_suffix(".merge_meta.json")
    sidecar.write_text(json.dumps(merged_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, args.output)
    print(
        f"merged {len(files)} files -> {args.output} "
        f"({payload['curves'].shape[0]} records x {payload['lags_ms'].size} lags)",
        flush=True,
    )


if __name__ == "__main__":
    main()
