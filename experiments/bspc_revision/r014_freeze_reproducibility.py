"""Freeze the completed revision experiments into a checksummed local package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_FILES = {
    "r011_summary": "results/bspc_revision_v2/r011_full_comparison_20260829/summary.json",
    "r011_report": "results/bspc_revision_v2/r011_full_comparison_20260829/r011_report.json",
    "r011_strata": "results/bspc_revision_v2/r011_full_comparison_20260829/r011_aopc_stratified.csv",
    "r012_training": "results/bspc_revision_v2/r012_resnet1d_full_20260829/protocol_summary.json",
    "r012_onset_protocol": "results/bspc_revision_v2/r012_resnet1d_onset_protocol_20260829.json",
    "r012_detector": "results/bspc_revision_v2/r012_resnet1d_detector_report_20260829.json",
    "r012_strata": "results/bspc_revision_v2/r012_resnet1d_stratified_report_20260829.json",
    "r012_label_shuffle": "results/bspc_revision_v2/r012_resnet1d_label_shuffle_20260829/protocol_summary.json",
    "r012_label_detector": "results/bspc_revision_v2/r012_resnet1d_label_shuffle_detector_report_20260829.json",
    "r013_training": "results/bspc_revision_v2/r013_e2_lstm_full_20260829/protocol_summary.json",
    "r013_onset_protocol": "results/bspc_revision_v2/r013_e2_onset_protocol_20260829.json",
    "r013_detector": "results/bspc_revision_v2/r013_e2_detector_report_20260829.json",
    "r013_strata": "results/bspc_revision_v2/r013_e2_stratified_report_20260829.json",
    "r013_label_shuffle": "results/bspc_revision_v2/r013_e2_label_shuffle_20260829/protocol_summary.json",
    "r013_label_detector": "results/bspc_revision_v2/r013_e2_label_shuffle_detector_report_20260829.json",
    "r018_summary": "results/bspc_revision_v2/r018_comparison_methods/summary.json",
    "r016_summary": "results/bspc_revision_v2/a1_synthetic_detector_validation/summary.json",
    "r017_summary": "results/bspc_revision_v2/r017_db2_fixed_replay/summary.json",
    "r019_summary": "results/bspc_revision_v2/r019_falsifier_sensitivity/summary.json",
}

CHECKPOINT_DIRS = {
    "r005b_main_lstm": "checkpoints_bspc_v2/r005b_balanced_full_seed20260815",
    "r012_resnet1d": "checkpoints_bspc_v2/r012_resnet1d_full_20260829",
    "r012_resnet1d_label_shuffle": "checkpoints_bspc_v2/r012_resnet1d_label_shuffle_20260829",
    "r013_e2_lstm": "checkpoints_bspc_v2/r013_e2_lstm_full_20260829",
    "r013_e2_label_shuffle": "checkpoints_bspc_v2/r013_e2_label_shuffle_20260829",
}

SOURCE_FILES = (
    "experiments/bspc_revision/leakage_free_db2.py",
    "experiments/bspc_revision/train_label_shuffle.py",
    "experiments/bspc_revision/collect_st_sri_curves.py",
    "experiments/bspc_revision/baseline_representation_scan.py",
    "experiments/bspc_revision/comparison_methods_scan.py",
    "experiments/bspc_revision/r011_comparison_report.py",
    "experiments/bspc_revision/r014_freeze_reproducibility.py",
    "experiments/bspc_revision/a1_synthetic_detector_validation.py",
    "experiments/bspc_revision/audit_onset_protocol.py",
    "experiments/bspc_revision/merge_curve_npz.py",
    "experiments/bspc_revision/r017_db2_fixed_replay.py",
    "experiments/bspc_revision/r019_falsifier_sensitivity.py",
    "experiments/bspc_revision/r010_baseline_repr_analysis.py",
    "experiments/bspc_revision/st_sri_detector.py",
    "experiments/bspc_revision/stratified_st_sri_report.py",
    "requirements.txt",
)

LAUNCHER_FILES = (
    "tmp/run_r012_full_20260829.ps1",
    "tmp/run_r011_full_20260829.ps1",
    "tmp/run_r013_e2_full_20260829.ps1",
    "tmp/post_r012_queue_20260829.ps1",
    "tmp/post_r013_audit_queue_20260829.ps1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bspc_revision_v2" / "r014_reproducibility_20260829",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_completed_run(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") != "completed":
        raise ValueError(f"{path}: state={payload.get('state')!r}, expected 'completed'")
    if len(payload.get("training_results", [])) != 40:
        raise ValueError(f"{path}: expected 40 training results")
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.output_dir / "reports"
    code_dir = args.output_dir / "code"
    launcher_dir = args.output_dir / "launchers"
    report_dir.mkdir(exist_ok=True)
    code_dir.mkdir(exist_ok=True)
    launcher_dir.mkdir(exist_ok=True)

    required_paths = {name: PROJECT_ROOT / relative for name, relative in RESULT_FILES.items()}
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("required results are missing: " + "; ".join(missing))
    load_completed_run(required_paths["r012_training"])
    load_completed_run(required_paths["r012_label_shuffle"])
    load_completed_run(required_paths["r013_training"])
    load_completed_run(required_paths["r013_label_shuffle"])

    artifact_manifest: dict[str, dict[str, object]] = {}
    for name, path in required_paths.items():
        destination = report_dir / f"{name}{path.suffix}"
        shutil.copy2(path, destination)
        artifact_manifest[name] = {
            "source": str(path.relative_to(PROJECT_ROOT)),
            "package_copy": str(destination.relative_to(args.output_dir)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    checkpoint_manifest: dict[str, list[dict[str, object]]] = {}
    for name, relative in CHECKPOINT_DIRS.items():
        directory = PROJECT_ROOT / relative
        checkpoints = sorted(directory.glob("S??_best.pth"))
        if len(checkpoints) != 40:
            raise ValueError(f"{directory}: expected 40 checkpoints, found {len(checkpoints)}")
        checkpoint_manifest[name] = [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in checkpoints
        ]

    source_manifest: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = PROJECT_ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        destination = code_dir / source.name
        shutil.copy2(source, destination)
        source_manifest[relative] = sha256_file(source)
    for relative in LAUNCHER_FILES:
        source = PROJECT_ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, launcher_dir / source.name)

    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deviation_from_project_pin": "Runtime uses torch 2.7.1+cu128 because the project pin 2.5.1+cu124 cannot execute on RTX 5090 D (sm_120).",
    }
    payload = {
        "status": "completed_local_freeze",
        "scope": "R005b/R011/R012/R013/R016/R017/R018/R019 revision evidence and runnable code",
        "environment": environment,
        "result_artifacts": artifact_manifest,
        "checkpoint_artifacts": checkpoint_manifest,
        "source_sha256": source_manifest,
        "external_assets_not_included": [
            {
                "asset": "R015b 3 x 40 remote checkpoints",
                "reason": "not present in the local workspace; R015b attribution is non-main evidence and is not used by this freeze",
                "action_required": "retrieve the remote checkpoint directories before a public archive that claims R015b replay coverage",
            },
            {
                "asset": "DOI archive deposit",
                "reason": "requires a repository account and explicit external publication authorization",
                "action_required": "upload this local package to the selected archive after review",
            },
        ],
    }
    manifest_path = args.output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "REPRODUCTION.md").write_text(
        "# Revision Experiment Reproduction\n\n"
        "Run from `03_bspc/05_revision_experiments` with the recorded Python environment.\n\n"
        "1. Verify `MANIFEST.json` SHA-256 values for checkpoints and result reports.\n"
        "2. Use the copied launcher scripts in `launchers/` for the full training and audit sequence.\n"
        "3. The raw DB2 and E2 datasets are referenced in the run configuration and are intentionally not copied.\n"
        "4. R015b checkpoints and DOI deposition are external follow-up items listed in the manifest.\n",
        encoding="utf-8",
    )
    print(f"completed output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
