"""基于 NinaPro `restimulus` 标签起点的 ST-SRI 对齐协议。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.bspc_revision.leakage_free_db2 import TrialRecord


@dataclass(frozen=True)
class OnsetAuditRecord:
    subject: int
    split: str
    active_class: int
    repetition_index: int
    segment_index: int
    phase_ms: float
    phase_samples: int
    restimulus_onset_sample: int
    current_endpoint_sample: int
    current_block_start_sample: int
    current_block_end_sample: int
    input_start_sample: int
    input_end_sample: int
    restimulus_onset_in_window: int
    current_endpoint_in_window: int
    current_block_start_in_window: int
    current_block_end_in_window: int


def milliseconds_to_samples(milliseconds: float, fs: int) -> int:
    samples = milliseconds * fs / 1000.0
    rounded = int(round(samples))
    if not np.isclose(samples, rounded):
        raise ValueError(f"{milliseconds} ms is not an integer number of samples at {fs} Hz")
    return rounded


def build_onset_audit_records(
    trials: Iterable[TrialRecord],
    labels: np.ndarray,
    split: str,
    phases_ms: Sequence[float],
    fs: int = 2000,
    window_samples: int = 600,
    block_samples: int = 10,
) -> list[OnsetAuditRecord]:
    if window_samples < block_samples:
        raise ValueError("window_samples must be at least block_samples")
    if block_samples < 1:
        raise ValueError("block_samples must be positive")

    phase_pairs = [(float(phase_ms), milliseconds_to_samples(float(phase_ms), fs)) for phase_ms in phases_ms]
    if any(phase_samples < 0 for _, phase_samples in phase_pairs):
        raise ValueError("onset phases must be nonnegative")

    records = []
    for trial in trials:
        if trial.split != split:
            continue
        onset = trial.active_start
        if onset < 1 or labels[onset - 1] != 0 or labels[onset] != trial.active_class:
            raise ValueError(
                f"S{trial.subject}: active_start={onset} is not a restimulus "
                f"0->{trial.active_class} transition"
            )

        for phase_ms, phase_samples in phase_pairs:
            current_endpoint = onset + phase_samples
            current_block_start = current_endpoint - block_samples + 1
            input_end = current_endpoint + 1
            input_start = input_end - window_samples
            if input_start < trial.raw_start or input_end > trial.raw_end:
                raise ValueError(
                    f"S{trial.subject} class {trial.active_class} repetition "
                    f"{trial.repetition_index}: phase {phase_ms:g} ms window "
                    f"[{input_start}, {input_end}) exceeds trial support "
                    f"[{trial.raw_start}, {trial.raw_end})"
                )
            if current_endpoint >= trial.active_end:
                raise ValueError(
                    f"S{trial.subject} class {trial.active_class}: phase {phase_ms:g} ms "
                    "extends beyond the refined active segment"
                )

            record = OnsetAuditRecord(
                subject=trial.subject,
                split=trial.split,
                active_class=trial.active_class,
                repetition_index=trial.repetition_index,
                segment_index=trial.segment_index,
                phase_ms=phase_ms,
                phase_samples=phase_samples,
                restimulus_onset_sample=onset,
                current_endpoint_sample=current_endpoint,
                current_block_start_sample=current_block_start,
                current_block_end_sample=current_endpoint + 1,
                input_start_sample=input_start,
                input_end_sample=input_end,
                restimulus_onset_in_window=onset - input_start,
                current_endpoint_in_window=current_endpoint - input_start,
                current_block_start_in_window=current_block_start - input_start,
                current_block_end_in_window=current_endpoint + 1 - input_start,
            )
            verify_onset_audit_record(record, window_samples, block_samples)
            records.append(record)
    return records


def verify_onset_audit_record(
    record: OnsetAuditRecord,
    window_samples: int,
    block_samples: int,
) -> None:
    if record.input_end_sample - record.input_start_sample != window_samples:
        raise AssertionError("onset-aligned input has the wrong length")
    if record.current_endpoint_in_window != window_samples - 1:
        raise AssertionError("causal input must end at the declared current endpoint")
    if record.current_block_end_sample - record.current_block_start_sample != block_samples:
        raise AssertionError("current block has the wrong absolute width")
    if record.current_block_end_in_window - record.current_block_start_in_window != block_samples:
        raise AssertionError("current block has the wrong relative width")
    if record.current_block_end_in_window != window_samples:
        raise AssertionError("current block must end with the causal input window")
    if record.current_endpoint_sample - record.restimulus_onset_sample != record.phase_samples:
        raise AssertionError("current endpoint is not at the declared onset phase")
    expected_onset_position = window_samples - record.phase_samples - 1
    if record.restimulus_onset_in_window != expected_onset_position:
        raise AssertionError("stored restimulus onset position is inconsistent")


class OnsetAuditDataset(Dataset):
    def __init__(
        self,
        data_path,
        records: Sequence[OnsetAuditRecord],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.data_path = data_path
        self.records = list(records)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self._data: np.ndarray | None = None

    def _get_data(self) -> np.ndarray:
        if self._data is None:
            self._data = np.load(self.data_path, mmap_mode="r")
        return self._data

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        raw = np.asarray(
            self._get_data()[record.input_start_sample : record.input_end_sample],
            dtype=np.float32,
        )
        normalized = (raw - self.mean) / self.std
        return torch.from_numpy(normalized), torch.tensor(record.active_class, dtype=torch.long)
