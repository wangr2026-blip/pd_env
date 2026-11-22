"""Dataset interface. Language and task are attributes, so new corpora reuse every stage."""
from abc import ABC, abstractmethod

import numpy as np

# Columns every dataset must provide from ``recordings()``. Missing values are NaN.
STANDARD_COLUMNS = {
    "recording_id": "unique recording key",
    "participant_id": "participant key; folds are participant-disjoint",
    "diagnosis": "'pd', 'control' or other (other = unlabeled, excluded)",
    "age": "years at baseline, float",
    "sex": "'Male' / 'Female' / NaN",
    "medication": "medication timepoint response (source wording) or NaN",
    "task": "recording task, e.g. 'sustained_phonation'",
    "recorded_at": "timestamp used to pick each participant's initial recording",
    "duplicate_component": "group of participants linked by duplicate audio (else participant_id)",
    "original_split": "source development split; only 'train'/'val' may ever appear",
}
# Optional precomputed QC columns (used when cohort.qc_source == 'precomputed').
PRECOMPUTED_QC = ("decode_ok", "duration_s", "voiced_s")
SAMPLE_RATE = 16000


class Dataset(ABC):
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.language = cfg.get("language", "en")
        self.task = cfg.get("task", "sustained_vowel")

    @abstractmethod
    def recordings(self):
        """Development recordings only, with STANDARD_COLUMNS. Must never read the sealed test."""

    @abstractmethod
    def load_audio(self, row):
        """Return mono 16 kHz float32 audio for one recordings() row."""

    @abstractmethod
    def fingerprint(self):
        """JSON-serializable identity of the source data, for cache keys."""

    def check(self, frame):
        missing = [c for c in STANDARD_COLUMNS if c not in frame]
        if missing:
            raise ValueError(f"{self.name}: recordings() missing columns {missing}")
        if frame.recording_id.duplicated().any():
            raise ValueError(f"{self.name}: duplicate recording_id")
        return frame


def validate_waveform(wave):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1:
        raise ValueError("Audio must be mono")
    if not np.isfinite(wave).all():
        raise ValueError("nonfinite_audio")
    return wave
