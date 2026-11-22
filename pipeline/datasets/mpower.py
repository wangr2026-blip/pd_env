"""mPower sustained-phonation recordings from the existing processed manifests.

Reads ONLY ``$MPOWER_OUTPUT_ROOT/processed/cohort_<X>/{train,val}_manifest.csv``,
the participant/duplicate-component-linked development splits produced by
``scripts/create_participant_splits.py``. The test manifest and the
participant-wide label tables are never opened, so sealed-test labels cannot
reach any stage.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.datasets.base import SAMPLE_RATE, Dataset, validate_waveform

RENAME = {"split": "original_split", "pitch_detected_s": "voiced_s"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class MPowerDataset(Dataset):
    name = "mpower"

    def __init__(self, cfg, seed=None):
        super().__init__(cfg)
        env = cfg.get("output_root_env", "MPOWER_OUTPUT_ROOT")
        if not os.environ.get(env):
            raise ValueError(f"Set {env} to the approved mpower_phase1 directory")
        cohort = cfg.get("source_cohort", "A")
        self.source = Path(os.environ[env]).expanduser().resolve() / "processed" / f"cohort_{cohort}"
        self.files = [self.source / f"{s}_manifest.csv" for s in ("train", "val")]
        for f in self.files:
            if not f.exists():
                raise FileNotFoundError(f"Development manifest missing: {f}")
        self.max_decode_seconds = cfg.get("max_decode_seconds", 180)

    def fingerprint(self):
        return {"name": self.name, "source": str(self.source),
                "manifests": {f.name: _sha256(f) for f in self.files}}

    def recordings(self):
        frame = pd.concat([pd.read_csv(f, dtype={"participant_id": str, "recording_id": str},
                                       low_memory=False) for f in self.files], ignore_index=True)
        frame = frame.rename(columns={k: v for k, v in RENAME.items() if k in frame})
        if "duplicate_component" not in frame:
            frame["duplicate_component"] = frame.participant_id
        frame["recorded_at"] = pd.to_datetime(frame.get("recorded_at"), errors="coerce",
                                              utc=True, format="mixed")
        frame["age"] = pd.to_numeric(frame.get("age"), errors="coerce")
        if "age_invalid" in frame:
            frame.loc[frame.age_invalid.astype("boolean").fillna(False).astype(bool), "age"] = np.nan
        frame["sex"] = frame.get("sex").where(frame.get("sex").isin(["Male", "Female"]))
        if "decode_status" in frame:
            frame["decode_ok"] = frame.decode_status.eq("ok")
        for col in ("medication", "duration_s", "voiced_s"):
            if col not in frame:
                frame[col] = np.nan
        return self.check(frame)

    def load_audio(self, row):
        # Existing repository decoder: FFmpeg -> mono 16 kHz float32, bounded duration.
        from mpower_common import decode
        wave, sr = decode(row["audio_path"], {"sample_rate": SAMPLE_RATE,
                                              "max_decode_seconds": self.max_decode_seconds})
        if sr != SAMPLE_RATE:
            raise ValueError("Decoder returned unexpected sample rate")
        return validate_waveform(wave)
