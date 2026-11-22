"""mPower adapter on FAKE manifests (synthetic identifiers): sealed test never opened."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import cohort as stage0
from pipeline.lib.config import load_config

REPO = Path(__file__).resolve().parents[2]


def fake_manifest(rng, split, n_people, offset):
    rows = []
    for i in range(n_people):
        pid = f"fake-{offset + i:04d}"
        pd_ = i % 2 == 0
        for v in range(2):
            rows.append({"recording_id": f"{pid}-r{v}", "participant_id": pid,
                         "diagnosis": "pd" if pd_ else "control",
                         "age": float(rng.integers(45, 80)), "sex": ["Male", "Female"][i % 3 == 0],
                         "age_invalid": False, "medication": "Another time" if pd_ else None,
                         "task": "sustained_phonation", "recorded_at": f"2015-0{v + 3}-1{i % 9}T10:00:00Z",
                         "duplicate_component": pid, "split": split, "decode_status": "ok",
                         "duration_s": 9.8, "pitch_detected_s": 6.0 if v == 0 else 0.2,
                         "audio_path": f"/nonexistent/{pid}-{v}.m4a"})
    return pd.DataFrame(rows)


def test_mpower_reads_only_development_manifests(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    source = tmp_path / "phase1/processed/cohort_A"
    source.mkdir(parents=True)
    fake_manifest(rng, "train", 40, 0).to_csv(source / "train_manifest.csv", index=False)
    fake_manifest(rng, "val", 10, 100).to_csv(source / "val_manifest.csv", index=False)
    fake_manifest(rng, "test", 10, 200).to_csv(source / "test_manifest.csv", index=False)
    monkeypatch.setenv("MPOWER_OUTPUT_ROOT", str(tmp_path / "phase1"))
    monkeypatch.setenv("MPOWER_BENCHMARK_ROOT", str(tmp_path / "bench"))

    opened = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p, *a, **k: (opened.append(str(p)), real(p, *a, **k))[1])
    cfg = load_config(REPO / "configs/pipeline/mpower_vowel.yaml")
    cache = stage0.run(cfg, log=lambda *_: None)
    monkeypatch.setattr(pd, "read_csv", real)

    assert not any("test_manifest" in p for p in opened)
    cohort = pd.read_csv(cache.path("cohort.csv"))
    assert not cohort.participant_id.str.startswith("fake-02").any()      # test ids 200+
    # Precomputed QC: second visit of everyone has 0.2 s voicing -> excluded.
    assert cohort.recording_id.str.endswith("-r0").all()
    excl = pd.read_csv(cache.path("exclusions.csv"))
    assert set(excl.criterion) == {"no_voiced_audio"}


def test_real_data_outputs_must_stay_outside_repo(monkeypatch):
    monkeypatch.setenv("MPOWER_BENCHMARK_ROOT", str(REPO / "results"))
    with pytest.raises(ValueError, match="outside the Git repository"):
        load_config(REPO / "configs/pipeline/mpower_vowel.yaml")
