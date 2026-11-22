"""Stage 0 (cohort, QC, folds, matched subset) on synthetic data. No participant data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import cohort as stage0
from pipeline.datasets import get_dataset
from pipeline.lib.config import load_config
from pipeline.lib.folds import aggregate_participants, assert_development_only

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/pipeline/smoke_synthetic.yaml"
SMALL = {"dataset": {"params": {"n_participants": 70, "duration_s": [2.5, 3.0],
                                "bad_audio_fraction": 0.08}}}


def config(root, extra=None):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PIPELINE_OUTPUT_ROOT", str(root))
        from pipeline.lib.config import _merge
        return load_config(CONFIG, _merge(SMALL, extra or {}))


def read(cache, name):
    return pd.read_csv(cache.path(name), dtype={"participant_id": str, "recording_id": str})


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cfg = config(tmp_path_factory.mktemp("stage0"))
    return cfg, stage0.run(cfg, log=lambda *_: None)


def test_sealed_test_rows_never_enter(built):
    cfg, cache = built
    every = get_dataset(cfg).all_recordings()
    sealed = set(every.recording_id[every.original_split.eq("test")])
    assert sealed, "fixture must contain sealed rows to be meaningful"
    for name in ("cohort.csv", "qc.csv", "exclusions.csv"):
        assert not sealed & set(read(cache, name).recording_id)
    assert set(read(cache, "cohort.csv").original_split) <= {"train", "val"}
    assert cache.manifest()["summary"]["test_rows_read"] is False


def test_guard_rejects_sealed_rows():
    with pytest.raises(ValueError):
        assert_development_only(pd.DataFrame({"original_split": ["train", "test"]}))
    with pytest.raises(ValueError):
        assert_development_only(pd.DataFrame({"original_split": ["train", None]}))


def test_folds_are_participant_and_group_disjoint(built):
    cfg, cache = built
    cohort, folds = read(cache, "cohort.csv"), read(cache, "folds.csv")
    assert not folds.participant_id.duplicated().any()
    assert cohort.groupby("participant_id").fold.nunique().max() == 1
    assert cohort.groupby("duplicate_component").fold.nunique().max() == 1
    assert sorted(folds.fold.unique()) == list(range(cfg["folds"]["n_splits"]))
    assert folds.groupby("fold").label.nunique().min() == 2
    linked = cohort.groupby("duplicate_component").participant_id.nunique()
    assert linked.gt(1).any(), "fixture must contain duplicate-linked participants"


def test_each_failure_mode_is_excluded_for_the_right_reason(built):
    cfg, cache = built
    every = get_dataset(cfg).recordings()
    excl = read(cache, "exclusions.csv")
    reasons = excl.groupby("recording_id").criterion.agg(set)
    kind = every.set_index("recording_id")
    short = kind.index[kind.duration_target_s.lt(2) & kind.diagnosis.isin(["pd", "control"])]
    silent = kind.index[kind.audio_kind.eq("silent") & kind.diagnosis.isin(["pd", "control"])]
    unvoiced = kind.index[kind.audio_kind.eq("unvoiced") & kind.diagnosis.isin(["pd", "control"])]
    assert len(short) and len(silent) and len(unvoiced), "fixture must contain every bad-audio type"
    assert all("too_short" in reasons[r] for r in short)
    assert all("no_voiced_audio" in reasons[r] for r in silent.union(unvoiced))
    assert all("missing_label" in reasons[r] for r in kind.index[kind.diagnosis.eq("unknown")])
    assert all("missing_age" in reasons[r] for r in kind.index[kind.age.isna()])
    cohort = read(cache, "cohort.csv")
    assert not set(cohort.recording_id) & set(excl.recording_id)
    assert cohort.duration_s.min() >= cfg["cohort"]["min_duration_s"]


def test_cache_hit_and_determinism(built, tmp_path):
    cfg, cache = built
    before = cache.manifest_path.stat().st_mtime_ns
    again = stage0.run(cfg, log=lambda *_: None)
    assert again.dir == cache.dir and cache.manifest_path.stat().st_mtime_ns == before
    fresh = stage0.run(config(tmp_path), log=lambda *_: None)
    assert fresh.key == cache.key
    pd.testing.assert_frame_equal(read(fresh, "folds.csv"), read(cache, "folds.csv"))
    pd.testing.assert_frame_equal(read(fresh, "cohort.csv"), read(cache, "cohort.csv"))


def test_config_change_changes_key(built, tmp_path):
    _, cache = built
    other = config(tmp_path, {"seed": 7})
    from pipeline.lib.config import stage_key
    key, _ = stage_key(other, stage0.SECTIONS, code=stage0.CODE,
                       extra={"dataset": get_dataset(other).fingerprint()})
    assert key != cache.key


def test_matched_subset_is_balanced_within_strata(built):
    cfg, cache = built
    folds = read(cache, "folds.csv")
    m = folds[folds.in_matched]
    assert len(m) > 0
    lo, hi = cfg["matched"]["age_range"]
    assert m.age.between(lo, hi).all()
    w = cfg["matched"]["age_bin_width"]
    m = m.assign(bin=lo + ((m.age - lo) // w) * w)
    counts = m.groupby(["sex", "bin", "label"]).size().unstack("label", fill_value=0)
    assert (counts[0] == counts[1]).all()
    flow = read(cache, "matching_flow.csv")
    assert (flow.retained <= flow.available).all()


def test_initial_recording_is_earliest_per_participant(built):
    _, cache = built
    cohort = read(cache, "cohort.csv")
    cohort["recorded_at"] = pd.to_datetime(cohort.recorded_at)
    assert cohort.groupby("participant_id").initial_recording.sum().eq(1).all()
    first = cohort.sort_values(["participant_id", "recorded_at"]).drop_duplicates("participant_id")
    assert first.initial_recording.all()


def test_summary_tables(built):
    _, cache = built
    summary = read(cache, "cohort_summary.csv")
    assert set(summary.cohort) == {"raw", "matched"} and set(summary.group) == {"PD", "control"}
    folds = read(cache, "folds.csv")
    raw = summary[summary.cohort.eq("raw")].set_index("group")
    assert raw.loc["PD", "participants"] == folds.label.eq(1).sum()
    flow = read(cache, "exclusion_flow.csv")
    assert flow.groupby("diagnosis").remaining_recordings.apply(lambda s: s.is_monotonic_decreasing).all()


def test_medication_filter(tmp_path):
    keep = ["Another time"]
    cfg = config(tmp_path, {"cohort": {"medication_filter": {"pd": keep}}})
    cache = stage0.run(cfg, log=lambda *_: None)
    cohort = read(cache, "cohort.csv")
    assert set(cohort.medication[cohort.label.eq(1)]) == set(keep)
    assert cohort.label.eq(0).any()
    assert "medication_pd" in set(read(cache, "exclusions.csv").criterion)


def test_aggregate_participants():
    preds = pd.DataFrame({"participant_id": ["a", "a", "b"], "fold": [0, 0, 1],
                          "label": [1, 1, 0], "probability": [.2, .6, .3]})
    out = aggregate_participants(preds, "mean").set_index("participant_id")
    assert out.loc["a", "probability"] == pytest.approx(.4) and out.loc["a", "recordings"] == 2
    with pytest.raises(ValueError):
        aggregate_participants(preds.assign(label=[1, 0, 0]))


def test_synthetic_acoustics_are_recoverable(built):
    """Planted F0/jitter/HNR must be measurable, or later probe tests are meaningless."""
    parselmouth = pytest.importorskip("parselmouth")
    from parselmouth.praat import call
    from scipy.stats import spearmanr
    cfg, _ = built
    ds = get_dataset(cfg)
    rows = ds.recordings()
    rows = rows[rows.audio_kind.eq("voiced") & rows.duration_target_s.gt(2)].head(30)
    measured = []
    for _, r in rows.iterrows():
        s = parselmouth.Sound(ds.load_audio(r).astype(float), 16000)
        f0 = s.to_pitch_ac(0.01, 75, 500).selected_array["frequency"]
        pp = call(s, "To PointProcess (periodic, cc)", 75, 500)
        measured.append({"f0": f0[f0 > 0].mean(),
                         "jitter": call(pp, "Get jitter (local)", 0, 0, 1 / 500, 1 / 75, 1.3),
                         "hnr": call(s.to_harmonicity_cc(0.01, 75, 0.1, 1.0), "Get mean", 0, 0)})
    m = pd.DataFrame(measured, index=rows.index)
    assert spearmanr(rows.true_f0_hz, m.f0)[0] > .9
    assert spearmanr(rows.true_jitter_local, m.jitter)[0] > .6
    assert spearmanr(rows.true_hnr_db, m.hnr)[0] > .7
    np.testing.assert_array_equal(ds.load_audio(rows.iloc[0]), ds.load_audio(rows.iloc[0]))
