"""Stage 0: cohort definition, audio QC, participant-level folds, matched subset.

    python -m pipeline.cohort --config configs/pipeline/smoke_synthetic.yaml [--force]

Outputs (under <output_root>/<run_name>/cohort/<key>/):
    cohort.csv           eligible recordings + fold, initial_recording, in_matched
    folds.csv            one row per participant: fold_group, label, fold, in_matched
    cohort_summary.csv   N / recordings / age / sex by cohort (raw, matched) x group
    fold_summary.csv     participants and recordings per fold x label x cohort
    exclusions.csv       every (recording, failed criterion) pair, overlapping
    exclusion_flow.csv   sequential flow: remaining recordings/participants by diagnosis
    qc.csv / qc_summary.csv   per-recording QC and duration/voicing distributions
    matching_flow.csv    available vs retained participants per matching stratum
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.datasets import get_dataset
from pipeline.datasets.base import PRECOMPUTED_QC, SAMPLE_RATE, STANDARD_COLUMNS
from pipeline.lib import REPO
from pipeline.lib.cache import StageCache, write_table
from pipeline.lib.config import load_config, stage_key
from pipeline.lib.folds import assert_development_only, assign_folds

STAGE = "cohort"
CODE = [REPO / "pipeline/cohort.py", REPO / "pipeline/lib", REPO / "pipeline/datasets"]
SECTIONS = ["seed", "dataset", "cohort", "folds", "matched"]
QC_COLUMNS = ["decode_ok", "decode_error", "duration_s", "voiced_s", "rms_dbfs", "peak"]
LABELS = {"pd": 1, "control": 0}


# --------------------------------------------------------------------------- QC
def audio_qc(wave, voicing):
    """Duration, level and Praat-detected voiced time. Flags only; audio is never trimmed."""
    import parselmouth
    sound = parselmouth.Sound(wave.astype(np.float64), sampling_frequency=SAMPLE_RATE)
    pitch = sound.to_pitch_ac(time_step=voicing["time_step_s"], pitch_floor=voicing["pitch_floor_hz"],
                              pitch_ceiling=voicing["pitch_ceiling_hz"])
    f0 = pitch.selected_array["frequency"]
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))) if len(wave) else 0.0
    return {"decode_ok": True, "decode_error": None, "duration_s": len(wave) / SAMPLE_RATE,
            "voiced_s": float((f0 > 0).sum() * voicing["time_step_s"]),
            "rms_dbfs": 20 * np.log10(max(rms, 1e-12)),
            "peak": float(np.abs(wave).max()) if len(wave) else 0.0}


_WORKER = {}


def _init_worker(dataset, voicing):
    _WORKER.update(dataset=dataset, voicing=voicing)


def _qc_one(row):
    try:
        wave = _WORKER["dataset"].load_audio(row)
        return {"recording_id": row["recording_id"], **audio_qc(wave, _WORKER["voicing"])}
    except Exception as exc:  # decode failures are recorded, never silently dropped
        return {"recording_id": row["recording_id"], "decode_ok": False,
                "decode_error": f"{type(exc).__name__}: {exc}"[:300]}


def compute_qc(dataset, frame, ccfg):
    if ccfg["qc_source"] == "precomputed":
        missing = [c for c in PRECOMPUTED_QC if c not in frame or frame[c].isna().all()]
        if missing:
            raise ValueError(f"qc_source=precomputed but dataset lacks {missing}; use qc_source=compute")
        qc = frame[["recording_id", *PRECOMPUTED_QC]].copy()
        qc["decode_ok"] = qc.decode_ok.astype("boolean").fillna(False).astype(bool)
        return qc.reindex(columns=["recording_id", *QC_COLUMNS])
    if ccfg["qc_source"] != "compute":
        raise ValueError("cohort.qc_source must be 'compute' or 'precomputed'")
    rows = frame.to_dict("records")
    workers = int(ccfg.get("workers", 1))
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker,
                                 initargs=(dataset, ccfg["voicing"])) as pool:
            out = list(pool.map(_qc_one, rows, chunksize=8))
    else:
        _init_worker(dataset, ccfg["voicing"])
        out = [_qc_one(r) for r in rows]
    return pd.DataFrame(out).reindex(columns=["recording_id", *QC_COLUMNS])


# ------------------------------------------------------------------- exclusions
def criteria(frame, ccfg):
    """Ordered exclusion criteria -> boolean masks (True = excluded)."""
    require = set(ccfg.get("require", ["label", "age", "sex"]))
    lo, hi = ccfg["age_range"]
    c = {"missing_label": frame.label.isna()}
    if "age" in require:
        c["missing_age"] = frame.age.isna()
    c["age_out_of_range"] = frame.age.notna() & ~frame.age.between(lo, hi)
    if "sex" in require:
        c["missing_sex"] = frame.sex.isna()
    if ccfg.get("task_filter"):
        c["wrong_task"] = ~frame.task.isin(ccfg["task_filter"])
    c["decode_failure"] = ~frame.decode_ok.astype(bool)
    c["too_short"] = frame.decode_ok & ~(frame.duration_s >= ccfg["min_duration_s"])
    c["no_voiced_audio"] = frame.decode_ok & ~(frame.voiced_s >= ccfg["min_voiced_s"])
    med = ccfg.get("medication_filter") or {}
    for group, label in (("pd", 1), ("control", 0)):
        allowed = med.get(group)
        if allowed is not None:
            c[f"medication_{group}"] = frame.label.eq(label) & ~frame.medication.isin(allowed)
    return c


def apply_exclusions(frame, ccfg):
    masks = criteria(frame, ccfg)
    keep = pd.Series(True, index=frame.index)
    exclusions, flow = [], []
    start = [{"criterion": "start", "diagnosis": d, "remaining_recordings": len(g),
              "remaining_participants": g.participant_id.nunique()} for d, g in frame.groupby("diagnosis")]
    for name, mask in masks.items():
        hit = frame.loc[mask, ["recording_id", "participant_id", "diagnosis"]].assign(criterion=name)
        exclusions.append(hit)
        keep &= ~mask
        for d, g in frame.groupby("diagnosis"):
            s = g[keep.loc[g.index]]
            flow.append({"criterion": name, "diagnosis": d, "remaining_recordings": len(s),
                         "remaining_participants": s.participant_id.nunique()})
    return keep, pd.concat(exclusions, ignore_index=True), pd.DataFrame(start + flow)


# ----------------------------------------------------------------- participants
def participant_table(frame):
    agg = frame.groupby("participant_id").agg(
        label=("label", "first"), label_n=("label", "nunique"), age=("age", "first"),
        age_n=("age", "nunique"), sex=("sex", "first"), sex_n=("sex", "nunique"),
        duplicate_component=("duplicate_component", "first"), recordings=("recording_id", "size"))
    for col in ("label", "age", "sex"):
        if agg[f"{col}_n"].gt(1).any():
            raise ValueError(f"Participants with conflicting {col} across recordings")
    return agg.drop(columns=["label_n", "age_n", "sex_n"]).reset_index()


def matched_subset(people, mcfg, seed):
    """Equal PD/control participant counts within (age bin x by-columns) strata.

    Same rule as scripts/build_cohorts.py cohort C, rebuilt inside the development
    set (the original cohort C includes sealed-test participants).
    """
    lo, hi = mcfg["age_range"]
    by = list(mcfg.get("by", ["sex"]))
    pool = people[people.age.between(lo, hi) & people[by].notna().all(axis=1)].copy()
    pool["age_bin_start"] = lo + ((pool.age - lo) // mcfg["age_bin_width"]) * mcfg["age_bin_width"]
    rng = np.random.default_rng([int(seed), 1])
    selected, flow = [], []
    for stratum, g in pool.groupby(by + ["age_bin_start"], sort=True):
        groups = {lab: g[g.label.eq(lab)].sort_values("participant_id") for lab in (1, 0)}
        count = min(len(v) for v in groups.values())
        for lab, grp in groups.items():
            if count:
                selected.extend(rng.choice(grp.participant_id.to_numpy(), count, replace=False))
            stratum = stratum if isinstance(stratum, tuple) else (stratum,)
            flow.append({**dict(zip(by + ["age_bin_start"], stratum)),
                         "label": lab, "available": len(grp), "retained": count})
    return set(selected), pd.DataFrame(flow)


def initial_recordings(frame):
    order = frame.sort_values(["participant_id", "recorded_at", "recording_id"], na_position="last")
    first = order.drop_duplicates("participant_id").recording_id
    return frame.recording_id.isin(set(first))


# -------------------------------------------------------------------- summaries
def cohort_summary(cohort):
    rows = []
    for name, sub in (("raw", cohort), ("matched", cohort[cohort.in_matched])):
        for group, g in sub.groupby("label"):
            people = g.drop_duplicates("participant_id")
            per = g.groupby("participant_id").size()
            rows.append({
                "cohort": name, "group": "PD" if group == 1 else "control",
                "participants": len(people), "recordings": len(g),
                "recordings_per_participant_median": float(per.median()),
                "age_mean": people.age.mean(), "age_sd": people.age.std(),
                "age_median": people.age.median(), "age_q1": people.age.quantile(.25),
                "age_q3": people.age.quantile(.75), "age_min": people.age.min(), "age_max": people.age.max(),
                "female_n": int(people.sex.eq("Female").sum()),
                "female_pct": 100 * people.sex.eq("Female").mean(),
                "duration_median_s": g.duration_s.median(), "voiced_median_s": g.voiced_s.median()})
    return pd.DataFrame(rows)


def fold_summary(cohort):
    rows = []
    for name, sub in (("raw", cohort), ("matched", cohort[cohort.in_matched])):
        for (fold, label), g in sub.groupby(["fold", "label"]):
            rows.append({"cohort": name, "fold": fold, "group": "PD" if label == 1 else "control",
                         "participants": g.participant_id.nunique(), "recordings": len(g),
                         "age_mean": g.drop_duplicates("participant_id").age.mean()})
    return pd.DataFrame(rows)


def qc_summary(frame):
    rows = []
    q = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    for group, g in [("all", frame), *frame.groupby("diagnosis")]:
        for col in ("duration_s", "voiced_s", "rms_dbfs"):
            v = pd.to_numeric(g[col], errors="coerce")
            rows.append({"group": group, "measure": col, "n": int(v.notna().sum()),
                         "missing": int(v.isna().sum()), "mean": v.mean(),
                         **{f"p{int(x * 100)}": v.quantile(x) for x in q}})
    return pd.DataFrame(rows)


def duration_figure(frame, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, col, title in zip(axes, ("duration_s", "voiced_s"), ("Decoded duration", "Praat voiced time")):
        for d, g in frame.groupby("diagnosis"):
            ax.hist(g[col].dropna(), bins=30, alpha=.55, label=str(d))
        ax.set(title=title, xlabel="seconds", ylabel="recordings")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ------------------------------------------------------------------------- run
def run(cfg, force=False, log=print):
    dataset = get_dataset(cfg)
    key, payload = stage_key(cfg, SECTIONS, code=CODE, extra={"dataset": dataset.fingerprint()})
    cache = StageCache(cfg, STAGE, key, payload)
    if cache.complete() and not force:
        log(f"[cohort] cache hit {cache.dir}")
        return cache
    cache.prepare(force)
    ccfg, fcfg, mcfg = cfg["cohort"], cfg["folds"], cfg["matched"]

    frame = dataset.recordings().sort_values("recording_id").reset_index(drop=True)
    assert_development_only(frame)
    frame["label"] = frame.diagnosis.map(LABELS)
    qc = compute_qc(dataset, frame, ccfg)
    frame = frame.drop(columns=[c for c in QC_COLUMNS if c in frame]).merge(
        qc, on="recording_id", how="left", validate="one_to_one")
    write_table(qc.merge(frame[["recording_id", "diagnosis"]], on="recording_id"), cache.path("qc.csv"))
    write_table(qc_summary(frame), cache.path("qc_summary.csv"))
    figure = duration_figure(frame, cache.path("duration_distribution.png"))

    keep, exclusions, flow = apply_exclusions(frame, ccfg)
    write_table(exclusions, cache.path("exclusions.csv"))
    write_table(flow, cache.path("exclusion_flow.csv"))
    eligible = frame[keep].copy()
    if eligible.label.nunique() < 2:
        raise RuntimeError("Eligible cohort lacks one of the two classes")

    people = participant_table(eligible)
    folds = assign_folds(people, fcfg["n_splits"], cfg["seed"], fcfg.get("stratify", ["label"]),
                         fcfg.get("group", "duplicate_component"), fcfg.get("age_bins"))
    warnings = []
    if mcfg.get("enabled", True):
        matched, matching_flow = matched_subset(folds, mcfg, cfg["seed"])
    else:
        matched, matching_flow = set(), pd.DataFrame()
    folds["in_matched"] = folds.participant_id.isin(matched)
    m = folds[folds.in_matched]
    if len(m) and m.groupby("fold").label.nunique().reindex(range(fcfg["n_splits"])).fillna(0).lt(2).any():
        warnings.append("matched cohort has a fold missing a class; matched-cohort CV is unreliable")
    write_table(matching_flow, cache.path("matching_flow.csv"))

    cohort = eligible.merge(folds[["participant_id", "fold_group", "fold", "in_matched"]],
                            on="participant_id", validate="many_to_one")
    cohort["initial_recording"] = initial_recordings(cohort)
    first = [*STANDARD_COLUMNS, "label", "fold", "fold_group", "initial_recording", "in_matched", *QC_COLUMNS]
    cohort = cohort[[c for c in first if c in cohort] + [c for c in cohort if c not in first]]
    write_table(cohort, cache.path("cohort.csv"))
    write_table(folds[["participant_id", "fold_group", "label", "age", "sex", "recordings", "fold",
                       "in_matched"]], cache.path("folds.csv"))
    write_table(cohort_summary(cohort), cache.path("cohort_summary.csv"))
    write_table(fold_summary(cohort), cache.path("fold_summary.csv"))

    summary = {
        "test_rows_read": False,
        "source_recordings": len(frame), "eligible_recordings": len(cohort),
        "eligible_participants": int(cohort.participant_id.nunique()),
        "pd_participants": int(folds.label.eq(1).sum()), "control_participants": int(folds.label.eq(0).sum()),
        "matched_participants": int(folds.in_matched.sum()),
        "excluded_recordings": int((~keep).sum()),
        "exclusions_by_criterion": exclusions.criterion.value_counts().to_dict(),
        "warnings": warnings,
    }
    outputs = ["cohort.csv", "folds.csv", "cohort_summary.csv", "fold_summary.csv", "exclusions.csv",
               "exclusion_flow.csv", "qc.csv", "qc_summary.csv", "matching_flow.csv"]
    cache.finalize(outputs + (["duration_distribution.png"] if figure else []), summary)
    log(f"[cohort] {summary['eligible_participants']} participants "
        f"({summary['pd_participants']} PD / {summary['control_participants']} control), "
        f"{summary['eligible_recordings']} recordings, {summary['matched_participants']} matched -> {cache.dir}")
    for w in warnings:
        log(f"[cohort] WARNING: {w}")
    return cache


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    a = p.parse_args(argv)
    run(load_config(a.config), force=a.force)


if __name__ == "__main__":
    main()
