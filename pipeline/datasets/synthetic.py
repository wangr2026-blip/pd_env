"""Synthetic sustained vowels with planted acoustic ground truth.

Purpose: smoke-test every stage end to end without participant data, and give
probes a known answer. Each recording is a Rosenberg glottal pulse train with
controlled F0, cycle-to-cycle period perturbation (jitter), cycle-to-cycle
amplitude perturbation (shimmer) and additive aspiration noise (HNR), passed
through an /a/-like formant cascade.

The PD effect is planted in the perturbation/noise features and, deliberately,
in age (PD participants are older), mimicking the mPower age confound. Some
recordings are deliberately unusable (silent, too short, unvoiced/whispered),
some participants lack age/sex/label, and some rows belong to the sealed test
split, so every Stage 0 exclusion and the test guard are exercised.

Nothing here resembles real participant records; identifiers are synthetic.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from pipeline.datasets.base import SAMPLE_RATE, Dataset, validate_waveform

DEFAULTS = {
    "n_participants": 60,
    "pd_fraction": 0.4,
    "recordings_per_participant": [1, 3],
    "duration_s": [4.0, 6.0],
    "split_fractions": [0.7, 0.15, 0.15],   # train / val / sealed test
    "unlabeled_fraction": 0.05,
    "missing_age_fraction": 0.03,
    "missing_sex_fraction": 0.03,
    "bad_audio_fraction": 0.06,             # split across silent / short / unvoiced
    "duplicate_pairs": 2,
    "formants_hz": [700, 1220, 2600],
    "bandwidths_hz": [80, 90, 120],
}
MEDICATION = ["Immediately before Parkinson medication",
              "Just after Parkinson medication (at your best)",
              "Another time",
              "I don't take Parkinson medications"]
TRUTH = ["true_f0_hz", "true_jitter_local", "true_shimmer_local", "true_hnr_db"]


def _rosenberg(tau, period, open_q=0.4, close_q=0.16):
    tp, tn = open_q * period, close_q * period
    g = np.zeros_like(tau)
    rise = (tau >= 0) & (tau < tp)
    fall = (tau >= tp) & (tau < tp + tn)
    g[rise] = 0.5 * (1 - np.cos(np.pi * tau[rise] / tp))
    g[fall] = np.cos(np.pi * (tau[fall] - tp) / (2 * tn))
    return g


def _formant_filter(x, formants, bandwidths, sr):
    for f, b in zip(formants, bandwidths):
        r = np.exp(-np.pi * b / sr)
        a = [1.0, -2 * r * np.cos(2 * np.pi * f / sr), r * r]
        x = lfilter([sum(a)], a, x)  # unity gain at DC
    return x


def synth_vowel(rng, duration, f0, jitter, shimmer, hnr_db, formants, bandwidths,
                kind="voiced", sr=SAMPLE_RATE):
    """One recording. jitter/shimmer are target *local* perturbations (fractions).

    For i.i.d. Gaussian relative perturbation with SD s, E|x_i - x_{i+1}| = 2s/sqrt(pi),
    so s = target / 1.128 yields the requested local jitter/shimmer in expectation.
    """
    n = int(round(duration * sr))
    lead = int(rng.uniform(0.2, 0.6) * sr)          # leading silence, as in real recordings
    floor = 10 ** (-60 / 20) * rng.standard_normal(n)
    if kind == "silent":
        return (floor * 0.5).astype(np.float32)
    voiced_n = n - lead - int(0.1 * sr)
    if kind == "unvoiced":
        source = rng.standard_normal(voiced_n)
    else:
        sj, ss = jitter / 1.128, shimmer / 1.128
        phase, end = rng.uniform(0, 2 * np.pi), voiced_n / sr
        source = np.zeros(voiced_n)
        t = 0.0
        while t < end:
            f = f0 * (1 + 0.01 * np.sin(2 * np.pi * 0.4 * t + phase))  # slow 1% drift
            period = (1 / f) * (1 + sj * rng.standard_normal())
            amp = max(0.05, 1 + ss * rng.standard_normal())
            lo, hi = int(np.ceil(t * sr)), min(voiced_n, int((t + 0.56 * period) * sr) + 1)
            idx = np.arange(lo, hi)
            source[idx] += amp * _rosenberg(idx / sr - t, period)
            t += period
        source = np.diff(source, prepend=0.0)        # lip radiation
    voiced = _formant_filter(source, formants, bandwidths, sr)
    if kind == "voiced":
        noise = _formant_filter(rng.standard_normal(voiced_n), formants, bandwidths, sr)
        scale = np.sqrt(np.mean(voiced ** 2) / (np.mean(noise ** 2) * 10 ** (hnr_db / 10)))
        voiced = voiced + scale * noise
    ramp = int(0.05 * sr)
    env = np.ones(voiced_n)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    voiced = voiced * env
    voiced = voiced / np.max(np.abs(voiced)) * rng.uniform(0.15, 0.6)
    out = floor.copy()
    out[lead:lead + voiced_n] += voiced
    return out.astype(np.float32)


class SyntheticDataset(Dataset):
    name = "synthetic"

    def __init__(self, cfg, seed):
        super().__init__(cfg)
        self.seed = int(seed)
        self.params = {**DEFAULTS, **(cfg.get("params") or {})}
        self._all = None

    def fingerprint(self):
        return {"name": self.name, "seed": self.seed, "params": self.params}

    def all_recordings(self):
        """Every generated row INCLUDING the sealed test split (for tests only)."""
        if self._all is None:
            self._all = self._generate()
        return self._all

    def recordings(self):
        frame = self.all_recordings()
        # The test guard: sealed rows are dropped before anything else sees them.
        return self.check(frame[frame.original_split.isin(["train", "val"])].reset_index(drop=True))

    def load_audio(self, row):
        p = self.params
        rng = np.random.default_rng([self.seed, int(row["audio_seed"])])
        wave = synth_vowel(rng, row["duration_target_s"], row["true_f0_hz"], row["true_jitter_local"],
                           row["true_shimmer_local"], row["true_hnr_db"], p["formants_hz"],
                           p["bandwidths_hz"], kind=row["audio_kind"])
        return validate_waveform(wave)

    def _generate(self):
        p = self.params
        rng = np.random.default_rng([self.seed, 0])
        n = int(p["n_participants"])
        pd_ = rng.random(n) < p["pd_fraction"]
        age = np.where(pd_, rng.normal(64, 9, n), rng.normal(46, 15, n)).clip(18, 90).round()
        male = rng.random(n) < np.where(pd_, 0.6, 0.5)
        diagnosis = np.where(pd_, "pd", "control").astype(object)
        diagnosis[rng.random(n) < p["unlabeled_fraction"]] = "unknown"
        people = pd.DataFrame({
            "participant_id": [f"synth-p{i:04d}" for i in range(n)],
            "diagnosis": diagnosis,
            "age": age,
            "sex": np.where(male, "Male", "Female").astype(object),
            "original_split": rng.choice(["train", "val", "test"], n, p=p["split_fractions"]),
        })
        people.loc[rng.random(n) < p["missing_age_fraction"], "age"] = np.nan
        people.loc[rng.random(n) < p["missing_sex_fraction"], "sex"] = np.nan
        people["duplicate_component"] = people.participant_id
        # Pairs of same-label participants linked by duplicate audio must share a fold.
        for k in range(int(p["duplicate_pairs"])):
            for label in ("pd", "control"):
                pool = people.index[people.diagnosis.eq(label) & people.original_split.ne("test")]
                if len(pool) >= 2 * (k + 1):
                    a, b = pool[2 * k], pool[2 * k + 1]
                    people.loc[b, ["duplicate_component", "original_split"]] = (
                        people.loc[a, "duplicate_component"], people.loc[a, "original_split"])
        # Participant-level voice baselines; PD and age raise perturbation, lower HNR.
        is_pd = people.diagnosis.eq("pd").to_numpy()
        age_c = np.nan_to_num(people.age.to_numpy() - 50)
        f0_base = np.where(people.sex.eq("Male"), rng.normal(120, 15, n), rng.normal(205, 20, n))
        jit_base = np.exp(rng.normal(np.log(np.where(is_pd, 0.011, 0.005)), 0.25)) * (1 + 0.004 * age_c)
        shim_base = np.exp(rng.normal(np.log(np.where(is_pd, 0.06, 0.035)), 0.2))
        hnr_base = rng.normal(np.where(is_pd, 15, 21), 2) - 0.05 * age_c

        rows, audio_seed, n_bad = [], 1, 0
        bad_kinds = ["short", "silent", "unvoiced"]  # cycled so each type occurs
        lo, hi = p["recordings_per_participant"]
        start = pd.Timestamp("2015-03-09", tz="UTC")
        for i, person in people.iterrows():
            for v in range(int(rng.integers(lo, hi + 1))):
                kind = "voiced"
                if rng.random() < p["bad_audio_fraction"]:
                    kind = bad_kinds[n_bad % len(bad_kinds)]
                    n_bad += 1
                duration = 1.2 if kind == "short" else float(rng.uniform(*p["duration_s"]))
                rows.append({
                    "recording_id": f"synth-r{audio_seed:05d}",
                    **person.to_dict(),
                    "task": "sustained_phonation",
                    "medication": (rng.choice(MEDICATION) if person.diagnosis == "pd"
                                   else "I don't take Parkinson medications"),
                    "recorded_at": start + pd.Timedelta(days=int(rng.integers(0, 180)) + 200 * v),
                    "audio_kind": "voiced" if kind == "short" else kind,
                    "duration_target_s": duration,
                    "true_f0_hz": float(max(70, f0_base[i] * rng.normal(1, 0.02))),
                    "true_jitter_local": float(max(0.001, jit_base[i] * rng.normal(1, 0.1))),
                    "true_shimmer_local": float(max(0.005, shim_base[i] * rng.normal(1, 0.1))),
                    "true_hnr_db": float(hnr_base[i] + rng.normal(0, 1)),
                    "audio_seed": audio_seed,
                })
                audio_seed += 1
        return pd.DataFrame(rows)
