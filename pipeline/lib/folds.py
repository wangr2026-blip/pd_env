"""Participant-level folds, sealed-test guard and participant aggregation.

Every downstream classifier and probe must iterate folds through
``iter_folds`` so all stages share identical train/test partitions.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DEVELOPMENT_SPLITS = ("train", "val")


def assert_development_only(frame, column="original_split"):
    """Refuse any row from the sealed final-test split."""
    if column not in frame:
        raise ValueError(f"Missing {column}; cannot verify the test split is sealed")
    bad = set(frame[column].dropna().unique()) - set(DEVELOPMENT_SPLITS)
    if bad or frame[column].isna().any():
        raise ValueError(f"Sealed/unknown split rows present: {sorted(map(str, bad))}")


def stratification_key(people, columns, age_bins=None):
    parts = []
    for col in columns:
        if col == "age_bin":
            if not age_bins:
                raise ValueError("Stratifying on age_bin requires folds.age_bins")
            parts.append(pd.cut(people.age, age_bins, right=False).astype(str))
        else:
            parts.append(people[col].astype(str))
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def assign_folds(people, n_splits, seed, stratify=("label",), group="duplicate_component",
                 age_bins=None):
    """One row per participant in, the same rows plus ``fold`` out.

    Groups (duplicate-connected components, else participants) are never
    divided between folds. Stratification is approximate when groups are large.
    """
    if people.participant_id.duplicated().any():
        raise ValueError("assign_folds expects one row per participant")
    people = people.sort_values("participant_id").reset_index(drop=True)
    groups = people[group] if group in people else people.participant_id
    groups = groups.fillna(people.participant_id)
    key = stratification_key(people, stratify, age_bins)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(people), -1)
    for index, (_, test) in enumerate(splitter.split(np.zeros(len(people)), key, groups)):
        fold[test] = index
    out = people.assign(fold=fold, fold_group=groups.to_numpy())
    verify_folds(out, n_splits)
    return out


def verify_folds(frame, n_splits=None):
    if (frame.fold < 0).any():
        raise AssertionError("Unassigned participants")
    for col in ("participant_id", "fold_group"):
        if col in frame and frame.groupby(col).fold.nunique().gt(1).any():
            raise AssertionError(f"Leakage: a {col} spans multiple folds")
    if n_splits is not None and frame.fold.nunique() != n_splits:
        raise AssertionError("Empty fold")
    classes = frame.groupby("fold").label.nunique()
    if classes.lt(2).any():
        raise AssertionError(f"Folds missing a class: {classes[classes.lt(2)].index.tolist()}")


def iter_folds(frame):
    """Yield (fold, train_mask, test_mask) over rows of a fold-annotated frame."""
    for fold in sorted(frame.fold.unique()):
        test = frame.fold.to_numpy() == fold
        yield int(fold), ~test, test


def aggregate_participants(predictions, how="mean"):
    """Recording-level predictions -> one row per participant (and fold)."""
    if how not in ("mean", "median"):
        raise ValueError("Participant aggregation must be 'mean' or 'median'")
    if predictions.groupby("participant_id").label.nunique().gt(1).any():
        raise ValueError("Conflicting labels within a participant")
    keys = ["participant_id"] + (["fold"] if "fold" in predictions else [])
    if "fold" in predictions and predictions.groupby("participant_id").fold.nunique().gt(1).any():
        raise ValueError("Participant spans folds")
    return predictions.groupby(keys, as_index=False).agg(
        label=("label", "first"), probability=("probability", how),
        recordings=("probability", "size"))
