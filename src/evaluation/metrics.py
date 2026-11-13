"""Fixed-threshold binary metrics and calibration at recording and participant levels."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             balanced_accuracy_score, confusion_matrix, brier_score_loss, log_loss)


def binary_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    predicted = p >= .5
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    bins = []
    ece = 0.
    assignments = np.minimum((p * 10).astype(int), 9)
    for index in range(10):
        lo = index / 10
        selected = assignments == index
        n = int(selected.sum())
        observed = float(y[selected].mean()) if n else None
        probability = float(p[selected].mean()) if n else None
        if n:
            ece += n / len(y) * abs(observed - probability)
        bins.append({'lower': float(lo), 'n': n, 'observed_pd': observed, 'mean_probability': probability})
    both = len(np.unique(y)) == 2
    return {'n': len(y), 'auroc': float(roc_auc_score(y, p)) if both else None,
            'auprc': float(average_precision_score(y, p)) if both else None,
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predicted)) if both else None,
            'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
            'specificity': float(tn / (tn + fp)) if tn + fp else None,
            'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'ece_10_bins': ece, 'calibration_bins': bins, 'threshold': .5}


def evaluate(predictions, aggregation='mean'):
    if aggregation not in ('mean', 'median'):
        raise ValueError('Participant aggregation must be mean or median')
    if predictions.groupby('participant_id').label.nunique().gt(1).any():
        raise ValueError('Conflicting participant labels in predictions')
    if predictions.groupby('participant_id').split.nunique().gt(1).any():
        raise ValueError('Participant leakage in predictions')
    participants = predictions.groupby(['participant_id', 'split'], as_index=False).agg(
        label=('label', 'first'), probability=('probability', aggregation), recordings=('recording_id', 'size'))
    metrics = {}
    for level, table in [('recording', predictions), ('participant', participants)]:
        metrics[level] = {split: binary_metrics(g.label, g.probability) for split, g in table.groupby('split')}
    return metrics, participants
