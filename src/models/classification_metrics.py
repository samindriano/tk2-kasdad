"""Binary metrics and inner-only threshold policy for Classification Round 1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def binary_metrics(actual, probability, threshold=0.5):
    y = np.asarray(actual)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Aligned binary labels and one-dimensional probabilities required.')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Finite probabilities in [0,1] required.')
    cut = np.asarray(threshold, dtype=float)
    if cut.ndim > 1 or (cut.ndim == 1 and cut.shape != p.shape) or not np.isfinite(cut).all() or not ((cut >= 0) & (cut <= 1)).all():
        raise ValueError('Invalid threshold.')
    prediction = (p >= cut).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (y.astype(int), prediction), 1)
    support = cm.sum(axis=1)
    per_class = []
    for label in (0, 1):
        tp = int(cm[label, label])
        fp = int(cm[:, label].sum() - tp)
        fn = int(cm[label, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class.append({'label': label, 'support': int(support[label]),
                          'precision': precision, 'recall': recall, 'f1': f1})
    n = len(y)
    both = bool((support > 0).all())
    result = {'n': n, 'threshold': float(cut) if cut.ndim == 0 else None,
              'threshold_range': [float(cut.min()), float(cut.max())] if cut.size else None, 'label_order': [0, 1],
              'confusion_matrix': cm.tolist(), 'per_class': per_class,
              'macro_f1': float(np.mean([c['f1'] for c in per_class])) if n else None,
              'balanced_accuracy': float(np.mean([c['recall'] for c in per_class])) if both else None,
              'average_precision': float(average_precision_score(y, p)) if both else None,
              'positive_rate': float(y.mean()) if n else None,
              'positive_prediction_rate': float(prediction.mean()) if n else None,
              'mean_probability': float(p.mean()) if n else None,
              'brier_score': float(np.mean((p-y)**2)) if n else None,
              'false_positive': int(cm[0, 1]), 'false_negative': int(cm[1, 0]),
              'single_class_actual': bool(n and not both),
              'metric_convention': 'Fixed labels [0,1], undefined per-class precision/recall/F1=0; AP and balanced accuracy null without both actual classes.'}
    return result


def choose_threshold(y_inner, probability_inner, config):
    """Caller must supply only the inner temporal holdout, never outer labels."""
    y = np.asarray(y_inner)
    binary_metrics(y, probability_inner, 0.5)
    if len(y) < config['minimum_inner_validation_rows'] or len(np.unique(y)) < 2:
        return {'threshold': 0.5, 'fallback': True, 'reason': 'Insufficient inner validation class/row coverage',
                'inner_macro_f1': None, 'selection_n': len(y)}
    p = np.asarray(probability_inner)
    # Only confusion counts vary over thresholds; avoid repeatedly computing AP.
    scores = []
    for t in config['grid']:
        pred = p >= t
        tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
        fn, tp = int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())
        scores.append((t, (2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0)/2 +
                             (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)/2))
    best_score = max(score for _, score in scores)
    tied = [(t, score) for t, score in scores if abs(score-best_score) <= 1e-12]
    threshold, score = min(tied, key=lambda v: (round(abs(v[0]-0.5), 12), v[0]))
    return {'threshold': float(threshold), 'fallback': False, 'reason': 'Max inner pooled Macro F1; ties closest to 0.5 then lower',
            'inner_macro_f1': float(score), 'selection_n': len(y)}


def subgroup_masks(frame, fit_frame):
    """Membership is descriptive; vocabulary comes only from the relevant fit."""
    stream = frame.signal_stream.fillna('').astype(str).str.strip().eq('').to_numpy()
    impact = frame.impact_load.to_numpy()
    unseen = np.zeros(len(frame), dtype=bool)
    missing = np.zeros(len(frame), dtype=bool)
    lower, upper = fit_frame.impact_load.quantile([1/3, 2/3]).to_numpy()
    masks = {'EF0': frame.entry_friction.eq(0).to_numpy(), 'EF1': frame.entry_friction.eq(1).to_numpy(),
             'stream_missing': stream, 'stream_present': ~stream,
             'impact_low': impact < lower, 'impact_middle': (impact >= lower) & (impact < upper),
             'impact_high': impact >= upper}
    for ef in [0, 1]:
        for name, present in [('missing', stream), ('present', ~stream)]:
            masks[f'EF{ef}_stream_{name}'] = frame.entry_friction.eq(ef).to_numpy() & present
    for column in ['org_code', 'module_code', 'pattern_code']:
        known = set(fit_frame[column].dropna().astype(str).str.strip())
        m = frame[column].isna().to_numpy() | frame[column].fillna('').astype(str).str.strip().eq('').to_numpy()
        new = (~m) & (~frame[column].astype(str).str.strip().isin(known)).to_numpy()
        unseen |= new
        missing |= m
        masks[column+'_missing'] = m
        masks[column+'_present'] = ~m
    masks.update({'all_codes_known_nonmissing': ~unseen & ~missing,
                  'any_unseen_code': unseen, 'any_missing_code': missing})
    return masks


def reliability(y, probability):
    y, p = np.asarray(y), np.asarray(probability)
    rows = []
    for index in range(10):
        mask = (p >= index/10) & ((p < (index+1)/10) if index < 9 else (p <= 1))
        rows.append({'lower': index/10, 'upper': (index+1)/10, 'n': int(mask.sum()),
                     'mean_probability': float(p[mask].mean()) if mask.any() else None,
                     'observed_positive_rate': float(y[mask].mean()) if mask.any() else None})
    return rows
