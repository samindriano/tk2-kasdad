"""Describe neighbor/query baseline-confidence gap, without fits or new policies."""
from __future__ import annotations

import gc
import json

import joblib
import numpy as np
import pandas as pd

from scripts.optimization_v3_a02_next_decision import ROOT, REPORT, LEDGER, TRAIN_HASH, sha


def run():
    before = sha(LEDGER)
    train_path = ROOT / 'data/raw/classification/Train.csv'
    assert sha(train_path) == TRAIN_HASH
    train = pd.read_csv(train_path, usecols=['artifact_signal', 'cycle_code'])
    output, inputs = [], {}
    for origin in (59, 63, 67, 71, 75):
        phase = ('optimization_v3_a02_neighbors_v1' if origin in (67, 75)
                 else 'optimization_v3_a02_continuation_v1')
        path = ROOT/'experiments/runs'/phase/f'E{origin}'/'observed'
        receipt = json.loads((path/'fit_receipt.json').read_text())
        assert sha(path/'model.joblib') == receipt['model_sha256']
        inputs[str((path/'model.joblib').relative_to(ROOT))] = sha(path/'model.joblib')
        inputs[str((path/'predictions.npz').relative_to(ROOT))] = sha(path/'predictions.npz')
        model = joblib.load(path/'model.joblib')
        with np.load(path/'predictions.npz', allow_pickle=False) as data:
            rows, cycles = data['score_rows'], data['score_cycles']
            base, candidate = data['baseline_prediction'], data['prediction']
            y = train.iloc[rows].artifact_signal.to_numpy()
            assert np.array_equal(train.iloc[rows].cycle_code.to_numpy(), cycles)
            lookup = {int(row): i for i, row in enumerate(model.bank_row_positions)}
            take = (cycles <= origin+4) & ~data['missing_mask'].astype(bool)
            gap, compatible = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
            # Read compressed arrays once, rather than decompressing them per row.
            counts = data['neighbor_count']
            neighbors = data['neighbor_source_rows']
            similarities = data['neighbor_similarities']
            for i in np.flatnonzero(take):
                n = int(counts[i])
                if not n:
                    continue
                idx = np.array([lookup[int(r)] for r in neighbors[i, :n]])
                assert np.all(model.bank_cycles[idx] <= origin) and cycles[i] > origin
                weight = similarities[i, :n]**2
                dist = abs(model.bank_oof_probabilities[idx]-base[i])
                gap[i] = np.average(dist, weights=weight)
                compatible[i] = weight[dist <= .1].sum()/weight.sum()
            flips = (base >= .5) != (candidate >= .5)
            good = flips & ((candidate >= .5) == y)
            bad = flips & ((candidate >= .5) != y)
            for lo, hi in ((0., 1.01), (.45, .475), (.475, .5), (.5, .525), (.525, np.nextafter(.55, 1))):
                band = take & (base >= lo) & (base < hi)
                record = dict(origin=origin, lo=lo, hi=hi, band_n=int(band.sum()))
                for label, mask in (('useful', good), ('harmful', bad)):
                    subset = band & mask & np.isfinite(gap)
                    record[label+'_n'] = int(subset.sum())
                    record[label+'_gap_median'] = float(np.median(gap[subset])) if subset.any() else None
                    record[label+'_compatible_median'] = float(np.median(compatible[subset])) if subset.any() else None
                output.append(record)
        del model
        gc.collect()
    assert sha(LEDGER) == before
    result = dict(status='DESCRIPTIVE_NO_FITS', model_inputs=inputs, rows=output,
                  definition='Squared-cosine-weighted abs(bank OOF p - query baseline p); compatible fraction gap<=.1',
                  candidate_probabilities_created=False, ledger_before=before, ledger_after=sha(LEDGER),
                  source_sha256=sha(ROOT/'scripts/optimization_v3_a02_confidence_gap.py'))
    (REPORT/'confidence_gap_descriptor.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    data = run()
    for row in data['rows']:
        if row['lo'] == 0.:
            print(json.dumps(row))
