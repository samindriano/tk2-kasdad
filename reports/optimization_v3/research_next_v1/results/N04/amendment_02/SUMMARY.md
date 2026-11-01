# N04 — Covariate weighting beyond time allocation

Status: SKIPPED_SCIENTIFIC_PREFLIGHT

At least one cutoff failed the frozen domain-overlap or weight-stability gates; task estimators were not fit.

## Domain overlap and pre-task gates

| Cutoff | OOF heldout AUCs | ESS fraction | Source clipped | Gate |
|---:|---:|---:|---:|:---:|
| 55 | 0.550, 0.542 | 0.829 | 0.062 | FAIL |
| 67 | 0.592, 0.605 | 0.816 | 0.036 | PASS |

OOF domain probabilities were fit by cycle-parity folds. The recent/source labels were derived only inside each fit prefix; the domain learner excluded cycle_code, record_id and task targets. Score labels were used only for temporal evaluation. Test and Kaggle were not used.
