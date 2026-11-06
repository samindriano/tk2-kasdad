# CLS_POS_RECENCY_E79_V1 — one authorized full-Train benchmark

Status: one exploratory-only E79 benchmark authorized by root decision at
reports/simplification_v1/CLS_POS_RECENCY_V1/ROOT_RESULTS_DECISION.md.
The original pilot's scientific verdict remains FAIL. This lane cannot change
the champion, scientific gate, or any other experiment.

## Frozen recipe

- Fit population: official Classification Train cycles 47–79, 50,608 rows.
- Gamma fit population: stream-present rows only. No preprocessing, vocabulary,
  IDF, encoder, scaler, OHE, bag model, A02 model, or uniform gamma refit.
- Recency base weight: max(0.25, 2**(-(79-cycle_code)/8)).
- Normalize base weights by their mean separately within each target class on
  the stream-present fit rows. This preserves each class's row mass.
- Fit gamma from zero on the frozen positional matrix with the frozen C=1
  summed weighted binary-logistic objective plus 0.5 * ||gamma||^2.
  Reuse the pilot L-BFGS-B options: maxiter 2000, ftol 1e-12, gtol 1e-6,
  maxls 50. The independent verifier recalculates weights, loss, and gradient;
  it never refits.
- Do not fit a shuffled, uniform, or other gamma comparator.

## Source and application boundaries

- Reuse only the frozen MAIN TP2 feature transforms and fitted states named in
  SOURCE_INVENTORY.json. Rebuild and verify full-Train base/position matrix
  parity; use the position matrix and C2_B logit offset for this fit. The
  verified DF-map may be read for token-order/source audit only; no global
  strength penalty or aligned weights enter the E79 objective.
- The original full-Train C=1 uniform POS_BAG_OFFSET gamma is used only to
  reconstruct the frozen full champion on Test.
- After independent fit verification passes, one feature-only official Test
  transform may run. Test target columns are forbidden. SampleSubmission is
  limited to ordered IDs and schema. The original champion CSV and A02 Test
  probability cache are references, not training inputs.
- Present stream: 0.5 * sigmoid(z_BAG + X_position @ gamma_E79) + 0.5 * p_A02,
  threshold >= 0.50.
- Missing stream: p_A02 copied exactly, threshold >= 0.53230976129477.
- Write one local 29,421-row CSV. Root owns any later upload decision; this
  lane must not contact Kaggle or submit.

## One-shot controls

The source commit, synthetic lifecycle QA, input hashes, prefit QA, freeze, and
cap-one reservation must be committed before the official optimizer starts.
The official optimizer may start once. Any started or interrupted optimizer is
consumed and must not be retried. Feature-only Test application is allowed only
after the saved independent fit verification has PASS status. No extra
experiments, parameter rescue, Test-based selection, notebook changes, push,
merge, rebase, or reset are authorized.

## Reproduction and resume sequence

Run once from the audit repository root with the official `Train.csv` available
at the pinned MAIN path. These first two QA commands write immutable receipts;
if the corresponding files already exist, inspect them instead of rerunning.
The source inventory, bridge, and helper run read-only against MAIN; generated
feature arrays, fit state, and CSV remain local/ignored.

```powershell
python scripts/cls_pos_recency_e79_v1.py --synthetic
python scripts/cls_pos_recency_e79_v1.py --test-suite
# Commit only the prepared source, protocol, inventory, and PASS QA outputs.
python scripts/cls_pos_recency_e79_v1.py --build-freeze
# Commit PREFIT_QA.json, FREEZE.json, and RESERVATION.json before fitting.
python scripts/cls_pos_recency_e79_v1.py --fit-once
python scripts/verify_cls_pos_recency_e79_v1.py --fit
python scripts/cls_pos_recency_e79_v1.py --apply-test-once
python scripts/verify_cls_pos_recency_e79_v1.py --application
```

If `ATTEMPT.started.json` exists, do not rerun `--fit-once`, even when the
process was interrupted. If `APPLICATION_ATTEMPT.started.json` exists, do not
rerun Test transform or CSV preparation. Inspect and preserve the existing
receipt/failure record; report the state for review. The separately retained
`SYNTHETIC_QA.json` and `TEST_SUITE_QA.json` preserve the first lifecycle/test
outputs. The first test run had 8 passes and 1 incorrect expected-value
assertion; the frozen prefit source binds only the corrected
`TEST_SUITE_QA_FINAL.json` and the repeated passing `SYNTHETIC_QA_FINAL.json`.
