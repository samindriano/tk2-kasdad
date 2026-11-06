# CLS_POS_EF_CONTEXT_V1 — frozen pilot protocol

## Question

Does an entry-friction-conditioned positional correction add predictive value
to the existing A02 + uniform POS-offset classification champion, after the
structured entry-friction main effect is already present in the fixed bag-logit
offset? This is a predictive interaction diagnostic, not a causal claim.

## Scope and source lineage

The pilot uses only the official Classification `Train.csv` (cycles 47–79),
the verified `POSITIONAL_TEXT_PILOT_V1` C2_B fit/matrices, the original
`POS_BAG_OFFSET_V1` uniform positional gamma, and the original-only members of
the verified Poisson source cache. The full champion must replay bit-for-bit
before the new candidate is constructed. The loader reads only explicit
allowlisted parent-array members; Poisson-replica predictions and saved label
members remain unopened. No Test/SampleSubmission, Kaggle, full-Train refit,
base refit, vocabulary/IDF refit, POS-gamma refit, or new feature transform is
part of this pilot.

## Frozen model

For each origin E67 and E75, the fit population is prefix rows with a present
`signal_stream`; query rows are strictly after the origin. `entry_friction`
must be binary 0/1 and each level must have at least 200 present fit rows. From
present fit rows only, compute `mu = mean(EF1)` and
`scale = sqrt(mu * (1-mu))`; use `q=(EF1-mu)/scale`.

The fixed base offset is reconstructed as `z_POS = z_BAG + X_POS @
gamma_original`. The new design is `X_POS * q`, on present rows only. Fit only
the added coefficient vector, without an intercept, from zeros, using summed
logistic loss plus `0.5 * ||gamma||^2` and L-BFGS-B (`C=1`, maxiter 2000,
ftol 1e-12, gtol 1e-6, maxls 50). The optimizer must succeed and final loss
must not exceed zero-gamma loss by more than 1e-8. Each failed/interrupted fit
ID is consumed and never retried; a durable receipt may be resumed without
calling the optimizer.

The matched control independently shuffles EF assignments within each exact
cycle × stream-presence stratum for fit and query populations, preserving
stratum counts with seed 42. At least 100 present assignments must move in
each fit/query population. The original structured EF main effect remains in
`z_POS` for both arms.

For present rows, candidate probability is a 50/50 blend of the fixed A02
probability and the newly corrected POS probability. Present threshold is
0.50. Missing-stream probability and decision are the exact A02 fallback with
the inherited missing threshold 0.53230976129477. New gamma fits are exactly
four: E67/E75 × REAL/SHUFFLED.

## Validation and scoring boundary

Before any fit, run the complete synthetic CLI lifecycle, persistence and
readback, resume-from-receipt, tamper/failure checks, input-skip poison test,
and independent metric arithmetic. Freeze and commit source/input hashes and
the exact cap-4 reservation before any official fit. Run one optimizer at a
time under the shared OS advisory lock, with two threads and live RAM gates.

The producer, independent target-free prelabel verifier, and one-pass scoring
CLI are separate stages. Query target values may be read only by the scoring
CLI, after `PRELABEL_VERIFICATION.json` has been committed and reverified from
Git. The target selector skips all non-selected CSV records before parsing the
target column. Fit targets are limited to each origin's present prefix rows.

The primary unique pool is E67 cycles 68–75 plus E75 cycles 76–79 (20,903
non-overlapping rows). E67 W4/W5/W6/H12 and E75 W6 are descriptive overlapping
windows, not independent replications. Report each view, each cycle's gain,
availability/EF/fit-prefix text-load subgroups (support n≥200 and at least 30
per class), and real-versus-shuffled controls. Unsupported groups remain
UNKNOWN.

## Frozen gates and stop point

Scientific qualification requires unique Macro-F1 gain ≥0.0005, E67 H12 and
E75 W6 gains ≥0, real-minus-shuffled unique gain ≥0.0002, real-minus-shuffled
E75 W6 gain ≥0, every supported subgroup harm no worse than 0.01 Macro F1 or
0.02 positive recall, exact missing fallback, and independent QA PASS.

Exploratory eligibility requires unique gain >0, or E75 W6 gain ≥0.0005 with
unique gain ≥−0.0005; all supported subgroup harm must be no worse than 0.02
Macro F1 or 0.03 positive recall, missing fallback must be exact, and QA must
PASS. These gates do not promote a candidate or authorize follow-on work.
Regardless of the outcome, this pilot stops before E79, Test, SampleSubmission,
Kaggle, or champion changes and returns for Sol's review.

## Commands

Run from the audit repository root, supplying the official TP2 repository root:

```powershell
python scripts/cls_pos_ef_context_v1.py synthetic --output $env:TEMP/cls_pos_ef_context_synthetic
python scripts/cls_pos_ef_context_v1.py prepare-source --official-root <official-tp2-root>
python scripts/cls_pos_ef_context_v1.py freeze --official-root <official-tp2-root>
python scripts/cls_pos_ef_context_v1.py reserve --official-root <official-tp2-root>
python scripts/cls_pos_ef_context_v1.py fit --official-root <official-tp2-root> --fit-id CLS_POS_EF_CONTEXT_V1_E67_REAL
# Invoke each of the four frozen IDs separately.
python scripts/cls_pos_ef_context_v1.py build --official-root <official-tp2-root>
python scripts/verify_cls_pos_ef_context_v1.py prelabel --official-root <official-tp2-root>
# Commit the target-free prelabel report before this command.
python scripts/score_cls_pos_ef_context_v1.py --official-root <official-tp2-root>
python scripts/verify_cls_pos_ef_context_v1.py score
```
