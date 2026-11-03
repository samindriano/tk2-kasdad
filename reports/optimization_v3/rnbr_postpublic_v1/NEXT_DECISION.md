# Sol decision: one exact-impact residual-neighbor pilot

**GO: four target-memory fits, diagnostic only.** No new Test inference/upload.
Sam asked to continue after the user-reported R-NBR public11.19115 versus R06
11.16620. The local CSV is reviewed, but server hash/ID are unknown; see
PUBLIC_RECEIPT.json. Public loss0.02495 does not identify a statistically
significant difference, private performance, a cause, or an optimal blend weight.

## What the new zero-fit audit actually shows

Reproduce: python -m scripts.optimization_v3_rnbr_postpublic_audit.
The script reuses safe numeric bundles, checks row IDs/cycles and prediction
formula, and writes small aggregate tables. It performs no model fit, Test
access or ledger mutation. This analysis is explicitly adaptive/descriptive.

| Origin67 forecast horizon | n | R-NBR MAE gain vs R06 |
|---|---:|---:|
|1-4 cycles|4,950|-0.112556|
|5-8 cycles|7,462|-0.041044|
|9-12 cycles|8,491|-0.017145|

The harm decreases with horizon in this origin: a simple claim that corrections
become worse as forecasts age is unsupported. Do not fit a horizon decay rescue.
H12 target<=5 improves0.956944 MAE on1,048 rows, while the middle target range
harms0.108793 on18,798 rows. Tail gain can coexist with aggregate loss. Actual
target strata are descriptive and must never be used for inference routing.

## Why one different mechanism is worth testing

Original R-NBR retrieves text-only neighbors and transfers residuals. Hypothesis:
restricting retrieval to donors with the query's exact observed impact_load
may make residual transfer more comparable. This is not a causal claim.
R06 already sees structured features, so matching might be redundant or hurt
text relevance; a negative result is informative and ends this specific lane.

Novelty checks: R18 is a global recent median offset; R19 is an EF-specific
median offset; the old shallow residual head failed. None of these implements
exact-impact-conditioned text-neighbor retrieval. Original R-NBR is text-only.
See research_backlog/cards/R18.md, R19.md, regression_residual/FINDINGS.md and
rnbr_readiness_v1/R-NBR_PROPOSED_CARD.md. This is not a claim every historical
line of code was exhaustively searched.

Label-free availability check from existing banks: E67 has19,480/19,528
present queries with>=5 same-impact donors; E75 has7,959/7,965. These are donor
counts, not positive-cosine support. Exact matching does not require arbitrary
bin edges: impact_load has80 observed values and a concentrated distribution.

## Frozen pilot recipe: RNBR_IMPACT_MATCH_V1

- Origins67 and75; donors present cycles56-origin, same R06 final-mapped OOF
  residuals and strictly earlier source cutoffs as original R-NBR.
- E67 query68-79 fixed-origin H12; W4 is its68-71 subset. E75 query76-79 W6.
  Pilot primary=W4+W6,n13,441. H12=E67 all68-79,n20,903. Never overlay maps.
- Reuse each origin's fitted A02 target-free TF-IDF vectorizer (67 or75), its
  original source hashes/receipt and fixed R06 query baseline. Extract only
  fitted vectorizer state; no Classification labels/probabilities, no refit.
- Search top32 positive-cosine neighbors **within the full same-impact donor
  bank**, not by filtering the old top32 list. Exact impact equality uses the
  original finite float64 parsed from the same CSV, no rounding/binning.
  Missing impact or no matching support gets exact R06 fallback; invalid IDs,
  provenance or nonfinite values fail integrity checks rather than silent repair.
- Stable order descending cosine then source row. Same cosine-squared weighted
  median/tie rule, delta=clip(.25*median,-5,5), finalclip0..100, no rerank.
  Missing stream or<5 positive neighbors falls back exactly to R06.
- Matched control shuffles residual assignments within exact donor cycle x
  exact impact_load x fixed R06 prediction decile, seed42 reset each origin,
  ascending cycle/impact/decile/source-row ordering. Decile=min(9,floor(p/10)),
  p clipped0..100. Geometry/fallback/marginal residual cell distributions match.
  This is a new control, not the original R-NBR's less-conditioned shuffle.

Reviewer raised a legitimate small-cell control concern. Sol measured it rather
than assume:95.26%/95.60% of donors are in cells with>=2 members; the exact fixed
source permutation moves87.65%/88.34% at67/75. Report residual-value movement
after fit as well: source movement alone does not prove residuals differ.
Do not substitute a random-donor control after seeing outcomes.

## Preflight before target-memory fit

Compute new geometry using target-free features and saved fitted transforms.
Require>=95% of present queries have>=5 positive same-impact neighbors in each
of W4,W6 and E67 H12; preserve exact denominators and unmodified R06 fallback.
Require>=80% donor source-position movement for fixed shuffle at each origin.
If either fails, report SKIPPED_SCIENTIFIC_PREFLIGHT without task fits; do not
widen matching, lower support or change shuffle. Missing/corrupt required state
is a technical blocker to Sol, never a skipped scientific failure or refit.

Synthetic actual-wrapper persistence/readback/resume and independent formula,
tie, same-impact, strict time, fallback and view-isolation checks must pass.
Freeze recipe/source/input/geometry hashes before new residual policy fitting
or opening query targets for evaluation. Saved residuals still form a new
target-memory predictor and count against the fit ledger.

## Prospective pilot decision criteria

Compare on all declared rows to origin-matched R06, original R-NBR and new
matched shuffle. Also report common-supported rows; do not gate only there.
All are required to label this new pilot ADVANCE:

1. W4 and W6 each nonnegative gain versus R06; pooled gain>=0.03.
2. E67 H12 nonnegative gain versus R06. This stricter new pilot policy targets
   the deployment horizon; it never changes original R-NBR's old PASS verdict.
3. Pooled gain versus new shuffle>=0.01, and H12 gain versus shuffle>=0.
4. Pooled gain versus original R-NBR>=0.03 and H12 gain versus original>=0.
5. Supported missing/target<=5/target>=95 groups,n>=100, harm<=0.25 versus R06
   in W4,W6,pooled,H12; missing incremental change exactly0. Other inherited
   R06/legacy risks remain separately visible, never retrospectively cleared.
6. Integrity/independent verification/accounting PASS; all unsupported groups
   disclosed. Report per-cycle signs, cosine/donor-age and fallback coverage.

Failure means RETIRE this exact matching recipe, not a grid of rescue thresholds.
ADVANCE means pilot only; no automatic continuation, full-Train fit or upload.
The old development cycles are reused and these criteria do not create an
independent holdout or erase adaptive/public feedback.

## Execution cap and ownership

Append explicit cap77->81,pilot38->42; continuation38,technical_retry1 and
control_replay0 unchanged. Verify ledger current state before reserving four
unique fit IDs: E67/E75 observed/shuffled. No retries, refunds, base learner fits
or new preprocessing fits. Preserve all events and historical frozen sources.
One job,two threads,6GiB ceiling plus2GiB free reserve. No installs,Test access,
Kaggle,notebook/shared-team edits,Git binaries/CSVs or push.

Luna owns new namespace experiments/optimization_v3/rnbr_impact_match_v1,
reports/optimization_v3/rnbr_impact_match_v1, matching new scripts/src/tests and
ignored run folders. Do not edit frozen R-NBR runner/verifiers. Persist outputs
atomically before reports and verify allow_pickle=False/readback/idempotence.
Commit scoped files locally; send findings, counterevidence, support/gates, QA,
ledger and commit back to Sol before the executor's final reply, then STOP.
Routine implementation choices are authorized; real blockers come to Sol.
