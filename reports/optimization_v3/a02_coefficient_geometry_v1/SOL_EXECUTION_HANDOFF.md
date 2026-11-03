# A02 coefficient-informed geometry: one new Classification lane

GO_IMPLEMENT_AND_EXECUTE_ONE_PILOT. Sam explicitly chose "Buka satu lane riset
baru untuk mengejar skor lagi" after B01 closure. This authorizes only the lane
below, not automatic successor experiments, full-Train/Test or submissions.

Repository C:/Users/Sam/OneDrive/Documents/KASDAD/TP 2. Incoming HEAD
07c6b20f186dda96f9b86e3b83db71615a274937, clean. Luna owns implementation,
focused tests, synthetic lifecycle, official pilot, reports and scoped local
commits. Sol owns the frozen question and subsequent review and will stop writing
after dispatch. Existing Luna chat is the executor; one official job at a time.

## Hypothesis, novelty and counterevidence

Hypothesis: label-informed token emphasis changes A02's donor selection in a
useful way beyond unchanged lexical TF-IDF cosine, even after correction mass is
held fixed. It does not claim token meaning or causal relevance.
Read these exact sources and existing A02 core before implementation:
- src/models/classification_a02_neighbors.py
- reports/optimization_v3/a02_neighbors_v1/FINAL_SUMMARY.md
- reports/optimization_v3/a02_next_decision_v1/DECISION.md
- reports/optimization_v3/a02_reliability_audit_v1/SUMMARY.md
- reports/optimization_v3/classification_diagnostics/FINDINGS.md
- src/models/campaign_classification.py (DC_NBSVM)
- scripts/optimization_v3_positional_text_pilot_v1.py
- src/models/optimization_v3_positional_text_pilot_v1.py

Bounded independent read-only novelty review found no coefficient-to-A02-cosine
experiment in targeted reports/code. This is scoped evidence, not exhaustive
proof. Prior NB-SVM applies signed class log-ratios then trains LR; it reportedly
lost about .0027 primary/.0005 H12 versus C2_B. That is relevant adverse evidence
for supervised lexical weighting but does not test donor retrieval. C09 feature
crosses, N02 sketch, failed FM, B01 centering and rejected confidence-compatible
neighbor proposals also do not implement this geometry. Do not reopen them.

## Exact recipe: CGEOM_V1

Use original A02 E67/E75 text vectorizer, complete residual donor bank, same
OOF Sparse25 bank probabilities, observed residuals and query baseline. Reuse
same-origin cached faithful C2_B BAG_LR token coefficients, not positional/FM
weights. These units exist in experiments/runs/optimization_v3_research_next_v1/
POSITIONAL_TEXT_PILOT_V1/units/POSITIONAL_TEXT_PILOT_V1_E{67,75}_BAG_LR.joblib;
associated C2B_TFIDF, structured metadata and receipts also exist there.

Before fitting anything, verify exact fit prefix, Train/model/receipt hashes,
class order [0,1], feature assembly and token-column extraction. Map by exact
token identity, never assumed column offsets. Require matching full token sets,
tokenization/TF-IDF settings, vocabulary mapping and IDF values between C2_B
and A02. Verify all baseline A02 row/source/bank bindings and immutable scores.
Missing/incompatible source is a real blocker to Sol; do not refit/approximate.

Let beta_j be the C2_B LR coefficient for A02 token j. Define
a_j=1+abs(beta_j)/(1+abs(beta_j)), in [1,2), and d_j=sqrt(a_j).
Aligned arm: multiply each TF-IDF column by d_j, then row-L2 normalize.
Permutation arm: permute the same d vector across the A02 vocabulary column
order with np.random.default_rng(42).permutation, reset per origin, then use
identical operations. This preserves the weight multiset, not all token-frequency
associations. It is an alignment diagnostic, not a formal randomization null.

Search top32 positive weighted-cosine donors over the FULL original donor bank,
not only the historical top32. Stable ties: source row position. Bank/query IDs
disjoint; every donor cycle<=origin<query cycle. No nearest-neighbor search using
targets. Preserve original bank populations and residual y_bank-p_bank_OOF.

To avoid conflating geometry with correction strength, fix mass to original A02:
M0(q)=sum(original A02 top32 cosine^2), computed from verified cached geometry.
For selected new donors, u_i=new_cosine_i^2; define w_i=u_i*M0/sum(u_i).
Thus sum(w_i)=M0 for aligned and permutation arms. Apply unchanged A02 residual
formula delta=clip(sum(w_i*r_i)/(4+M0),-.05,.05), p=clip(b+delta,0,1).
Use threshold>=.50, original missing/zero-neighbor baseline fallback. M0=0
falls back exactly to original baseline. If M0>0 but new positive support is
unexpectedly zero, fail integrity rather than silent fallback. Positive bounded
column scaling should not create that mismatch. Do not change minimum support,
neighbor count, cap, shrinkage, baseline, residual centering or route.
Report neighbor overlap, weight distribution, correction-cap frequency and
coverage to interpret any gain; they do not select a new policy.

## Boundaries and prospective gates

E67: all preprocessing/coefficient sources fit through67; query68-79,n20903,
W4=68-71,W5=72-75,W6=76-79. E75: sources fit through75; query76-79,n8491.
Same-origin cached A02 is the mandatory comparator. Primary deployment view is
E67 H12; E75 is history sensitivity on overlapping W6, not independent evidence.
Also report pooled W4(E67)+W6(E75),n13441; never mix origin maps or duplicate
overlapping rows into a purported independent denominator.

All for ADVANCE_TO_SOL_REVIEW only:
1. Aligned gain>=.0005 Macro F1 vs A02 on E67 H12 and on E75 W6.
2. Aligned gain>=.0002 vs permuted on E67 H12; E75 W6 gain>=0 vs permuted.
3. Each E67 W4/W5/W6 harm<=.005 vs A02 and vs permutation.
4. Availability, fit-defined text_load and entry_friction groups on every view:
   supported n>=200 and >=30/class; harm<=.01 F1 and<=.02 positive recall
   vs A02 and permutation. Missing baseline predictions remain exact A02.
   Unsupported -> UNKNOWN disclosed; no fabricated estimate. Reuse verified
   previous group definitions and fit-only thresholds, not query quantiles.
5. Full numerical, lineage, label-boundary, budget and independent QA PASS.
Tolerance1e-12. Report all signs, per-cycle counts, confusion/probability metrics
with an explicit epsilon convention. Macro F1 gates cannot be replaced by AUC.
Any failed performance gate closes this exact lane; no post hoc weight rescue.
All labels are repeatedly reused development labels, not a fresh holdout.

## Accounting, technical lifecycle and scope

New namespace a02_coefficient_geometry_v1; card CGEOM_V1. Zero new classifier,
vectorizer or learned preprocessing fits: cached coefficients are reused,
not re-estimated. Nevertheless NEW indexed residual-memory predictors count:
four fits CGEOM_V1_E67_ALIGNED, _E67_PERMUTED, _E75_ALIGNED, _E75_PERMUTED.
Do not call the campaign zero-fit merely because no optimizer is required.

Initial ledger expected292 events,cap117,active114,starts114,saved110,failed4,
pending0; SHA d23c8d365fdf3daae99600ab74a4c02818b0fad27f030fd051a5bdf5c79e8c9f.
Verify before writing. Append cap117->121,pilot76->80 amendment and four unique
new pilot reservations. Other caps unchanged. Active/starts max118 for this
milestone; old three slots remain unused historical headroom, not spending
permission. No retry/refund/reset; release only unstarted IDs on failure.

Commit implementation/protocol/tests -> source-bound temporary synthetic E2E
-> label-free preflight -> freeze -> reserve -> four sequential predictor fits
and durable outputs -> independent pre-label verification -> score once ->
independent metric/gate check -> concise results/receipt -> scoped local commit.
Preflight may sample deterministic query feature rows for bounded-memory full-
bank search timing without query labels. Use sparse/block computation, one job,
two threads,6GiB RSS limit,4GiB available before start and2GiB reserve. No installs.
Estimated implementation+QA+pilot45-120min; actual retrieval runtime uncertain.

Synthetic tests must exercise real pipeline serialization/readback/resume with
all NPZ members loadable allow_pickle=False (explicit Unicode IDs), long IDs,
feature-map mismatch, permuted/aligned identity, all-one weights recovering A02,
mass matching, zero/fallback/ties, full-bank retrieval, no future/self donors,
source-derived residuals, isolated origin maps and gates on every view.
Explicit temporary ledger paths only; production writer blocked in synthetic
mode; verify live bytes unchanged even on injected failure. No patching frozen
historical source or global writer defaults. Current HEAD/document-only changes
must not be confused with fitted historical source receipts; verify both.

Fit/construct state using authorized prefix and historical donor labels only.
Query evaluation labels stay closed until all four outputs are immutable and
independent numerical reconstruction/identity/routing/source checks pass.
Do not use label-bearing cached result arrays as a substitute for boundaries.
Preserve every failure; post-fit repairs that require changing source or
artifacts return to Sol first. Routine implementation choices and synthetic
corrections before freeze are authorized; do not stop for cosmetic approval.

No Test/SampleSubmission/Kaggle, refit full-Train, new EDA/notebook edits,
teammate changes, push, binaries/raw data in Git, or automatic next lane.
Preserve all historical work including FM/B01 failed verdicts and public receipts.
One substantive RESULTS.md plus machine-readable metrics/accounting/verification
is enough. On completion or real blocker, send handoff to KASDAD - TK2 - Sol,
thread01a19569-16e0-7253-aa14-57a0a3160e03,host local, under Sam's explicit
result-return authorization. Include commit, source hashes, fit counts, gains,
negative evidence, support, UNKNOWNs, QA and report paths. Then STOP; no upload.
