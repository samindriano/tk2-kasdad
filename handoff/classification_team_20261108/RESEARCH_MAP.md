# TP2 Classification research map for teammate Codex

**Reviewed and packaged:** 8 November 2026
**Purpose:** orient a second Codex to Samuel's strongest Classification evidence and prepare a careful comparison with Abhi's reported v3. This is a navigation and interpretation map, not a claim that every experiment is reproducible from this file alone.

## Read this first

The strongest Samuel public result documented here is **Macro F1 0.83225**, Kaggle submission `58001282`, from `classification_a02_missing_temporal_e79.csv`. It is a *decision-policy recomposition* of previously trained predictions: present-stream rows use a 50/50 blend of the positional route and A02 probabilities at threshold 0.50; missing-stream rows use A02 probability with a Train-OOF-selected threshold of `0.53230976129477`. No model was refit for this threshold step. The temporal check for the threshold itself was negative, so the public result must not be retold as a locally validated generalization gain.

Abhi's v3 score **0.83860** is reported by Sam from the current Kaggle page. This map has not inspected Abhi's notebook, probability files, or submission artifacts, and does not independently verify that score. Public scores alone cannot tell whether the two models' errors complement each other or whether a blend would improve Macro F1.

**Important distinction:** `C4_HGB` in Samuel's academic notebook is the explainable individual benchmark, not the current best-performing team submission. C4 is a route between two component models; the 0.83225 pipeline uses later sparse, lexical-neighbor, and positional prediction layers.

## Score and mechanism lineage

| Stage | Observed public Macro F1 | What changed | Evidence and caveat |
|---|---:|---|---|
| `C4_HGB` initial | 0.82641 | Present `signal_stream`: structured + TF-IDF Logistic Regression (`C2_B`); missing/blank stream: structured HistGradientBoosting (`C1_B`); threshold 0.50. | `reports/submissions/KAGGLE_RESULTS.md`. This is the clean coursework baseline, not the best ensemble. |
| Sparse25 | 0.82908 | Add a sparse LightGBM expert at 25% weight on present-stream probabilities, keeping 75% `C2_B`; retain the CatBoost missing-stream expert and 0.50 decision threshold. | `reports/optimization_v3/classification_sparse_full/FULLSIX.md`; score receipt/history in `reports/optimization_v3/measured_benchmark_v1/FINAL_MILESTONE_REPORT.md`. The fixed six-window temporal result gained +0.002297 pooled Macro F1 over its champion and improved all 6 windows; later lanes used the same Train cycles repeatedly. |
| A02 | 0.82930 | Add a past-only lexical-neighbor residual correction to the sparse25 route. Similar past training examples supply a bounded correction to the base prediction; A02 does not mean a new single estimator replacing the sparse route. | Mechanism and local evidence: `reports/optimization_v3/a02_neighbors_v1/FINAL_SUMMARY.md`; verified implementation: `scripts/optimization_v3_a02_neighbors.py`. The later positional comparison records A02 as 0.82930 in `reports/optimization_v3/posoffset_a02_endgame_v1/SOL_FINAL_DECISION.md`. Local A02 pilot evidence was positive on several views but had calibration/log-loss counterevidence and overlapping temporal views; it was not a fresh holdout. |
| Positional bag offset | 0.83123 | Learn a correction from a three-segment positional unigram representation on top of a frozen `C2_B` logit; missing stream falls back to `C1_B`. This adds token-order position signal to bag-of-words evidence. | `reports/optimization_v3/positional_bag_offset_v1/RESULTS.md`, `RECIPE.json`, and `SOL_CLOSE_NEW_FRONTIER.md`; public receipt: `reports/optimization_v3/a02_cgeom_benchmark_v1/KAGGLE_RESULTS_20261104.md`. It missed its frozen E67 H12 minimum (+0.001151 vs required +0.003), so its later public result did not retroactively pass the gate. |
| POS/A02 MIX50 | 0.83144 | Present stream: equal average of original positional-route and original A02 probabilities. Missing stream: positional route exactly. Threshold 0.50. | `reports/optimization_v3/posoffset_a02_endgame_v1/FINAL_HANDOFF.md` and `SOL_FINAL_DECISION.md`. Frozen blend improved two overlapping development views over the positional route; adaptive selection and overlap remain material. |
| MIX50 present / A02 missing | 0.83204 | Keep present-stream MIX50; use A02 probability, rather than positional probability, for missing-stream rows; threshold 0.50. | `reports/optimization_v3/cls_mix50_a02_missing_e79_v1/RESULTS.md` and `SOL_UPLOAD/RESULT.md`. Local unique-row contrast was only +0.000053, with opposite E67/E75 signs; public score improved over the prior observed public best. This is a concrete example of local/public disagreement. |
| Current Samuel high-water mark in the inspected reports: A02 missing threshold | **0.83225** | Keep present MIX50 at 0.50. On missing rows only, apply A02 probability threshold `0.53230976129477`, selected from frozen causal Train OOF predictions. No classifier or preprocessing was refit. | Exact public confirmation and CSV hash: `reports/optimization_v3/cls_a02_missing_temporal_decision_v2/SOL_UPLOAD/RESULT.md`. The selector report is `THRESHOLD_DIAGNOSTIC.md`. It reports W5 harm −0.002561, W6 harm −0.001071, pooled unique W5+W6 harm −0.001745 against the prior missing-`pA`-at-0.50 policy. The +0.000369 same-row selection-set gain is adaptive, not independent validation. Public score is +0.00021 over 0.83204; no private-score claim. |

The last row is **not simply “POS/A02 learned model = 0.83225.”** It is POS/A02 probability blending for present rows plus a threshold-only missing-stream policy. The local threshold pilot chose its cutoff using 20,903 already-used development rows; it worsened W5/W6. Keep the frozen local verdict and observed public result side by side.

## Negative evidence and important near-misses

Use these as counterexamples before proposing a new blend or another “text model.” A frozen-gate failure is not always proof that a mechanism contains no signal; distinguish outright predictive losses from a candidate that was positive but failed a risk/consistency rule.

| Lane | What the evidence says | Read |
|---|---|---|
| C16 / C17 / C11 | Public scores recorded in the measured benchmark were 0.82844 / 0.82825 / 0.82739, below Sparse25. C16's later continuation had primary gain +0.000304 and H12 gain +0.002837, but did not qualify. | Small/unstable candidates; don't rank on H12 alone. Start with `final_continuation/CAMPAIGN_SUMMARY.md`, `final_continuation/C16/` and `measured_benchmark_v1/FINAL_MILESTONE_REPORT.md`. |
| C19 XGBoost + SVD32 | Full-six +0.000667 F1 and H12 +0.004152 vs Sparse25; AUC/AP/proper-loss improved, but W6 was −0.002599, only 3/6 windows nonnegative, and W2/Q3 exceeded an F1 harm guard. Public 0.82760. | Real ranking information may have improved while the hard-label decision did not transfer consistently. `classification_next_v1/FINAL_SUMMARY.md`, `DECISION.md`. |
| Class-only weighting | Internal full-six F1 +0.000687; public 0.82774, below Sparse25. AUC/AP/log-loss did not corroborate a broad ranking improvement. | Likely threshold/allocation trade-off, not evidence of robust new ranking information. `classification_next_v1/DECISION.md`, `FINAL_SUMMARY.md`. |
| Recency weighting A01 | Primary −0.000809 vs Sparse25; pooled H12 +0.000849, with mixed W1/W4. | Positive later horizon did not pass the primary gate; do not generalize a late gain to Train→Test. `classification_recency_a01/FINAL_SUMMARY.md`. |
| Scalar TTR tree | Exact token count plus TTR did not beat the matched count-only control: primary −0.000686 F1; H12 −0.000116. | The EDA association remains valid as an association, but scalar TTR did not add reliable predictive value to this specific text-rich tree recipe. `classification_next_v1/FINAL_SUMMARY.md`. |
| Position moments / bigram / trigram / contrast | These were tested as distinct positional summaries or n-gram transforms; public submissions in the user-provided history varied and did not replace the 0.83225 reference. | Browse the short reports before proposing a renamed variant: `positional_moments_real_e79_benchmark_v1/`, `bigram_position_real_e79_v1/`, `c10_trigram_e79_exploratory_v1/`, `contrast_real_e79_exploratory_v1/`. |
| PAIR/scalar meta-head | Both heads lost materially to fixed MIX50 on pooled W5/W6: −0.007629 / −0.006237 F1; PAIR public 0.82598. | Strong negative evidence for this learned combination head, not for all possible probability blends. `cls_pos_a02_meta_v1/FINAL_REVIEW_R1.md`, `SOL_UPLOAD/RESULT.md`. |
| FM, learned pairs | V1 stopped at optimizer non-convergence before opening query labels: inconclusive. V2's exact fitted recipe was much worse than the no-pair and A02 baselines on both E67 H12 and E75 W6, with every per-cycle gain nonpositive. | V1 is an optimization failure, V2 is negative predictive evidence for the tested recipe; neither proves every FM impossible. Reports are in the branch-only worktree path below. |
| Coefficient geometry / CGEOM | Small E67 gains reversed on E75 W6 (−0.000536 vs A02); the E79 CGEOM public score was 0.82884. | History-sensitive, weak benefit; no evidence to select a post-hoc blend based on geometry. Main report: `a02_cgeom_benchmark_v1/`; additional review in the branch-only path below. |
| PPMI context | On E67 W6, real was +0.001997 vs champion while shuffled was −0.006669; on E75 W6, real −0.002638 and shuffled −0.002502. Some supported text-load groups had material harm. | The “real beats shuffled” contrast was mostly degradation of a geometry-sensitive shuffled control, not stable gain over champion. No-go for this exact PPMI32/HGB lane. Branch-only report below. |
| Sparse MLP | Primary −0.001698 vs Sparse25; H12 at origin 67 −0.005971; several Q2 subgroup harms. | The tested sparse one-hidden-layer MLP is not evidence against every neural method, and it was explicitly not the full DAN previously considered. `classification_mlp_v1/FINAL_SUMMARY.md` and `DECISION.md`. |
| NB-SVM / SVC / word-char | Cached diagnostics show NB-SVM primary −0.0027 F1; SVC and word-char were also negative on primary/H12. | Avoid reopening under a renamed implementation absent a distinct mechanism. `classification_diagnostics/FINDINGS.md`. |

### Branch-only evidence (not merged into the inspected TP2 checkout)

The following summaries were read in `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1`, observed on branch `codex/reg-position-failure-mechanism-v1` at `827a7c5`. That worktree had unrelated uncommitted Regression changes. Treat these paths as read-only, branch-specific evidence; verify ancestry/status before copying code or citing it as part of current `TP 2` HEAD.

- FM V1 failure / V2 final review: `reports/optimization_v3/learned_pair_pilot_v1/FAILURE_DIAGNOSTIC.md`, `learned_pair_pilot_v2/SOL_FINAL_REVIEW.md`.
- PPMI: `reports/optimization_v3/CLS_PPMI_CONTEXT_V1/SOURCE_MECHANISM_HANDOFF.md` (read-only no-go synthesis) and `PROTOCOL.md` for the frozen representation.
- Sparse MLP: `reports/optimization_v3/classification_mlp_v1/FINAL_SUMMARY.md` and `DECISION.md`.
- Coefficient geometry: `reports/optimization_v3/a02_coefficient_geometry_v1/SOL_FINAL_REVIEW.md`.

A later full DAN pilot **was run** on another research branch. Read
`codex/simplification-audit-v1` at commit `711a5b9b758dea5c3d8f07100822d6371ef323a3`,
`reports/simplification_v1/CLS_DAN_V1/PROTOCOL.md` and
`EVALUATION_RECOVERY_R2_RESULTS.json`. It tested a learned 32-dimensional mean
unigram embedding with 64/32 nonlinear layers against a matched LINEAR head.
The saved R2 results report FAIL for mechanism, original scientific, amended
scientific, and exploratory eligibility rules; identity/metric QA passed.
This is negative evidence for that tested recipe. Distinguish it from the earlier
sparse MLP and preserve its evaluation-recovery provenance. Do not reopen DAN on
an assumption that it was never tried.

## What to request from Abhi before a blend test

Request a frozen prediction handoff, not just a Kaggle CSV or a public score:

1. The v3 recipe, training/source commit, feature/preprocessing fit boundaries, missing-stream handling, threshold, and any ensemble weights.
2. **Raw positive-class probabilities** with `record_id`, `cycle_code`, and stream-availability flag for temporal predictions made without using each query row's label. Include fit origin/cutoff and hashes/manifests. Hard labels alone cannot support a probability blend or calibration analysis.
3. The same information for official Test IDs 29,421 in exact SampleSubmission order, plus a schema/finite-value check. Test probabilities are only for a final predeclared policy after validation; Test has no labels for selection.
4. The exact temporal validation IDs (and which were actually excluded from training for each prediction). If Abhi cannot provide same-row, prequential predictions, the local blend comparison is not identified and should stop until such artifacts can be generated correctly.

Do not infer complementarity from `.83225` and `.83860` alone. The former is a public-subset Macro F1 and the latter is user-reported; neither reveals the overlap of their errors.

## Minimal, safe blend comparison

This is a proposal for the next decision, **not authorization to fit or submit**.

- First verify identical prediction IDs, cycle codes, origin, and availability mask. Compare all solo and blended predictions on their exact common rows; report any dropped-row count and never silently fill missing probabilities. Use the same denominator for Samuel, Abhi, and each fixed blend.
- Prefer the previously defined stitched unique Train-development rows: E67-origin predictions for cycles 68–75 (12,412 rows) plus E75-origin predictions for cycles 76–79 (8,491 rows), total 20,903. E75 W6 is a subset of E67 H12; do not count it as independent replication or add it twice. If the partner only has full-Train in-sample predictions for these rows, they are invalid for validation.
- Start with **one predeclared 50/50 probability blend** of the best frozen Samuel and Abhi probabilities, compared against both individual models and, only if compatible saved predictions are available, historical Sparse25/C4 controls on the same rows. The exported packet includes POS, A02 and final Samuel, not Sparse25/C4 probabilities; do not silently substitute in-sample controls. Do not search blend weights or use public leaderboard outcomes to choose them. Report pooled Macro F1 from pooled confusion counts, per-window/per-cycle effects, supported missing/present and text-load groups, and Brier/log-loss as calibration diagnostics.
- Keep the decision threshold fixed at 0.50 for this first comparison. The existing `0.53230976129477` applies only to the old A02 probabilities on missing rows; it must not be copied to a changed blend. If thresholding is later examined, fit one threshold from inner, past-only Train OOF predictions before looking at the outer query labels, freeze it, and report the fixed-0.50 result alongside it. Threshold selection on the same rows used to report performance is not independent evidence.
- A blend must improve over the stronger solo model on the same denominator and not hide meaningful window/subgroup losses. If it only wins against C4 or Sparse25 but loses against Abhi v3, it is not a Kaggle candidate. Public scores may be reported descriptively after a single already-approved upload; do not use repeated public-score probing as a search loop.
- If raw partner validation probabilities do not exist and cannot be produced from a reproducible recipe with exact temporal exclusion, the correct result is `NOT EVALUABLE`, not an estimate from different row sets or public-score arithmetic.

The development cycles have been revisited by many experiments. The stitched set is **not fresh holdout data**. A strong result here supports one controlled benchmark decision, not an unbiased estimate of leaderboard gain or a promise of 0.84+.

## Layered reading path for the next Codex

**Layer 1 — 15-minute orientation:** read this map; then `reports/optimization_v3/posoffset_a02_endgame_v1/SOL_FINAL_DECISION.md`, `reports/optimization_v3/cls_a02_missing_temporal_decision_v2/SOL_UPLOAD/RESULT.md`, and `reports/optimization_v3/a02_neighbors_v1/FINAL_SUMMARY.md`.

**Layer 2 — reproduce the model lineage:** read `reports/optimization_v3/classification_sparse_full/FULLSIX.md`, `reports/optimization_v3/positional_bag_offset_v1/RECIPE.json`, `reports/optimization_v3/positional_bag_offset_v1/RESULTS.md`, `reports/optimization_v3/cls_mix50_a02_missing_e79_v1/SOL_UPLOAD/RESULT.md`, `reports/optimization_v3/cls_a02_missing_temporal_decision_v2/THRESHOLD_DIAGNOSTIC.md`; inspect only relevant functions in `scripts/optimization_v3_a02_neighbors.py`, `scripts/optimization_v3_positional_bag_offset_v1.py`, and `scripts/optimization_v3_cls_a02_missing_temporal_decision_v2.py`.

**Layer 3 — challenge the proposed blend:** read `reports/optimization_v3/classification_next_v1/FINAL_SUMMARY.md` and `DECISION.md`; `classification_recency_a01/FINAL_SUMMARY.md`; `cls_pos_a02_meta_v1/FINAL_REVIEW_R1.md`; `a02_cgeom_benchmark_v1/BATCH_REPORT.md`; and, only if needed, the branch-only FM/PPMI/MLP/geometry Markdown paths above.

Avoid bulk-reading `experiments/runs`, large JSON/NPZ bundles, raw CSVs, or every historical log. Most prediction bundles are local/ignored and tied to frozen hashes; a missing bundle is a reproducibility blocker to report, not a reason to refit silently.

## Code entry points and access to research branches

The root repository now publishes these branch snapshots separately, without
merging research worktrees or changing their local files:

- `codex/simplification-audit-v1`: `711a5b9b758dea5c3d8f07100822d6371ef323a3`
  (simplification, POS-context, later DAN).
- `codex/reg-position-failure-mechanism-v1`: `827a7c5bab27639de70ef83d18ff0f44c1446664`
  (including PPMI context evidence).

After fetching those branches, read a report without changing your checkout:

```console
git show origin/codex/simplification-audit-v1:reports/simplification_v1/CLS_DAN_V1/PROTOCOL.md
git show origin/codex/reg-position-failure-mechanism-v1:reports/optimization_v3/CLS_PPMI_CONTEXT_V1/SOURCE_MECHANISM_HANDOFF.md
```

Core source in `main`, ordered from final decision back to components:

1. `scripts/optimization_v3_cls_a02_missing_temporal_decision_v2.py`: final
   missing routing and threshold application, zero-fit.
2. `scripts/optimization_v3_a02_cgeom_benchmark_v1.py`: saved E79 component
   training/application lineage; includes alternatives not selected in champion.
3. `scripts/optimization_v3_positional_bag_offset_v1.py` and
   `src/models/positional_bag_offset_v1.py`: original bag-offset positional learner.
4. `scripts/optimization_v3_a02_continuation.py`,
   `scripts/optimization_v3_a02_neighbors.py`,
   `src/models/classification_a02_neighbors.py`: original A02 neighbor mechanism.
5. `src/models/optimization_v3_sparse_classification.py` and
   `src/models/classification_models.py`: sparse and structured/text base models.

These are historical runners with frozen inputs and receipts, not a standalone
one-command retraining package. Use the committed probability exports first.
Inspect recipe constants and call graph before porting a component into a fresh
teammate experiment. Do not bypass old freeze/receipt guards or overwrite runs.

## Scope and verification note

The inspected source checkout was branch `codex/cls-simple-present-threshold-v1`,
HEAD `ec96030` before the handoff commit. Existing unrelated notebook edits were
preserved. The map worker read reports only. Root additionally verified and
exported saved probability arrays, without new fits, inference, metric selection,
Kaggle access, or raw-label exports. The source manifest and tests accompany the
packet. GitHub publication is verified separately in the final handoff message.
