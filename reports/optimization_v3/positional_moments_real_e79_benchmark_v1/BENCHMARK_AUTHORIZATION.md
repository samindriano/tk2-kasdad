# POS_MOMENTS_REAL_E79_EXPLORATORY_BENCHMARK

**Status:** authorized benchmark exception; the original
`POS_MOMENTS_V1` scientific verdict remains `PILOT_GATE_FAILED` (45/47 gates).
This benchmark is diagnostic/submission preparation only and cannot qualify,
replace, or promote a champion.

## Authorization and scope

Sam's direct user message in Sol thread `01a19569-16e0-7253-aa14-57a0a3160e03`
(turn `01a1a4a2-4710-7240-a357-359ffbde17b6`, message
`01a1a4a2-53cc-7f63-8e82-b31875a51582`) says: “btw karena submission slot masih
banyak gpp lebih agresif dikit untuk submission. kalo misal ada mau coba
submission yg gamble dari hasil eksperimen gas aja”. Sam separately confirmed
in the active task: “Ya, otorisasi satu benchmark E79”.

Sol approved exactly one `POS_MOMENTS_REAL_E79_EXPLORATORY_BENCHMARK` after
independent review of the POS_MOMENTS pilot. This supersedes the pilot's
prospective no-E79 stop only for this one exploratory benchmark. It does not
change the scientific gates or historical result. Luna prepares one local CSV
and its evidence for Sol's review. Luna does not submit to Kaggle; no public
score is claimed or used to change the recipe.

## Frozen model recipe

- Population: all 50,608 official Classification Train rows, cycles 47–79.
- Reuse only the hash-verified full-Train C2_B structured state, scaler, OHE,
  TF-IDF state, and active `C2B_LR_INTEGRATION_R1` estimator in the A02 E79
  trained freeze. Do not refit or alter any of those components.
- Reuse the full-Train A02 E79 model's verified vectorizer. Build the REAL
  three-channel positional-moment matrix with the frozen C2_B tokenizer and
  the canonical scalar token-occurrence accumulation order:
  `W=[B, B*m1, B*m2]`, with the pilot's exact centering and joint L2
  normalization. No shuffle/control fit, vocabulary change, or feature sweep.
- Use the unchanged C2_B full-Train decision logit as the fixed offset `z0`.
  Fit only one `gamma` from zero by the existing summed logistic loss plus
  `||gamma||^2/2`, `C=1`, L-BFGS-B (`maxiter=2000`, `ftol=1e-12`,
  `gtol=1e-6`, `maxls=50`). No intercept, retry, or rescue.
- After the trained-state/inference freeze is committed, use the existing
  verified A02 E79 Test probability bundle. On present streams:
  `p = 0.5 * expit(z0 + W @ gamma) + 0.5 * p_A02`.
  On missing streams, use the existing full-Train C1_B HGB probability `pP`
  exactly. Hard labels are `p >= 0.50`.
- Reproduce A02, original POS_BAG_OFFSET, and the frozen MIX50 labels for
  identity/disagreement checks only. These comparisons cannot choose or alter
  this fixed recipe.

## Budget, boundaries, and stop

- Exactly one new unique learned fit ID; no retry. Do not revive the released
  `POS_CONTRAST_V1_E79_REAL` reservation. After confirming it remains released,
  add one append-only amendment from global cap 142 to 143 and continuation
  cap 51 to 52; all other category caps stay unchanged. Bind the amendment to
  the exact historical release event and the new pre-fit freeze.
- Perform synthetic fit/persistence/resume/formula checks and a Train-only,
  target-free real preflight before reserving the production slot.
- Commit the recipe/code/tests and pre-fit freeze before the gamma fit. Commit
  the Train-fitted gamma receipt and inference freeze before opening any Test
  feature or probability value.
- After the inference freeze, create one CSV with 29,421 rows, ordered by the
  official Test IDs, columns `record_id,artifact_signal`, and integer labels
  0/1. Do not open SampleSubmission or Test labels.
- Independently rebuild features, logits, probabilities, routes, labels, IDs,
  cycles, and masks. Hash the CSV and compare label-vector identities with
  existing local candidates for duplication only; do not use those differences
  for tuning or selection.
- Keep the CSV and binaries out of Git. Commit scoped code, freezes, manifest,
  QA, and accounting locally; do not push. Send exact CSV path/hash and all
  evidence to Sol, then stop for root review/upload.
