# Root upload confirmation: missing-only temporal threshold

Kaggle submission **58001282** is **COMPLETE**, public Macro F1 **0.83225**. One upload was performed with message `Samuel CLS A02 Missing Train OOF Threshold Benchmark`. Raw response and scoring confirmation are preserved alongside this report.

The prior observed best was 0.83204 (submission 57999920); the observed public difference is **+0.00021**. Original MIX50 scored 0.83144, giving an observed difference of +0.00081. Public differences do not establish private improvement or statistical significance. No final submission selections were changed.

## Exact artifact and independent checks

CSV: `submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/classification_a02_missing_temporal_e79.csv`.
SHA-256: `189a779b6100333a2fecfe443689fe7d340a331e828167221c81281a7ece4a4a`.

Root verified source/input hash pins, canonical manifest digest, frozen probability replay, 29,421 unique ordered IDs, binary labels, and absence of a prior same-file submission. Independent builder and verifier receipts additionally document exact official ID order and duplicate-vector checks. Present-stream labels (28,359 rows) remain unchanged. Among 1,062 missing-stream rows, 18 changed from 1 to 0. Total positives: 6,759.

Policy: present uses `(0.5*pP + 0.5*pA) >= 0.50`; missing uses `pA >= 0.53230976129477`. Threshold selection used frozen causal Train OOF predictions, not Test labels or public scores. This application reused predictions: zero model/preprocessing fits and zero new inference.

## Negative temporal evidence remains negative

D1 selected threshold 0.401662741677 and lost 0.002561 Macro F1 on E67 W5 against the previous missing-pA-at-0.50 policy. D2 selected 0.564553584031 and lost 0.001071 on E75 W6. The pooled unique 15,953-row comparison lost 0.001745. The final threshold was selected on 20,903 adaptive development rows; its same-row gain +0.000369 is selection performance, not independent validation.

The public outcome therefore supports only a new observed public best. It does not turn the historical local verdict into PASS, identify the reason for the discrepancy, validate the optimization policy generally, or authorize nearby public-score threshold searches. See `../THRESHOLD_DIAGNOSTIC.md` for denominators, subgroup/cycle results, and lineage.

## Review boundary

The one authorized upload is complete. No new fit, threshold variant, submission, REG work, or final-slot change was performed. Further research requires a concrete root decision; Luna routes questions and results to Sol while Sam is asleep. No scheduled monitor was created.
