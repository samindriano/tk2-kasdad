# Regression Kaggle research log

Exactly four additional Regression uploads were made in this phase (authorization maximum six). Classification was not submitted or optimized. All four have confirmed Kaggle COMPLETE statuses. The original initial Regression manifest and submission are unchanged.

| Candidate | Kaggle ref | Status | Public MAE | Delta vs baseline12.23275 |
|---|---:|---|---:|---:|
| Initial R1-D (prior phase) |57808231|COMPLETE|13.53473|+1.30198|
|hybrid_ridge_inner_v1|57809775|COMPLETE|12.55466|+0.32191|
|svd_hgb_inner_v1|57809802|COMPLETE|12.34519|+0.11244|
|svd128_inner_v1|57810028|COMPLETE|12.38411|+0.15136|
|lgb_text_inner_v1|57810150|COMPLETE|11.71363|-0.51912|

**Best: lgb_text_inner_v1**, public MAE11.71363. It improves the user-supplied official baseline by0.51912 (4.2437%) and meets the<12 stretch goal. It remains a strong provisional candidate, not a permanent final model.

Recipe:12 eligible structured inputs plus signal_stream; fit-only numeric parser/imputer/native categories, TF-IDF word1–2/min_df2/max20k/sublinearTF,32-component randomized SVD(seed42), missing-stream flag, LightGBM regression_l1. Parameter set and raw/rank weight are selected from a past inner fold. Full-Train choice was base31leaves/min_child_samples40/lambda1/400trees/lr.05, rank weight0.50; all50,608 Train rows then fit the final candidate.

Each output has29,421 unique complete IDs in the official SampleSubmission order, finite values clipped0–100, exact CSV hash and approved model manifest. CSVs and model binaries are not committed. Four manifests under `submissions/` here describe the model/data lineage; their approval status is historical and completion evidence lives in `kaggle_log.json`.

The ledger is hash-chained and records durable at-most-once reservations, preflight Kaggle history, submission refs and status transitions. Refresh commands never upload. An ambiguous response is held for reconciliation; it is not retried automatically. The gate counts submissions in both the ledger and Kaggle history and enforces the smaller of phase authorization and current daily availability.

Do not rerun `--execute` to seek another score. Two remaining authorized slots were intentionally not used after the breakthrough. No new candidate, classification change, ensemble or hyperparameter search follows automatically.

SVD128 improved internal scores slightly but public12.38411 was worse than SVD32public12.34519, a concrete validation-to-public counterexample. Public feedback is limited; the user describes roughly30% public and70% private. The split proportion is user-provided and private scores remain unavailable. Do not infer private performance or tune tiny parameter differences to this public subset.

Authentication used the pre-existing Kaggle credential through official CLI2.2.5 in a temporary environment. No credential contents are in this repository or report. Rules-page text was not retrievable through the checked API route; the authenticated API confirmed availability and accepted these submissions.
