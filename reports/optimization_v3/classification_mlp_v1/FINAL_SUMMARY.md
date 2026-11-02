# Classification sparse MLP pilot — hasil final

## Status

- **Checkpoint:** `COMPLETE_REVIEW_REQUIRED`.
- **Verdict pada gate yang dibekukan:** `NO_GO_GATE_FAILED`.
- **Model fits:** dua planned temporal origins selesai; cutoff 55 tetap berstatus historis `FIT_FAILED` karena serializer gagal setelah model tersimpan. Output cutoff 55 dipulihkan tanpa refit. Cutoff 67 berstatus `FIT_SAVED`.
- **Independent verification:** `PASS`; verifier menghitung ulang routing, views, subgroup, per-cycle metrics, gate, input identities, dan ledger.
- **Boundaries:** tidak ada Test, SampleSubmission, full-Train refit, CSV submission, atau Kaggle access.

## Recipe yang dipertahankan

Exact C2_B sparse main effects dengan preprocessing fit-only dan TF-IDF unigram/bigram; MLPClassifier hidden layer `[32]`, ReLU/Adam, alpha `0.01`, batch `256`, learning rate `0.001`, seed `42`, 30 epoch tanpa early stopping. Route present-stream tetap `.75 × sparse25 + .25 × MLP`; missing-stream memakai frozen CB; threshold `0.50`. Tidak ada recipe atau gate yang diubah.

Freeze signature `46d9362cdf543b87af58653fe1666d79cbe13b4d012397d5f730b15959debd18`; protocol SHA-256 `d110156b85b115857218c040d1dc88d8d249c3824310be9f05d58d9ed9906042`. Official Classification Train: 50.608 baris, SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.

## Hasil temporal

| View | n | MLP route Macro F1 | sparse25 Macro F1 | Δ vs sparse25 | Matched-control Macro F1 | Δ vs matched |
|---|---:|---:|---:|---:|---:|---:|
| Primary (W1+W4) | 9.064 | 0,822275 | 0,823972 | −0,001698 | 0,824138 | −0,001863 |
| W1 | 4.114 | 0,803409 | 0,803160 | +0,000249 | 0,801944 | +0,001465 |
| W4 | 4.950 | 0,838049 | 0,841418 | −0,003369 | 0,842730 | −0,004681 |
| H12 cutoff 55 | 17.685 | 0,799878 | 0,795032 | +0,004846 | 0,795053 | +0,004825 |
| H12 cutoff 67 | 20.903 | 0,802990 | 0,808961 | −0,005971 | 0,808066 | −0,005076 |
| H12 pooled | 38.588 | 0,802135 | 0,802966 | −0,000830 | 0,802475 | −0,000340 |

The positive cutoff-55 horizon did not persist at cutoff 67. Primary failed both required improvement checks; W4 exceeded the permitted harm; pooled H12 was negative. Three supported text-load Q2 groups exceeded the frozen 0.01 Macro F1 harm limit. All 48 subgroup cells had adequate support; the separate recall guard passed. The gate therefore remains NO-GO.

## Fit and recovery ledger

| Cutoff | Fit population | Score population | Estimator state | Ledger/report status |
|---|---:|---:|---|---|
| 55 | 12.020 | 17.685 | durable before the serialization error; recovered from the saved joblib | `FIT_FAILED_OUTPUT_RECOVERED_NO_REFIT` |
| 67 | 29.705 | 20.903 | one fit, exactly 30 epochs | `FIT_SAVED` |

The original cutoff-55 `FIT_FAILED` event was preserved. Its `fitted_state.joblib` SHA-256 is `e61dffb1f219c15e9680baca87ba73e804e76f73faa0c3a9df36ab778da720a8`; no second cutoff-55 estimator fit occurred. Cutoff 67 produced a ConvergenceWarning at the protocol's fixed 30-epoch maximum; convergence is not certified.

Two post-fit technical issues were handled transparently: the report finalizer referenced a nonexistent release-helper alias, then the independent verifier exposed a legacy CatBoost sidecar schema mismatch; the schema adapter encountered one recursive dispatch error on its first retry. These were corrected in isolated runtime/report code, recorded in post-fit amendments, and completed with an independent `PASS`. No estimator fit or ledger reservation was added by these corrections. The original training runner, frozen verifier, recipe, historical `FIT_FAILED` event, and result values remain intact.

## Budget snapshot

At the end of MLP finalization: phase cap `60`; gross reservations `54`; active reservations `50`; started/consumed fits `50`; saved `48`; failed/interrupted starts `2`; unused release total `4`; remaining unreserved cap `6`. MLP used its two original slots and released `0` unused slots. Original category caps remain pilot `26`, continuation `24`, control replay `6`, technical retry `4`; the ledger contains no category-cap amendment yet.

## Reproducibility and evidence

Run the recovery-only verification (no fit) with:

```powershell
python -m scripts.optimization_v3_classification_sparse_mlp_v1_finalize_adapter
```

It reuses saved estimators and bundles, recomputes metrics, checks the frozen gate and ledger, and writes the finalization adapter receipt. The outcome-aware sidecar and verifier resume amendments document the post-fit corrections.

Primary evidence: `results.json`, `independent_verification.json`, `budget_snapshot.json`, `POSTFIT_SERIALIZATION_RECOVERY.json`, `FINALIZATION_ADAPTER.json`, `POSTFIT_VERIFIER_CORRECTION.json`, and `POSTFIT_VERIFIER_RESUME_AMENDMENT.json` in this directory. The estimator joblibs and prediction bundles remain in the ignored `experiments/runs/optimization_v3_classification_mlp_v1/` directory.
