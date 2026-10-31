# Optimization V3 — live research checkpoint

This is an intermediate campaign checkpoint, not a stopping decision. Initial HEAD: `0be25dd`. All 758 preexisting tracked files are protected byte-for-byte; official EDA, academic notebooks, teammate work and old accepted evidence remain untouched.

## Current observed competition result

The first V3 Classification challenger is **COMPLETE**, reference **57837821**, public Macro F1 **0.82908**. This improves the V3 starting champion **0.82701** by **0.00207**. The authenticated public board then returned `kasdud lovers` in rank **1 of 4**, with 7 daily slots remaining and 4 team submissions in total. This is dated public feedback, not proof of private performance.

The challenger uses the old C2_B and a new raw-TF-IDF sparse LightGBM binary model in a fixed **75% / 25% probability blend on present streams**, while retaining the original CatBoost missing expert exactly. Threshold remains 0.50. All three components were freshly fitted on 50,608 Train rows; the 29,421-row Test CSV passed ID/schema/order/classes and source/hash gates. Exactly 310 labels changed, all on present streams; missing labels stayed unchanged. Frozen training source commit: `75ff5bd`. Full manifest and accepted API receipt are under `kaggle/`.

| Model | Primary Macro F1 | H12 Macro F1 | Public |
|---|---:|---:|---:|
| Initial V3 Classification champion | 0.827646 | 0.798984 | 0.82701 |
| Fixed sparse-LGB present blend25 | 0.829944 | 0.802966 | 0.82908 |

The candidate improves all six primary windows, AP and Brier; the missing branch is exactly equal. One counterexample is H12 high-text-volume Q4, with F1 delta −0.002279. The current initial Regression champion remains at public MAE **11.35970**, reference57835122; V3 Regression candidates are still under development.

## Diagnostics and predeclared research stages

- Classification: raw structured CatBoost is too weak to replace C2_B on present streams. It repairs 1,979 C2_B errors but breaks 4,374 correct cases. Its score is only an explicit ablation/input, not assumed useful. C2_B probability residuals change sign nonlinearly across confidence bands. See `classification_diagnostics/FINDINGS.md` and `audit.json`.
- Regression: the three-component incumbent reproduces primary MAE11.559390 and H12 MAE14.047533. Early H12 mean bias is +1.333, late H12 −1.296; tail over/underprediction persists. These are descriptive reused outcomes, not target-based routes. See `regression_diagnostics/FINDINGS.md` and `diagnostics.json`.
- Stage1 representation screen: sparse binary LightGBM passed, completed six-window validation, and received the first external check described above.
- Stage2 conditional correction: nonlinear shallow OOF heads use base forecasts made before their labels; the Classification head has a base+structured control and a weak-CatBoost-score ablation. Regression uses exact incumbent OOF blocks56..E and leaves the earliest cutoff on unchanged fallback, avoiding the outcome-informed historical inner mapping at52..55.
- Stage3 alternative Regression information: fixed ten-bin conditional-distribution/CDF median, and a character-only text Ridge channel. These differ from old SVD-dimension/weight grids.
- Later screen prepared: ordered supervised token summaries. Each training row uses only labels from strictly earlier cycles, with same-cycle labels withheld until the whole cycle is encoded. Synthetic perturbation tests verify that changing a cycle's labels cannot change its own or earlier features. No outer/Test labels are used for dictionaries.

## Regression pilot findings so far

Pilot primary is only W1+W4, 9,064 rows; it must not be confused with the six-window primary. H12 uses 38,588 rows. Deltas below are MAE candidate minus the V3 incumbent; negative is better.

| Fixed candidate policy | Pilot primary delta | H12 delta | Continue? |
|---|---:|---:|---|
| Ten-bin CDF median alone | +0.733931 | +0.827221 | No |
| 25% CDF median +75% incumbent | +0.005787 | +0.024628 | No |
| Character Ridge rank50 alone | +0.241089 | +0.117908 | No |
| 25% character-Ridge rank50 +75% incumbent | −0.098709 | −0.130071 | Advanced; six-window result fails below |

The distributional model's linear interpolation inside fixed bins assumes an approximate uniform conditional density within each ten-point bin. Its failure does not identify the true target generator. Both distributional and character pilots passed independent fingerprint, rank/CDF/mixture and metric reconciliation. No Regression upload has occurred in V3 yet.

The character continuation now reproduces **primary MAE11.717761 / H12 MAE13.917462**. Only two of six windows improve; primary worsens0.158371. There are57 subgroup comparisons above0.25 MAE harm among365 supported comparisons, with worst harm0.837118. The H12 improvement does not justify promotion or a Kaggle upload. MAIN opened the figure and independently repeated all14,478 numerical checks against the preserved verification receipt. Its exclusive receipt writer correctly rejects overwriting; the second audit compared the in-memory result instead.

The Regression OOF residual head also fails its pilot: pooled W1+W4 MAE13.551210 and H12 MAE14.128414, both worse than the unchanged incumbent. W1/H12_E stay exact fallback; only the cutoff67 head was trained on actual prior OOF blocks56–67. This family is retired without further fits or upload. The active verifier is `optimization_v3_verify_reg_residual_v5.py`; versions1–4 and their amendments preserve the original failures. V5 fixes only verifier fixtures/schema/positive-and-negative flag assertions, with a pre-check amendment and zero refits. Source, model outputs and metrics were not altered to obtain PASS.

## Validation and reproduction boundaries

Primary six fits end55/59/63/67/71/75 and predict the next four cycles. H12_E/L use fixed origins55/67. H12 reuses known Train outcomes; every conclusion remains adaptive development evidence. All learned vocabularies, IDF, SVD, imputers, encoders and estimators are learned inside their declared temporal fit. Nonlinear correction/meta-model labels are restricted further to genuinely past OOF blocks.

The sparse-classification continuation's original verifier omitted one cutoff field. The original source/freeze was preserved, and a separately frozen one-field adapter completed 3,718 independent metric-field checks without new fits. Other ongoing verifier failures and report-only amendments are kept as evidence; only passing active verifiers can advance a candidate. Verification re-fits are labelled separately from new candidate configurations/fits.

The existing research environment is reused. Raw datasets, prediction arrays, model binaries and CSV submissions stay ignored. JSON contracts bind exact bytes; source hashes are normalized LF where specified. Some independent frozen worker receipts bind literal source bytes; for a fresh checkout use `git -c core.autocrlf=false clone ...` or an equivalent LF-preserving checkout before replaying those receipts. Git commits are contextual; actual frozen source/input hashes govern reuse.

Further work continues on the independent lanes. Public rank1 or the first positive result is not this campaign's stopping criterion. No academic notebook, ZIP or SCeLE action is part of V3.

The accepted Class candidate additionally passed a fresh raw-data replay: three new full-Train fits, without reading ignored prediction caches, old CSVs or model binaries. Each component probability, final probability/label and CSV hash is identical to the accepted manifest (100.1 seconds). Reproduce with `python scripts/optimization_v3_replay_classification.py --output experiments/runs/optimization_v3_replay/new.csv --receipt experiments/runs/optimization_v3_replay/new.json`; paths must be new. No API call is made. Dependencies reuse `experiments/optimization_v2/requirements.txt`.

Before freezing the later supervised pilot, token aggregation was made canonical by sorting eligible tokens. A test runs separate processes with different `PYTHONHASHSEED` values and requires exact training/inference feature hashes. This prefit correction prevents tiny reduction-order differences; no pilot outcomes had been generated.
