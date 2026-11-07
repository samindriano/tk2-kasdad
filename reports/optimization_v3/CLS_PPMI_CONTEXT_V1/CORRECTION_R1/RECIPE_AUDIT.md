# CLS_PPMI_CONTEXT_V1 frozen fit and representation audit — CORRECTION_R1

## Scope and evidence

This is a read-only audit of the already completed four-fit pilot against the
root campaign at
`tk2-simplification-audit-v1/reports/simplification_v1/CLS_DUAL_REPRESENTATION_V1/ROOT_CAMPAIGN.md`.
No labels were reopened, no model was refit, and no historical artifact was
changed. The root correction authorization identifies the original scoring
report as invalid for its intended metric/view contract while authorizing zero
additional fits.

The committed source freeze is bound to signature
`2ff619a7fccd7ae7f49b875c0ab4bc652112f69805164eea3e77df99aae5792b` and the
official Train hash
`875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. The
correction authorization independently pins both prediction archives, the
original result/report/postlabel receipt, and the root MacroF1 anchors. The
four saved runtime artifact hashes in `FIT_SUMMARY.json` match the files still
present under the ignored runtime directory. The fit receipt states that
external frozen source hashes were unchanged.

## Recipe comparison

| Root plan requirement | Frozen implementation / fit receipt | Audit |
|---|---|---|
| Fit only stream-present prefix documents; E67 cycles 47–67, E75 cycles 47–75 | `command_fit` loads each origin with `require_fit_labels=True`; `_fit_one` filters `fit_present`. E67 has 29,705 prefix rows (27,676 present); E75 has 42,117 (39,239 present). | Matches |
| Reuse the verified frozen tokenizer and unigram vocabulary; no new vocabulary/IDF fit | The runner derives no-space terms from the committed A02 vectorizer and uses its frozen preprocessor/tokenizer. Freeze records exact model, vocabulary digest, TF-IDF block replay, C2_B widths, and source/input hashes. | Matches |
| Directed symmetric unit pairs at distances 1 and 2; do not cross documents or OOV gaps | `cooccurrence_counts` adds both directions at fixed distances and rejects an OOV in the intermediate path; documents are accumulated independently. | Matches |
| REAL order and deterministic within-document SHUFFLED order preserving each document’s token multiset | `shuffle_documents` uses one `default_rng(42)` over documents in stable row order and permutes each sequence, including OOV sentinels. The fit receipt logs distinct REAL/SHUFFLED count and PPMI matrices. | Matches |
| Unsmoothed sparse PPMI with `N` equal to the total directed count | `positive_ppmi` uses `log(Mij * matrix.sum() / (rowSum_i * colSum_j))`, clips only positive entries, and returns CSR. | Matches |
| Four prefix-specific `TruncatedSVD(32, random_state=42, n_iter=5)` states; no dense vocabulary square | Each representation receipt logs 5,675 active terms for E67 and 5,723 for E75, 32 components, SVD hash, graph nnz, and source/runtime. No dense V×V path exists in the builder. | Matches |
| Mean known-token embedding by occurrence, append only frozen structured C2_B prefix | The runner excludes OOVs from the mean, routes all-OOV documents to zero, and appends only the structured columns (284 at E67, 370 at E75). The text and missingness blocks are excluded from HGB inputs. | Matches |
| Four identical HGB fits, two numerical threads, fixed blend/thresholds | Ledger order is E67_REAL, E67_SHUFFLED, E75_REAL, E75_SHUFFLED. Every fit uses the frozen HGB parameter dictionary, and present/missing routing uses the predeclared champion blend and thresholds. | Matches |
| No holdout/test fitting; disclose E75-prefix overlap with E67 query years | E75 fit ends at cycle 75, as authorized. Primary scoring uses E67 predictions for cycles 68–75 and E75 predictions for cycles 76–79; the campaign explicitly calls out the E75-prefix/E67-query overlap. | Matches |

The saved REAL/SHUFFLED graph counts and PPMI nonzero counts are respectively
E67 REAL 516,279 / 406,497; E67 SHUFFLED 834,064 / 687,700; E75 REAL 638,946 /
491,671; and E75 SHUFFLED 1,037,384 / 840,624. All four have the required
active dimensions. Artifact hash checks pass for each counts, PPMI, SVD, and
HGB file. Root additionally confirmed that both origins’ champion probabilities
and row IDs match the audited recency cache bit-for-bit.

## What the scoring defects affect

The original helper’s `f1_score` call used sklearn’s binary positive-class
default. The same metric was used in its scorer and verifier, so the old
agreement check was not independent. This affected printed F1 values, subgroup
guards, and metric gates, including fit-quality diagnostics saved after HGB
training. It did not affect the HGB objective, fit inputs, thresholds, blend,
representation matrices, or stored prediction probabilities. The original
`command_evaluate` also parsed all 50,608 `artifact_signal` values even though
only 20,903 unique primary rows were scored. These are reporting/evaluation
and accounting defects; the committed fit recipe and immutable predictions
remain eligible inputs to the authorized zero-fit correction.

The corrected report will calculate MacroF1 as the mean of the class-0 and
class-1 F1 scores, use E67 windows 68–71 / 72–75 / 76–79, calculate protected
subgroup checks across all six views, and gate E75 on REAL-minus-SHUFFLED. No
new family-closure or next-lane conclusion is made here.
