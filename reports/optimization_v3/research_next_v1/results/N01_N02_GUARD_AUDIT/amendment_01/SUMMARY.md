# N01/N02 — zero-fit, all-view subgroup gate audit

## Scope

This amendment re-evaluates cached pilot predictions only. It uses official Classification Train labels for the already-declared temporal evaluation rows, the frozen cutoff-specific fit prefixes for text-load bins, and the existing C16 sparse25 route. It ran **zero official estimator fits** and left the Research Next budget ledger unchanged. The historical N01 and N02 pilot reports and verdicts were not edited.

For every origin and view, `text_load` quartiles are estimated from finite values in that origin's fit prefix, with duplicate quantile edges collapsed and `searchsorted(..., side="right")`. Pooled views retain each row's original origin-specific quartile. A subgroup is supported only when `n >= 200` and each class has at least 30 rows. Supported groups fail the guard if candidate harm exceeds 0.01 Macro F1 or 0.02 positive recall against sparse25. Unsupported groups remain descriptive.

Views audited: pooled primary W1+W4 (9,064 rows), W1 (4,114), W4 (4,950), H12 from cutoff 55 (17,685), H12 from cutoff 67 (20,903), and pooled H12 (38,588). Each variant and matched control is separately compared with sparse25. The N01 class-by-volume versus class-only and N02 sketch versus no-sketch primary contrasts are also retained.

## Results

Macro F1 gains below are versus sparse25. The matched-control gain is shown for the candidate variant only.

| Variant | Primary | W1 | W4 | H12 cutoff 55 | H12 cutoff 67 | Pooled H12 | Matched primary gain | All-view subgroup guard |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| N01 class-by-volume | +0.001203 | +0.002495 | +0.000104 | +0.001608 | -0.001503 | -0.000023 | -0.000731 vs class-only | Pass; original matched gate still fails |
| N01 class-only | +0.001934 | +0.003430 | +0.000666 | +0.002161 | -0.000867 | +0.000596 | — | Pass: 48/48 supported groups |
| N02 sketch | +0.000523 | +0.000788 | +0.000326 | -0.000155 | +0.001443 | +0.000690 | +0.000358 vs no-sketch | **Fail: 2 supported groups** |
| N02 no-sketch | +0.000166 | -0.001216 | +0.001312 | +0.000021 | -0.000943 | -0.000516 | — | Descriptive control |

The N02 sketch candidate meets the primary, H12, W1/W4, and matched-control aggregate thresholds. It nevertheless fails the expanded guard in the highest fit-defined `text_load` bin (coded `3`): at H12 cutoff 67, `n=5,863`, positive-recall harm is **0.02306**; in pooled H12, `n=9,634`, it is **0.02181**. Both exceed the 0.02 limit. Macro F1 harm in those groups is 0.00083 and 0.00068, respectively, below the F1 limit. The subgroup recall failure is sufficient to withhold N02 continuation.

N01 class-only passes the prespecified primary gain (+0.001934), pooled H12 harm, each primary-window harm, and every supported subgroup guard. This is a separate diagnostic branch derived from the already-fitted N01 class-only pilot bundle. It does **not** alter the historical N01 class-by-volume `RETIRE` verdict, whose primary gain over its matched class-only control was -0.000731.

## Gate decision and next authorized work

- **N01 class-by-volume:** preserve historical `RETIRE`; do not reinterpret the result.
- **N01 class-only:** all zero-fit diagnostic gates pass. The already-authorized next step is exactly four additional class-only fits at cutoffs 59, 63, 71, and 75. No class-by-volume or matched-control estimator fits are authorized by this finding. Treat the results as diagnostic-only.
- **N02 sketch:** do not run continuation fits; the supported H12 highest-bin recall guard fails.

## Verification and artifacts

The independent verifier reconstructed routed predictions from the saved N01/N02 component arrays, aligned them to source row IDs and cycles, recomputed confusion-count metrics and all **192** saved group rows (four variants × six views × supported/observed groups), rechecked the frozen fit-only quartile rule, verified hashes and confirmed the ledger was unchanged. Result: `PASS_INDEPENDENT_ZERO_FIT_VERIFICATION`.

- `results.json` — complete six-view metrics, gate decisions, hashes, and source-row provenance.
- `groups.csv` — denominators, class counts, support, metrics, harms, and source-row hashes.
- `verification.json` — independent cache-to-report verification.
- `scripts/optimization_v3_research_next_n01_n02_guard_audit.py` — reproducible zero-fit audit.
- `scripts/optimization_v3_research_next_n01_n02_guard_audit_verify.py` — independent verifier.
- `tests/test_research_next_n01_n02_guard_audit.py` — synthetic guard regression tests.

