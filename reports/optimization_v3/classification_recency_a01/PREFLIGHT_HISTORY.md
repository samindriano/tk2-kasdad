# A01 synthetic preflight history

`synthetic_preflight.json` is the preserved V1 attempt. It is superseded: V1 used a TF-IDF configuration that did not match the frozen Classification recipe and did not cover the complete interruption/resume contract.

`synthetic_preflight_v2.json` is the current pre-official-run preflight. Its independent verifier passed preprocessing parity, score-label perturbation invariance, route/weight reconstruction, and resume after model, bundle, and report publication failures without extra fits. The V2 synthetic scientific gate itself fails at H12_67 and W4; that synthetic result is diagnostic only and is not an official-data verdict.

The two `synthetic_preflight_v2_failed_attempt_00*` reports and run artifacts are retained as implementation history. They are not the final V2 receipt.
