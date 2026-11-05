# Sol handoff: R06-native hinge source package R2C

## Review request

Please review the exact R2C source and input package and, only if it passes,
create the immutable `experiments/reg_r06_native_hinge_v1/ROOT_SOURCE_PASS_R2C.json`
with authorization for D1 and D2 only. Do not authorize the final fit in this
source pass. No official R2C fit has started.

The pass must bind `PREFIT_FREEZE_R2C.json` and
`IDENTITY_PREFLIGHT_R2C.json` by SHA-256, copy the exact `source_commit`,
`source_hashes`, and `input_hashes` objects from the freeze, and use:

    experiment_id: REG_R06_NATIVE_HINGE_V1_R2C
    verdict: PASS_SOURCE_AND_INPUTS
    authorized_fit_ids: [R06_HINGE_D1, R06_HINGE_D2]
    final_fit_authorized: false

Include a canonical signature over the approval body. The executor checks this
signature and exact object equality before target access or any fit start.

## Source-ready evidence

- Source commit: `2b7910a7adde6bd38d181b046e2f31e209723528`
- R2C prefit freeze: `4ccc5f6716f40b79506f466be0ab9fde61ae85e0675d9f5970462eb588adffa1`
- Non-target preflight: `0c6550c9817913ad67595365557773ecc06948c9af33d449606a7f0e4ab00ce5`
- Synthetic QA receipt: `3cdd51589d895c18e550f44a82bc87b58e171a4f125f41c331ed5f157c874ac3`
- Source-ready verifier receipt: `162cab6bbc998283b50bc714b18a172591c3339f045761c51d34e9f7877aa13d`
- Target-free R06 inventory: `b7c8179e49ebd63c2785fbbff19834e912a207e54f2831d66e9ff031a9adb3a0`
- Cached label/identity bundle: `e75f70cccd1bb231499060b84155af876173214d984461e39e4c1364e4c5b2bf`
- R06 fit67 / fit75 caches: `4bda0ac6838388ea46ca768e0998f7b0f9a9a1b6defb2b1e5399f6ff83154804` / `61ef463a5bf5ee06f3c68162b8a7f874664b2736161979342731de98b5e09abb`
- R4 D1 / D2 states: `a2fe1297e15566bdc86506458a81de4b7b9f4c6c15359d9fbea4b2818ff3838a` / `9be290f35c89d7fd9199e6838a93a429d5ca5c3049117e4f30506a0b418bc51e`
- R4 D1 / D2 fit inputs: `d3646fe66040f064faa671f9ccc67f67d8c72b83e6192b681c2857a3163c76bd` / `29b1a7d52e9e4f9d502d329071769e260889e81817d82b9e12f73abd0cfc5445`
- Train.csv: `756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082`

The freeze contains the complete source/input hash maps and pins R4 root-pass,
diagnostic replay, ledger, D1/D2 states and fit inputs. The independent source
verifier rechecks R4 cutoff identity and row/cache alignment. Synthetic QA
reports 11 unit tests passing, final-state substitution rejected, and
qualification-fail/exploratory-pass yielding root-review eligibility without
starting FINAL.

R2C records 20,903 unique OOF identity rows, 7,462 D1 score rows, 8,491 D2
score rows, and zero official LP starts. Target values were not read; Test
labels, CSV output, and Kaggle were not accessed. `HINGE_FINAL` coefficients are
excluded from diagnostic gates. Source verification uses the frozen commit as
an ancestor plus exact source hashes, so later receipt commits do not invalidate
the source identity.

## Gate semantics

Both gates compare against the same-cutoff R4 hinge on same-history R06.
Qualification requires pooled gain >= 0.005, each window gain >= 0, and
supported availability/tail harm <= 0.25. Exploratory review requires pooled
gain >= 0, each window harm <= 0.02, and supported group harm <= 0.25. Group
support is n >= 200 and cycle support is n >= 50. All boundaries are inclusive;
unsupported groups/cycles are UNKNOWN and block scientific gate passes.
`root_review_eligible = qualification OR exploratory`.

If diagnostics are later root-review eligible, FINAL still requires the
separate immutable `ROOT_FINAL_FIT_APPROVAL_R2C.json`, bound to the exact
diagnostics SHA, freeze/preflight/source-pass SHAs, source/input hashes, source
commit, and FINAL fit ID. The full-OOF FINAL fit is calibration, not holdout
validation.

## Superseded generations

- R2 freeze `ca9d8b34eb0c09ee3013629195d79a08da3b825e849ce14176ddd4cac74aff2a`
  is preserved and marked superseded because its runner raised `NameError`
  during non-target preflight.
- R2B freeze `0dc45a5edb2de96169056d83f064be842bd42f7612b29d3560402caafea11cd1`
  and its receipts are preserved and marked superseded because its
  `verify-source-ready` CLI routed to the R1 verifier. The direct R2B verifier
  passed, but R2B is not authorized for fits.

Both failures happened before target values or official LPs. R2C has separate
immutable artifact paths.
