# R16 E79 legacy-component trace

**Status:** `NOTFOUND_SCOPELIMITED` for the exact E79 mapped Ridge `.1875`
and sparse rank50 `.25` component vectors. Their independent **raw** full-Train
Test predictions are present, but the exact rank-mapped arrays used by the
accepted blend are not stored in the inspected main artifacts. No mapping,
prediction, fit, Test-label access, CSV build, or Kaggle action was performed.

## What is present

The accepted full-Train baseline manifest is
`reports/optimization_v2/kaggle/v2_reg_sparse_lgb25.json` (SHA-256
`dd166e92089f24978244496a9a0afabc0d06551dbdc326579a7c898a8844e19a`). Its
recipe records three components:

| Component | Blend weight | Rank weight | Fit rows | Manifest state SHA-256 | Manifest raw-prediction SHA-256 |
|---|---:|---:|---:|---|---|
| SVD32 LightGBM | .5625 | .50 | 50,608 | `e6e4a20b6c9287f38f8ccec0691105a1eec4492284f779aed473adbaf219e566` | `8cc839f49b2ac22d9d6bae1ba6922c9e53b8ea51185437ab4ff6df5c42deff34` |
| Hybrid Ridge | .1875 | .75 | 50,608 | `be01a51f3d15cc8c8d16191d51e1e7a2072d73574bfa5dd9635cf1b5bea0cd77` | `204d5c343b4ef5392c9d8c7e224ef3d9161d63e47fb24615211459e0458e7475` |
| Sparse LightGBM L1 | .25 | .50 | 50,608 | `c99c3ab5640613d1b3de9f9d93212391b2866e9be33eed99644f535002afa5ca` | `66c3c631c7785c53c5946980b459b06e197e0e59b9baf9f2dfc231296cdd6c60` |

The manifest records source commit `3e86b173f035afca0bbc636964e367ae21b32c27`,
Train SHA-256 `756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082`,
Test SHA-256 `4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c`,
and ordered Test ID SHA-256
`0e325676d1e91ac8d594390a6fa44bee70270da489cdcb323b6fc8bff66f01ea`.
Its component audits state that each fit used 50,608 official Train rows and
learned only from Train.

The separately stored component archives are:

- Ridge raw predictions: `experiments/runs/optimization_v2_inference/cd4c7c42742489ca98f3cab7bacf8df9654b9941372ac28eb2362911bb7b57c0.npz`
  (file SHA-256 `349ca6fb853bce5e70a4efe394900bcab6bed17fccc1b3426e058de38a3058cb`).
  Sidecar: `cd4c7c42742489ca98f3cab7bacf8df9654b9941372ac28eb2362911bb7b57c0.json`
  (SHA-256 `c3d2d000e7acb9e0d9d007d68f17253f971a1bcaa59234125fca7a0c787c86aa`).
  Its only array is `prediction`, shape `(29421,)`, `float64`; its array digest
  equals the manifest Ridge prediction digest. Values range from `-3.8170` to
  `131.4383`, consistent with raw model scores rather than rank-mapped `[0,100]`
  values.
- Sparse LGB raw predictions: `experiments/runs/optimization_v2_inference/ccb4676f6113a43ece08a4d0752192d2c9fbe603024851944cdf4b1785fe9fa7.npz`
  (file SHA-256 `2c321ef1ab5b9070557478efe41ed1be31b1120b80b723408c559888a74e4cae`).
  Sidecar: `ccb4676f6113a43ece08a4d0752192d2c9fbe603024851944cdf4b1785fe9fa7.json`
  (SHA-256 `450d7d0651a5991802a5a0efc1b256e28d5d08e9447fa9aa5da427dda8e63f46`).
  Its only array is `prediction`, shape `(29421,)`, `float64`; its array digest
  equals the manifest sparse prediction digest. Values range from `-3.3850` to
  `105.0359`, also raw rather than rank-mapped.

Both arrays align to the manifest's 29,421-row official Test population by the
manifested ID sequence and exact component digests. Their component sidecars
are present, but the archives contain no mapped Ridge or mapped sparse array.

## R06 composite check

`experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz`
(file SHA-256 `ce034e82a7063ec5dd873b1126d60731dd291d3852c57936c7aa1ee449a81eba`)
contains only `champion`, `raw_candidate`, `rank_mapped_candidate`, `cycle_code`,
and final `prediction`. Its `champion` vector digest is
`57097f6be90301a87cba673482e23a75afb764261d10ec18f8a46f9cf4138cf1`, exactly
matching `experiments/runs/optimization_v2_inference/v2_reg_sparse_lgb25.npz`'s
single `prediction` array (`np.array_equal` was true). This confirms the R06
artifact reuses the accepted V2 **composite**; it does not expose the composite's
Ridge and sparse contributions separately. The saved `rank_mapped_candidate`
belongs to R06's character model, not either legacy component.

The dual-campaign standalone Ridge25 and Ridge50 Test archives were also checked
as possible lookalikes. Their prediction digests (`f709cd7c…` and
`baee5581…`) do not match the accepted V2 Ridge raw digest, so they are not
substitutes for this champion component.

Other inspected prediction bundles do not supply these E79 Test components:
`experiments/runs/dual_reg_v1/stage1_paired_blends.npz` stores rank/blend arrays
for Train evaluation row indices at cutoffs 55, 59, 63, 67, 71, and 75;
`experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz`
stores raw SGD and sparse-LGB predictions on Train evaluation rows. They are
not the official Test ID sequence.

## R16 checkpoint and stop

The R16 pilot artifacts exist at
`experiments/runs/optimization_v3_batch1/R16/R16_fit55.npz` and `R16_fit67.npz`.
`reports/optimization_v3/batch1/R16/results.json` (SHA-256
`f45dfc20f34842ec8c87f576c927be73e4899f806003c183c87bd41554570b1c`) records
`PILOT_COMPLETE_RETIRE`, two predictor fits, `test_data_used=false`; its gate
file records `FAIL_VALID_EVIDENCE` / `RETIRE`. The full-continuation namespace
`experiments/runs/optimization_v3_final_continuation/R16` was not present. This
trace does not reopen R16 or propose more fitting.

**Why this is not a recoverable mapped component:** deriving either mapped
vector from a raw vector requires applying the per-cycle average-tie rank map.
That would be a new transform, explicitly outside this read-only trace. The
composite R06/V2 vector cannot uniquely recover its individual contributions,
and clipping/ranking provide no admissible inverse here. The exact legacy
mapped vectors are therefore `NOTFOUND_SCOPELIMITED` in the reviewed artifacts.

## Scope and source state

Checked only the R16 recipe/freeze/results, R06 report and `R06_ORIGINAL`
inference namespace, V2 accepted manifest and inference namespace, and the
named dual-regression/V2-representation caches above. Numeric NPZs were opened
with `allow_pickle=False`; only existing prediction arrays and metadata were
read. No Test feature CSV, target column, serialized model, transformation,
estimator call, or unrelated task output was accessed.

Source checkout at trace start: branch `main`, HEAD
`b0451fb489c1be10c6e0cc31b7822a160a74c47c`. Only this memo is owned by this
trace. It is an inventory finding, not authorization to continue or change the
R16 verdict.
