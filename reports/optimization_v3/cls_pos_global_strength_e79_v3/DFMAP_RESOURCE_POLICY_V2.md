# E79 DF-map resource-policy amendment V2

## Scope and preserved evidence

This amendment changes only the local resource estimate for the already
authorized single Train-only DF-map pass. It does not change the V1 formula,
inputs, fitted states, token order, Train boundary, or fit count. The frozen V1
policy and source remain unchanged, and V1's original runner is not invoked by
this amendment. Both versions share the canonical map/receipt output namespace
so that a V2-created map or terminal receipt blocks a later V1 retry.

The earlier V1 resource preflight returned `BLOCKED_RESOURCE_GATE` before an
attempt marker, map, or receipt was written. It remains unconsumed. The V2 pass
requires the shared attempt marker, map, and receipt paths to be absent before
it can start. Once V2 begins, it writes its policy ID into the canonical
attempt-marker path, so an interrupted V2 run blocks the unchanged V1 runner.

## Resource estimate

The existing R3 full saved-state replay measured a process peak RSS of
556,793,856 bytes (531 MiB) while both the 50,608 x 20,427 base matrix and the
50,608 x 17,259 positional matrix were resident. DF-map creation does not build
the base matrix. Its extra sparse work is the positional-matrix copy, three
segment slices, sparse-union intermediates, and a bounded row-coordinate
scratch allowance. The V2 estimator uses the frozen 1,638,979 nonzeros and
conservative 8-byte CSR value/index sizes, even though the positional builder
produces float64 values and SciPy normally uses int32 indices at this shape.

The implementation computes a static upper bound for those sparse copies and
temporaries and refuses the run if it exceeds 160 MiB. For the frozen shape and
nonzero count, the bound is below that limit. Adding that allowance to the R3
measured peak is below 700 MiB; V2 rounds the incremental estimate up to 768
MiB. This is a conservative estimate from the R3 measurement and explicit
allocation bounds, not a claim that a DF-map peak was measured already.

The free-memory reserve remains 768 MiB. V2 therefore requires at least 1.5
GiB available before starting, rechecks the reserve at the post-fit boundary,
and records before/after available memory and process peak RSS. The project
shared heavy-job lock and the CGeom local lock are both required.

## Frozen map recipe

The original V1 runner's `compute_df_map` remains the sole formula
implementation. For each retained unigram, document frequency counts one row
if the token occurs in any positional block. It then computes
`raw = clip(sqrt(median(df) / df), 0.5, 2.0)`, `aligned = raw / mean(raw)`,
the seed-42 permutation, and `v=max(1, aligned); c=mean(v)` with a constant
`c` repeated across all 3V position coordinates. The V2 wrapper instruments
the exact matrix dtype and sparse allocation bound, then calls that unchanged
formula. It adds no estimator fit and does not load either target, Test, or
SampleSubmission.

## Run and verification

After the source, tests, and this policy are committed, create the V2 freeze and
run its independent verifier. Then run exactly one `--fit-df-map` action. That
action acquires both locks, takes one fresh resource snapshot, and passes it to
the separate independent preflight verifier while the locks remain held. Only
if that exact snapshot passes does it create the attempt marker and map. Any
interruption or failure after that marker is terminal; there is no retry. No
further gamma fit, Test inference, CSV, submission, or Kaggle access is part of
this amendment.

```powershell
python -m pytest -q tests/test_optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2.py
python -m scripts.optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2 --build-freeze
python -m scripts.verify_optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2 --verify-freeze
python -m scripts.optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2 --fit-df-map
python -m scripts.verify_optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2 --verify
```

The V1 runner and freeze remain unmodified, and the V1 fitting implementation
is not invoked. V2 owns the one canonical global attempt marker and receipt.
Its marker's explicit `policy_id` identifies the resource amendment; the
unchanged V1 runner sees the marker and refuses a second learned map.
