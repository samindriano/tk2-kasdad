# E79 deterministic Train-only DF-map continuation

## Authorization boundary

This is the separate deterministic DF-map step conditionally authorized after
the R3 descendant replay passed exact base and positional matrix parity. It
creates one preprocessing map from official Classification Train features.
It does not fit a supervised gamma, read either target, inspect Test or
SampleSubmission, infer predictions, create a submission, or contact Kaggle.

The original V1 formula and saved A02 vectorizer are fixed. For each retained
unigram, DF counts a Train row once if that unigram occurs in any of the three
positional blocks. Then `raw = clip(sqrt(median(df) / df), 0.5, 2.0)`,
`aligned = raw / mean(raw)`, and the diagnostic control uses
`numpy.default_rng(42).permutation(V)`. The global-strength vector is
`v=max(1, aligned)`, `c=mean(v)`, and `penalty=full(3*V, c)`.

The map fit reuses the verified full-Train A02 vectorizer and original
positional builder. It does not refit TF-IDF, IDF, structured preprocessing,
or a model. Its only new learned preprocessing state is the deterministic DF
map, counted as one fit. The one-shot artifact and receipt live under
`experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/df_map_v1/`.

## Execution and QA

After the protocol, implementation, verifier, and tests are committed, build
and commit the source freeze. The freeze pins the R3 attestation and replay
receipt, Train and saved-state hashes, implementation, test and recipe-source
hashes, formula, and output namespace.

```powershell
python -m pytest -q tests/test_optimization_v3_cls_pos_global_strength_e79_dfmap_v1.py
python -m scripts.optimization_v3_cls_pos_global_strength_e79_dfmap_v1 --build-freeze
```

Commit only the new protocol, implementation, verifier, test, and freeze. Then
run the read-only preflight and, only if its shared/local lock and RAM gates
pass, the one-shot Train-only map step:

```powershell
python -m scripts.optimization_v3_cls_pos_global_strength_e79_dfmap_v1 --preflight
python -m scripts.optimization_v3_cls_pos_global_strength_e79_dfmap_v1 --fit-df-map
python -m scripts.verify_optimization_v3_cls_pos_global_strength_e79_dfmap_v1 --verify
```

The independent verifier checks the freeze, R3 parity receipt, array schema,
formula, permutation, token order, output hashes, and declared source
boundaries. Synthetic tests cover per-row union DF, formula, NPZ persistence,
resource blocking, and no-retry behavior. Any attempt marker forbids rerunning
the official map fit.

## Following gamma-fit package

After the DF map and independent QA, the next deliverable is a separate
source-ready gamma-fit and application package using the frozen V1 objective,
full-Train fit population, A02/C2_B/CB assets, `0.5 * new POS + 0.5 * pA` on
present streams, exact `pA` fallback on missing streams, present threshold
`0.50`, and missing threshold `0.53230976129477`. The package may be prepared
and synthetically tested here, but official supervised gamma fitting requires
a distinct root source pass. Test inference, CSV generation, and Kaggle remain
outside this checkpoint.
