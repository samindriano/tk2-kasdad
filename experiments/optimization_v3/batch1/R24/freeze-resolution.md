# R24 freeze identity correction

`freeze.json` is preserved as an audit artifact. Its identity included the host's
momentary `available_physical_bytes` and derived `allowed_bytes` readings. The
first frozen-pilot invocation detected that those readings had changed and
stopped before starting either predictor fit; no R24 fit or R24 budget
reservation was made under that freeze.

The runner now binds deterministic workload estimates and resource limits in the
freeze identity. It still measures available and allowed RAM afresh and requires
the live resource preflight to pass before a fit. The corrected freeze is
`freeze.v2.json`, created after source commit `d551264`.

The earlier freeze remains unchanged. `freeze.v2.json` is the active R24 freeze.
