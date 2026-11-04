# Prefit Audit R2 Correction

The first feature-only preflight completed before any target or Test-value read.
Its audit field named the original A02 predictions NPZ as the candidate
probability bundle. That file was hashed only; its arrays were not opened.

The repository contains two distinct prediction artifacts:

- the original A02 Test predictions and receipt;
- the verified C2_B/A02 inference bundle, which carries `a02_probability`,
  the full-Train C1_B missing fallback, and POS/MIX50 comparator arrays.

R2 binds both artifacts to their own manifests and uses the latter for fixed
policy inference. The first preflight cache and audit remain preserved under
the original `prefit` run directory and `PREFIT_AUDIT.json`; they are superseded
for execution by `prefit_r2` and `PREFIT_AUDIT_R2.json`.
