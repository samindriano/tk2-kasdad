# CLS_A02_POSITION_KERNEL_V1 — incident amendment 01

Recorded before further execution on 2026-11-06, following Sol's adjudication in thread `01a19569-16e0-7253-aa14-57a0a3160e03`.

## Incident

During read-only cache schema inventory, a Python loop opened the E67 A02 `predictions.npz` and accessed every member to print each member's name, shape, and dtype. That loop materialized the query `labels` member (`n=20,903`, `int8`) before the four candidate prediction states and the independent prelabel reconstruction were durable. The label values were not printed, summarized, scored, or used for any decision. The probe stopped at `ModuleNotFoundError: No module named 'src.models.classification_a02_neighbors'` while loading the E67 saved model, so it never processed the E75 archive. No predictor construction, estimator fit, budget reservation, candidate output, or metric scoring had started. Exact hashes and the probe record are in `INCIDENT_01.json`.

This was a real breach of the original query-label boundary. `original_prelabel_boundary_compliant` is permanently recorded as `false`; no later QA can rewrite that fact or claim an outcome-blind evaluation. The access does not authorize use of those labels in construction, training, or recipe/control/weight selection.

## Authorized continuation and unchanged protocol

Sol authorized continuation as an exploratory result, with the deviation recorded. The already committed `ROOT_EXECUTION_PLAN.md` recipe, predictor-construction cap of four, controls, comparison set, metrics, support rules, performance thresholds, and exploratory-eligibility rule remain unchanged. The incident is an amendment to the execution record only; it does not amend the scientific gates or historical evidence.

Before proceeding, all loaders must use explicit non-label member allowlists. Schema inspection may read `z.files`, but must never index every NPZ key. Add a regression test with a poisoned query-label member and an access spy that proves feature/prediction loading leaves that member unopened. Resolve the saved-model import narrowly by verifying the historical source path and source/artifact hashes; do not install packages, refit, or approximate the artifact.

After the four frozen candidate/control bundles are durable, run independent prelabel reconstruction and source/identity/routing checks. Score only at the existing authorized scoring boundary, then report the frozen scientific performance-gate booleans separately from exploratory eligibility. Scientific qualification remains `false` regardless of numeric performance. Exploratory eligibility may be `true` only if the unmodified plan's rule passes, independent QA passes, and the model-construction training boundary is independently verified; it remains exploratory-only. No Test, E79 construction, SampleSubmission, or Kaggle action is authorized.
