# V3 research backlog — Luna overnight execution protocol (draft)

This draft describes how to run the experiment lanes after MAIN has reviewed and queued them. The optimization campaign is still active. At the inspection point, HEAD was `d768502`; V3 supervised-token results and Kaggle preflight snapshots were present in the working tree but uncommitted. MAIN should commit/push and freeze that checkpoint before handing Luna any new fit. Do not start from an inferred “clean HEAD” while that work is pending.

## What the present environment supports

The current Windows machine reported an AMD Ryzen 7 260, 8 cores / 16 logical processors, about 31.3 GiB visible RAM, and about 7.9 GiB free during inspection on 2026-11-01. Free memory changes; check it again before a large vocabulary job. Pin each LightGBM/process to two threads and run one fit pipeline at a time. Do not launch parallel CPU fits or multiple character-vocabulary jobs overnight. XGBoost 3.2.0 is installed, PyTorch is absent, and GPU availability was not verified; avoid making a deep-learning lane an overnight dependency or installing packages mid-queue.

Observed fit times are small for the existing CPU pipelines, but they are not an end-to-end promise. The four supervised-token pilot fits took 3.64–9.10 seconds each at 11k–30k fit rows. Sparse Classification continuation fits took 9.80, 12.98, 21.15, and 25.21 seconds as the fit set grew from 16k to 42k rows. Character Ridge fits took 13.89 and 25.84 seconds in its pilot, then 10.43–26.50 seconds across the four new continuation folds. A fresh raw-only Classification replay with three full-Train components took 100.1 seconds. Feature construction, verification, notebook-free research reporting, and a failed infrastructure retry add time; budget by observed receipts for the same pipeline and leave slack.

The just-completed supervised-token pilot is a useful cost comparator and a retirement result, not a candidate to extend: its fixed blend was worse than the latest Classification control on the primary pilot by 0.000679 Macro F1 and on H12 by 0.001932; Regression was worse by 0.116570 and 0.128519 MAE respectively. Its independent verification passed. Do not spend overnight fits on that family unless a new, materially distinct hypothesis appears.

## One lane at a time

MAIN assigns Luna exactly one experiment ID and one bounded result at a time. The queue may contain many hypotheses, but a result from one ID must be reviewed before starting the next. This controls RAM use and prevents candidate selection from silently becoming a large search against the same reused temporal outcomes.

For each assigned ID, Luna follows this sequence:

1. **Check the base.** Confirm repository, branch/worktree, commit, raw official Train hashes, dependencies, and named V3 source/report inputs. Inspect `git status`; do not overwrite existing user changes or accepted artifacts. Use the existing V3 official-data resolver and current active verification path.
2. **Read the experiment card.** Confirm the hypothesis, candidate recipe, fit cutoffs, primary/H12 controls, metric, support groups, advance gate, allowed files, expected fit count, and stop condition. If any item is absent or source/control identity has changed, report a blocker to MAIN before training.
3. **Check the harness before an official fit.** Run syntax checks, unit tests for newly authored transformations, synthetic temporal-boundary tests, and a dry-run that checks schemas, output paths, report aggregation, and zero-support subgroups. Resolve expected-output and directory-creation errors now. Ensure the runner refuses a partial/existing cache before it begins fitting; persist each fit atomically with an exclusive receipt so a restart cannot overwrite a prior result.
4. **Freeze once.** Bind the exact recipe, normalized source hashes, official Train hashes, allowed cache/report inputs, fold rows, fit population, dependency versions, and fixed selection gate before inspecting candidate scores. For temporal target encoders or score heads, explicitly bind row indices and prove that each training feature uses labels strictly earlier than its cycle. A Classification expert intentionally trained on present-only rows is a distinct fit-population choice; report it as part of the candidate.
5. **Run the declared pilot only.** Write one fit receipt per cutoff with fit-row fingerprint/count, prediction-row fingerprint/count, fit state, score/cache hashes, runtime, and `Test/Kaggle=false`. Do not add cutoffs, parameter variants, threshold/weight alternatives, or subgroup-specific branches after seeing pilot results.
6. **Verify separately.** Run the frozen independent verifier and compare its calculation with the report; do not merely call the same metric/aggregation helper twice. Check primary, H12 early/late, every declared temporal window, supported subgroups, tails, missing/present streams, row alignment, fit-only preprocessing, and unchanged fallback routes. Unknown, empty, or unsupported strata stay explicit; do not invent support or merge groups after seeing outcomes.
7. **Return to MAIN.** Report the experiment ID, freeze signature, source/input hashes, expected and actual fits, failures/retries, primary and H12 deltas, per-window signs, worst supported subgroup harm, verification result, and one of `ADVANCE`, `RETIRE`, or `REWORK`. Stop. MAIN decides whether the next card is run or whether a challenger earns six-window evaluation.

## Failure, resume, and artifact rules

- A worker that reaches a fit should create the designated output directories before the fit. One missing folder previously caused a completed Classification residual fit to abort before saving; its same-recipe recovery was logged as an unsaved execution attempt plus retry. Avoid this with a no-fit path preflight.
- A controller/report defect found after fits does not authorize refitting or editing frozen fit code. Preserve the original source, failed receipt, and predictions. Freeze a versioned report-only adapter before rebuilding metrics. Independent verification must reproduce the predictions from exact saved arrays and recalculate metrics on exactly the same row masks.
- A deterministic/environment retry is not a new model configuration, but count it. If some fit outputs were saved, verify their sidecars and exact hashes and reuse them. If output presence is ambiguous or a partial archive could be overwritten, stop and ask MAIN to recover it safely. Never delete caches to make a runner restart.
- Keep raw data, predictions, model binaries, and prepared CSVs out of Git. Put each lane's protocol/freeze/source/report/test under its assigned V3 subfolder; use an ignored run-cache namespace. Stage only the named experiment paths. Preserve old verifier failures/amendments as historical evidence and mark the current active verifier in the report.
- Fresh checkouts must preserve the line endings required by the freeze. Some older worker receipts hash literal source bytes; use an LF-preserving clone/check-out (`core.autocrlf=false`) when replaying them. Do not change repository-wide attributes to repair a new receipt.
- Do not call Kaggle in a pilot or verifier. Only MAIN handles a full-Train artifact, repeats the live authenticated quota/team/deadline preflight, verifies output IDs/schema/range/hash and decides whether the internally strong candidate deserves an explicitly authorized upload. Historical snapshots are stale and are not a quota guarantee.

## Guard against validation overuse

All existing cycles 47–79 have already appeared in several rounds of model selection. The six expanding primary windows and fixed-origin H12 are useful consistency checks, but they are adaptive development evidence, not fresh holdouts. A pilot on W1/W4 plus H12 reuses those same cycles; do not count it as independent confirmation. Use the queued pilot to reject weak families cheaply, not to tune multiple parameter values.

Use a small predeclared pilot set per candidate family, followed by one six-window continuation only if the gate is met. Each fit/validation cutoff must respect its past-only boundary, and all learned preprocessing belongs inside that cutoff. Report raw-support and common-support denominators for subgroup comparisons. Require the candidate to beat its current champion on the task's primary aggregate, retain acceptable H12 behavior, show no severe supported-window or tail/missing harm, and pass source/fit QA. Keep a candidate as an internal comparator when it misses its gate; do not lower a gate after seeing its score. Do not spend submissions as a leaderboard search.

Where practical, test the analysis/selection rule on earlier cycles before exposing later cycles, then lock the candidate and rule before the six-window comparison. This can reduce adaptation but cannot create an untouched holdout from these repeatedly used labels. Public Kaggle scoring is a separate noisy feedback signal, never hidden-label evidence or a substitute for temporal validation.

## Reusable single-experiment Luna handoff

MAIN should paste one experiment card into this template and replace every bracketed field. Luna is to complete that card and return to MAIN; it should not continue down the whole backlog autonomously.

> **Experiment:** `[ID and one-sentence hypothesis]`
>
> **Current verified control:** `[recipe, primary and H12 metrics, source/protocol paths]`
>
> **Candidate:** `[exact recipe, fit population, feature/target transforms, fixed parameters, fixed threshold/weight]`
>
> **Inputs and boundaries:** `[official task Train hash; fit cutoffs; validation cycles; allowed cached controls; forbidden sources/features]`
>
> **Pilot plan:** `[exact number of new fits; row counts; metrics; advance/retire gate; subgroup/tail support minimum]`
>
> **Owned paths:** `[unique experiments/optimization_v3/<ID>, reports/optimization_v3/<ID>, scripts/tests paths, ignored run-cache path]`
>
> **Required checks:** `[pre-fit dry run and temporal proof; freeze before scores; no cache overwrite; independent verifier; output hashes]`
>
> Work only on this ID. Do not use official Test labels, other-task labels, external/pretrained data, Kaggle, or teammate artifacts. Do not edit any frozen source, prior report, official notebook/EDA, or unassigned path. Do not run model fits until the gate and artifacts above are frozen. Run only the declared pilot; no grid or additional fold. Preserve failed attempts and partial outputs. If a correctness defect appears, stop fitting and report it; use a separately frozen analysis-only adapter if only reporting is affected. Return metrics, denominators, window directions, subgroup counterexamples, fit/time accounting, verifier status, and `ADVANCE/RETIRE/REWORK`. Stop after the result; MAIN will issue the next card.

## Unattended serial queue mode

For a sleeping-user batch, MAIN can issue one queue manifest with a fixed sequence of experiment IDs and authorize Luna to advance without waiting for per-ID Sol approval. Luna applies the written gates mechanically, writes one compact result per ID, and starts the next predeclared ID only after the prior fit has terminated and its verifier/receipt is recorded. `RETIRE` advances immediately. A bounded report-only repair can proceed under its own pre-metric amendment. `REWORK` caused by an unresolved source/data/control issue is recorded, the ID is skipped, and the queue continues to the next independent ID; do not keep repairing the same family or adapt its recipe to the score. Luna stops the whole queue on a protected-file/input-hash violation, ambiguous partial cache, missing required source, or a defect that could change the fit population/temporal boundary. It reports these as `BLOCKED` instead of silently continuing.

Use a **six-hour overnight budget** as a ceiling, not as a target. Cap the batch at **96 successful candidate fit executions** across both tasks; count every task, cutoff, and candidate configuration as one fit, and record deterministic verifier refits separately. Permit at most **two configurations per family in pilot** and advance at most **two fixed candidates per task to six-window evaluation**. These caps preserve room for breadth while limiting repeated selection against the same already-used cycles. Pilot/validation fit counts are not quota to consume: stop when the queue is exhausted or all remaining cards fail a prewritten gate. At the time ceiling, launch no new fit. Let the currently executing fit terminate naturally, save its receipt, and then stop; do not kill an active fit to hit the clock. Use a single CPU job with two threads, recheck free memory before any high-vocabulary representation, and skip it if memory headroom is insufficient.

The unattended preset has `Test_read=false` and `Kaggle_access=false`: prepare no full-Train submissions and make no Kaggle API calls overnight. This limits external actions in the unattended batch; it does not revoke the existing V3 submission authorization. MAIN can handle a competitive candidate later with a fresh authenticated rules/quota/team/deadline check and a distinct upload decision.

The queue's end report lists every ID and its state, fit/runtime counts, freeze/cache/source identities, pilot and full-six metric deltas, gate results, verifier status, skipped/rework reasons, and any unfinished active fit. It gives MAIN one consolidated review after the batch rather than asking for a decision between independent IDs. This change removes the per-experiment Sol review bottleneck while keeping all model-promotion thresholds fixed before scores.

## Suggested overnight sequence

Use the task-specific candidate cards in the adjacent Classification and Regression drafts. The current V3 supervised-pilot checkpoint has now been committed and pushed as `cb27597`. MAIN should review/freeze the combined backlog and issue the ordered unattended queue manifest. Luna then runs the cards serially under the fixed budget and gates, including auto-advance after `ADVANCE` or `RETIRE`; MAIN reviews the consolidated end report after waking. No automatic full-Train refit or Kaggle submission is part of this overnight preset.

At wake-up, MAIN should reconcile every queued ID as `NOT STARTED`, `PILOT PASS/FAIL`, `FULL SIX PASS/FAIL`, `RETIRED`, or `BLOCKED`; verify per-fit receipts and hashes, compare summaries against the frozen control, and integrate only assigned new paths. If Luna stops or the computer restarts, resume from the last complete fit receipt after verifying its exact hashes. A score without its frozen recipe, row lineage, and verifier status is not an experiment result.
