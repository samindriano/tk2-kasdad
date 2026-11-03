# Zero-fit correction of the A02 inference replay contract

Sol reviewed frozen trained checkpoint bcfb4fc43958b8eb4d1e1030efafa9c4a07ccccb.
All six remaining fits are saved; no new fits or reservations are authorized.
GO for this narrow inference-only repair and completion of the three-CSV batch.

Actual source contract: ResidualNeighborPredictor.predict in
src/models/classification_a02_neighbors.py returns correction_unclipped.
The batch _load_and_replay_a02 instead reads result['correction_raw'] while
constructing mapping['correction_unclipped']. It is a consumer-key error.
The separate CoefficientGeometryPredictor DOES return correction_raw; do not
rename or alter that correct, distinct contract.

Create a minimal, explicit postfit inference adapter plus focused tests/receipt.
Correct ONLY the A02 replay consumer to result['correction_unclipped'].
Preserve frozen trained runner/core source, trained freeze, all nine successful
component artifacts/receipts, both prior failures and original failure evidence.
No replacement fit or source-identity rewriting. A scoped in-memory replacement
of this one helper is allowed when the original source/function hash and exact
single expression are verified first, and the derived helper source/hash plus
adapter commit are independently recorded. No broad exception swallowing or
monkeypatch of guards, hashes, numbers, model methods, or dictionary values.

Original trained-source hashes and recipe still must match exactly. Record
amendment source separately and bind final inference manifest/output hashes to
the adapter receipt. Do not report inference as executed by entirely unmodified
frozen code. A new documented export namespace is permitted if outputs already
exist; preserve originals, and enforce byte-equality on resume if reusing files.

Before complete Test inference, test the actual replay helper boundary with a
contract-realistic prediction dictionary (including unclipped/clipped values)
and existing schema. Reject missing/altered keys and mismatched saved arrays.
Also check all remaining model.predict calls against actual signatures and
return keys, so this is not another happy-path test with mismatched mock APIs.
Keep full original A02 feature-only replay, exact saved probabilities/labels,
neighbors, corrections and source hashes; do not reduce or skip comparisons.
Synthetic fits if needed use synthetic data/temp roots only, never the shared
production ledger. All persisted units remain hash-identical.

Then resume inference using existing fitted components, independent formulas,
all three CSV schema/ID/fallback/probability checks, manifests and batch report.
No Test targets, temporal scoring, parameter choices, or new research.
Ledger remains byte-identical throughout this repair/inference except any
already-declared normal completion event that binds actual completed results.
No new allowance/reservation/FIT events. If another distinct integrity mismatch
appears, preserve evidence and return to Sol rather than silently weakening QA.

Luna owns adapter/tests/receipt/execution/scoped local commits; Sol stops writing
after dispatch. No push or upload: batch-first PREPARED_FOR_REVIEW stop still
applies despite authenticated Kaggle CLI availability. Return result/blocker
to Sol thread01a19569-16e0-7253-aa14-57a0a3160e03,host local, per Sam's standing
return-message authorization. Then STOP.
