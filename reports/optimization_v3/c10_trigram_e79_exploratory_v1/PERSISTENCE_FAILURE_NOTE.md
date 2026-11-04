# Single-fit persistence failure and recovery

The one authorized official C10 fit reached `joblib.dump` and completed model
fitting, then the original runner raised Windows `OSError: [Errno 9] Bad file
descriptor` when it called `os.fsync()` on a file reopened read-only. The
original `FIT_FAILED` event remains unchanged in `BUDGET_LEDGER.jsonl`. The fit
was not retried.

The only recovery path is byte-preserving persistence of the already-written
`.partial` artifact. Recovery is accepted only if the artifact loads as the
frozen C10 class, all 50,608 Train rows and target hash match the original
`FIT_STARTED` receipt, model state and recipe parameters reconcile, and a
writable-handle copy is byte-identical. Recovery records carry a separate
`FIT_RECOVERED` event and explicitly consume zero additional budget. Fit
duration is unavailable and remains null. Any mismatch blocks Test inference.
