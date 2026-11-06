# E79 Test application recovery R1

This recovery is a narrow adapter for the already-authorized, one-time Test
application. It preserves the frozen E79 runner, source freeze R2, ROOT_PASS,
the previously consumed application attempt, and both accepted fit artifacts.

The frozen `verify_root_pass` validates the committed ROOT_PASS, then returns a
compact summary without its seven verified input hashes. The frozen
`apply_test_once` subsequently expects those hashes in that return value. The
first application therefore stopped at the binding comparison before creating
predictions or a CSV. Recovery R1 runs the original verifier first, independently
checks the committed ROOT_PASS self-hash and all seven expected values, then
forwards those exact values to the unchanged frozen application function.

## Limits

- One recovery application attempt, with no retry.
- Zero official fits; both existing fitted gamma artifacts and their receipts
  must independently verify before inference.
- Input source, frozen recipes, predictions, thresholds, availability routing,
  and baseline champion parity remain those of E79.
- Test use is feature-only. No target label or SampleSubmission is loaded.
- No Kaggle access or upload is performed.
- Outputs use `experiments/CLS_POS_POISSON_BAG2_E79_V1/APPLICATION_RECOVERY_R1/`
  and a uniquely named CSV outside the frozen application namespace.
- Any application error consumes this recovery. Its failure receipt stays in
  the recovery namespace, and the old fit ledger and application attempt remain
  untouched.

## Pre-application checks

1. Run `python -m unittest tests.test_cls_pos_poisson_bag2_e79_application_recovery_r1`.
   The synthetic test executes the frozen application runner, performs a
   persistence/readback pass, checks baseline replay, and rejects CSV tampering.
2. Commit the wrapper, independent verifier, test, and this protocol.
3. Create and commit the write-once recovery authorization. It binds the exact
   official repository HEAD, source/freeze/ROOT_PASS hashes, old failed-attempt
   hash, both fit receipts, and the zero-fit/one-application cap.
4. Recheck RAM, locks, authorization, repository HEAD, official Test hash,
   baseline hash, fit receipts, and fresh recovery output paths.
5. Invoke the recovery exactly once. Do not retry after any persisted attempt.

## Post-application checks

Independent readback verifies the exact ordered official Test IDs and row count,
binary labels, candidate and baseline CSV schemas, champion decision replay,
missing-stream fallback and count, candidate changes, application receipt and
bundle hashes, fit artifact/receipt identity, and the zero-fit / no-target / no-
Kaggle boundaries. The final recovery QA receipt records all file hashes and
the independent verifier result.
