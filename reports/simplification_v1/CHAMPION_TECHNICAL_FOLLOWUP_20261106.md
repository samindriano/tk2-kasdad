# Champion continuation technical follow-up

## REG fit: numerical result confirmed, application source requires repair

Root inspected the completed full-champion scalar state and independently reconstructed its exact weighted lower median using Python `Fraction.from_float` weights and a separate cumulative-weight implementation. Result: delta **-1.3955577433171884**, 4,125 informative present rows, choice-dependent absolute residual sum **55,579.14220648316**. Saved delta and objective matched exactly. This was verification of the authorized completed fit, not a second fit or new out-of-sample evaluation. No E79/Test access occurred in this check.

Six application unit tests passed on root replay. Root's source verification encountered the worker's in-progress R2-to-R3 transition and found the new freeze not yet present. Root therefore issued no application pass and asked for an immutable, committed, versioned source-ready handoff before reviewing again.

Root also located a real idempotent-readback defect in `verify_reg_center_local_application_fit_v1.py`: it adds `signature` to the newly computed body, removes signature from the existing receipt only, and then compares unsigned existing with signed new body. That makes a second verifier invocation fail even when the numerical result is identical. Luna2 was authorized to repair/version this verifier and its QA receipt, preserve prior evidence, pin the replacement in the application freeze, and prove two identical verification replays. No refit, old receipt overwrite or waiver of checks was authorized.

## CLS DF map: proportional resource estimate amendment

Luna1's preflight failed before any attempt marker or DF-map fit, leaving the one shared map slot unconsumed. Root inspected actual DF-map code: it reconstructs only the position matrix, whereas the inherited 1 GiB estimate came from replaying full base plus position matrices. Recorded full replay process peak was 556,793,856 bytes (about 531 MiB); the position matrix has 1,638,979 nonzeros, approximately 20 MB for float64/int32 CSR. Root provisionally allowed a **versioned wrapper-local 768 MiB incremental estimate plus the unchanged 768 MiB free reserve**, conditional on explicit dtype/temporary allocation accounting staying within that estimate, new source freeze/tests and fresh preflight/locks.

The estimate uses the observed full-replay peak plus up to 160 MiB conservative additional union/copy allowance; it is not a measured DF-map peak or a guaranteed upper bound. Luna1 must reject it if allocation review contradicts it, record actual RSS/free RAM, preserve V1 freeze and enforce exactly one learned DF-map pass across versions. Attested R3 source/constants and mathematical map formula may not change. No estimator/gamma fit, Test/CSV/upload, app termination or reserve reduction was authorized. Waiting for the other chat to become idle is unnecessary: coordinate real heavy processes and lock ownership instead.

Both bounded technical instructions and return-handoff requirements were sent directly to the correct Luna chats. Root continues on meaningful callbacks, without scheduled checks or idle polling.
