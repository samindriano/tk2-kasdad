# E79 R3 receipt completion correction

R1 and R2 are preserved as separate historical attempts. R1 stopped at the
duplicate-column check before transforms. R2 canonicalized the feature request
and completed its exact matrix parity checks, then failed while assembling the
success payload because the returned matrix digest names were not in scope.
R2's receipt has a valid internal `receipt_sha256`, but it did not persist
matrix hashes; its exact file hash, internal digest, and error remain pinned by
this R3 checkpoint.

R3 keeps the same Train-only feature read, saved-state compatibility check,
preprocessing, matrix assembly, expected shape/nnz/order/logical-hash gates,
resource estimate, thread limit, and lock policy. The assembly helper now
returns the measured matrix metadata along with the checked matrices. The
success payload takes shape, nnz, and logical SHA-256 only from that returned
metadata, which is computed from the actual matrices before comparing them to
the frozen references.

Before the R3 replay, the synthetic end-to-end test calls the actual
`_execute_feature_only_replay` using a synthetic CSV and synthetic fitted
preprocessing states, serializes the complete payload, writes its one-shot
receipt, reads it back, and passes it to the independent verifier. The test
also scans runner and verifier scopes recursively with Python's AST/symtable
to catch unbound referenced globals, including references inside nested
comprehensions. No official estimator, vocabulary, IDF, DF map, or gamma is
fit or recomputed by this repair.

The official R3 attempt remains a one-shot feature-only replay. It must run
from the committed R3 source/attestation, use only official Classification
Train feature columns, and pass the existing lock and RAM gates. Test,
SampleSubmission, labels, prediction, inference, output CSV, and Kaggle remain
outside this checkpoint.
