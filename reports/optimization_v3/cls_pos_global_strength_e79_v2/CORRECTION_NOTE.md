# E79 R2 compatibility replay correction

R1 source/attestation and its failed replay receipt remain immutable. R1 commit
`7766a9e833e54402e534c52240cffd282e4a6b1e` is preserved, together with the
R1 attestation and attempt/receipt hashes recorded in the R2 attestation.

## R1 failure and exact cause

The R1 feature-only reader formed `["record_id", "cycle_code", *CLASSIFICATION_FEATURES]`.
`CLASSIFICATION_FEATURES` already contains
`cycle_code`, so the request list had 15 entries but only 14 unique names.
Pandas correctly returned the 14 unique requested columns. The R1 assertion
compared the result column count with the un-deduplicated request length and
stopped before preprocessing or matrix construction. No label or Test column
was requested.

## R2 correction

R2 canonicalizes the ordered request with
`list(dict.fromkeys(["record_id", "cycle_code", *CLASSIFICATION_FEATURES]))`
before `read_csv`. It rejects `artifact_signal` and `threat_rank` in both the
request and returned frame, then requires the returned column names to be
unique and to match the canonical request exactly. Feature transforms still
select `CLASSIFICATION_FEATURES` in its frozen order.

The synthetic tests use the actual `CLASSIFICATION_FEATURES` constant to
exercise duplicate removal and reject missing, extra, and target columns. A
separate synthetic matrix test exercises the same base assembly and expected
hash/shape/nnz/order checks that the official Train replay will use.

## Provenance boundary

The three exact source variances authorized by Sol for the C2_B structured,
scaler, and OHE receipts remain explicitly scoped to those receipts and those
two historical source paths. R2 does not broaden or alter those variances,
edit historical receipts, recompute a DF map, fit gamma, or access Test.

The R2 attestation pins this note, the corrected replay source, the unchanged
numerical helper closure, the immutable R1 evidence, saved-state hashes, and
the historical full-Train matrix audit. The R2 feature replay is separately
one-shot and remains subject to a fresh advisory-lock and RAM gate.
