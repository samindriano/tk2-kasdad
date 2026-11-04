# Prefit pin correction

The first target-free prefit invocation stopped before writing any freeze or
budget reservation. The new runner had a one-character typo in its expected raw
SHA-256 for the historical C10 recipe (`491f...` instead of the repository's
committed `491e...`). The working recipe and the Git blob at `HEAD` were byte
identical. The historical freeze, results, and verification files also matched
their recorded hashes. Only the new runner's expected pin was corrected; no
historical artifact or verdict was edited.

The corrected runner now checks all four historical files against both their
expected digest and the committed Git blob, then reconciles the recipe file hash,
recipe identity, freeze signature, result, and verification receipts. A focused
unit test accepts the exact pinned bytes and rejects a genuine byte mutation.
