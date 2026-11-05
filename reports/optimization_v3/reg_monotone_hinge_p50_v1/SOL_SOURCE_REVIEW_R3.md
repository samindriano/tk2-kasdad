# Sol source review — R3

PASS for HINGE_D1 and HINGE_D2 only. Source f1c1ab6914d642508e2089f7f3c200d158ceb3ba.

Independent checks: eight unit tests passed; committed source/input hashes matched the identity freeze; identity-only receipt validation passed for 20,903 unique OOF rows. AST comparison confirmed the numerical LP, fit persistence and scientific gate functions are unchanged from the reviewed R2 implementation. R3 repairs the real three-argument signed-receipt API and adds an actual pre-pass identity integration check. Preflight selectively loads non-target members, verifies both GLOBAL controls, and records zero target access and zero LP starts. Historical R2 failure remains preserved.

Freeze SHA256: 9ddf50d2796b9b19e4e4ca61767dea80cbc81161daa50a8536f3f5356aceedb5. Preflight SHA256: 0343e5e91aa934f9bd0270b1ebd110b7b2148a850179d0f8d1b7f01f2e9e8c0f.

Dispatch two diagnostic fits serially under the shared heavy-job lock and current small-LP RAM policy. Independently verify row alignment, saved states, metrics, subgroup guards and ledger before returning results. This pass does not authorize a final fit, Test inference, CSV or Kaggle upload. Return results to the root Sol chat before stopping.
