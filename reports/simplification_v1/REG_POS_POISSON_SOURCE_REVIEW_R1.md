# Root source review: REG positional Poisson BAG2 R1

Reviewed committed source `1161187ab4f9d34c8c8c59f5318705ed9ef3786e` in Luna2's isolated REG worktree, freeze file SHA256 `554e0b2b49356c41dfa1241c374f8a7d36546767933023cea681924c1a13b961`, internal signature `9bbf9a6ab2bdebb293a7fb10ff08c77a422f3c4633e80296c729691ca58e6611`.

Root independently reran all six unit tests: PASS (known sklearn sparse feature-name warnings only). Reviewed Poisson count generation/normalization, exact four-fit order/cap, frozen feature matrices and preprocessing reuse, training-target reads restricted to fit rows, original-head replay, equal raw-head averaging, unclipped positional rank50, fixed DIRECT blend/clipping, past-only hinge, unchanged missing route, durable fit receipts/ledger and separate scoring boundary. Target-free preflight records exact original comparator replay on D1 n7,462 and D2 n8,491. No official fit or scoring label access occurred in root review.

Decision: **REWORK coordination before source pass**, with no model/weight/recipe changes:

1. Both runner and verifier currently require literal current HEAD and unrelated external MAIN branch/status equality. Replace these with reviewed-source ancestry plus exact relevant Git blob/worktree/source/artifact/Train hashes. This preserves actual integrity checks while allowing concurrent unrelated CLS work and later receipt commits.
2. Official fit currently holds only a lane-local lock. Acquire the shared persistent OS advisory byte lock `%TEMP%/kasdad_single_heavy_job.lock` before feature rebuild/fit as well, release by context manager; file existence is not lock ownership. Synthetic tests may use a temporary lock file.
3. Align boundary operators with the already-declared root plan: D1gain>=-0.05 and supported-group harm<=0.25. R1 used strict >/< instead. This correction is requested before official outcomes, not a retrospective gate relaxation.

Luna2 was instructed to version/freeze the technical changes, preserve R1 history, rerun targeted tests/signature/target-blind checks, and return committed source hashes to root promptly. No repeated expensive feature replay is needed solely for canonical serialization metadata if unchanged relevant source/input bytes and prior replay proof remain bound. Four real predictor fits remain unstarted and pending this source decision. CLS Poisson-gamma source work proceeds independently.
