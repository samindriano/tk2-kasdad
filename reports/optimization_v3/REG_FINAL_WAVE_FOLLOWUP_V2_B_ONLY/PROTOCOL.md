# REG E79 B-only application protocol

## Scope

Create one zero-fit output by reusing the already saved global affine candidate
on present rows and restoring the original saved direct candidate on the exact
1,062-row missing mask. The global coefficients remain fixed at intercept
`2.169196428141562` and slope `0.9744451489746011`; the application reads the
saved global CSV and does not recompute them.

The output must contain 29,421 unique ordered IDs and exactly the
`record_id,threat_rank` columns. Values must be finite and within `[0,100]`.
Present-row values must be bit-exact to the saved global CSV; missing-row values
must be bit-exact to the saved direct CSV. The full output vector must differ
from both saved candidate vectors.

## Required order

1. Run the synthetic QA and unit tests. They use only synthetic values.
2. Commit the R2 application, independent verifier, tests, protocol, R2 synthetic
   QA, and append-only R1 failure receipt as a source change. Preserve the R1
   freeze and R1 synthetic QA byte-for-byte.
3. Run `python scripts/reg_final_wave_followup_v2_b_only.py --freeze`. It writes
   `B_ONLY_PREFIT_FREEZE_R2.json` and hashes
   saved input files and validates signed JSON metadata without parsing CSV rows
   or loading NPZ arrays. It also binds the committed source tree and R3
   diagnostic-closure ancestry.
4. Commit the write-once freeze receipt.
5. Run `python scripts/reg_final_wave_followup_v2_b_only.py --apply` once. It may
   read only the saved global/direct candidate CSVs and direct E79 archive.
6. Run `python scripts/verify_reg_final_wave_followup_v2_b_only.py --verify`
   once. It independently rebuilds the row-aligned formula and writes a
   write-once QA receipt.

The preserved R1 freeze `B_ONLY_PREFIT_FREEZE.json` and
`APPLICATION_GATE_FAILED_R1.json` record the first fail-closed application
attempt. R1 loaded the already authorized saved candidate CSVs and E79 arrays,
then stopped before writing output because it compared the direct manifest's
raw little-endian float64 digest with the global candidate's shape-tagged
digest. R2 checks the direct and global source-specific digest formats
separately. This repair does not change the fixed B-only rule or model output
sources.

## Frozen inputs and checks

The freeze pins the signed global manifest, global state and fit receipt,
global CSV and QA, signed direct manifest and QA, direct CSV and E79 archive,
direct trained-state freeze and prelabel receipt, the root decision, and the
closed R3 freeze/source-pass/diagnostic/ledger/closure receipts. Hashes are
checked before any content loading. A changed or missing input blocks the stage.

The archive's ID order is authoritative. Each saved CSV is checked against it;
the composition helper also aligns by `record_id` so synthetic permutation
tests prove that value routing does not rely on accidental file offsets. The
independent verifier separately parses the output and reconstructs every row
from ID-keyed source values and the saved mask.

## Boundaries

- No new fit, solver call, preprocessing fit, model inference, or `Train.csv`
  target read.
- No raw Test/Sample read, Test metric, public-score selection, or Kaggle upload.
- The result is for root review; this artifact does not claim held-out or public
  performance.
- The prior R3 runner, verifier, protocol, and freezes remain immutable. This is
  a distinct namespace with a separate source/input freeze.
- Outputs are write-once. A partial write or existing output blocks rerun and
  requires root review before any recovery action.

## Commands

```powershell
python -m unittest discover -s tests -p test_reg_final_wave_followup_v2_b_only.py
python scripts/reg_final_wave_followup_v2_b_only.py --synthetic-qa
python scripts/reg_final_wave_followup_v2_b_only.py --freeze
python scripts/reg_final_wave_followup_v2_b_only.py --apply
python scripts/verify_reg_final_wave_followup_v2_b_only.py --verify
```
