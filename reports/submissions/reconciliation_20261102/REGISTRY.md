# Kaggle submission reconciliation — 2 November 2026

## Result

Read-only, authenticated Kaggle browser inspection confirms **7 Classification** and **14 Regression** submissions visible for Samuel Indriano; all 21 show `Complete`. The current Classification scores range from `0.82640` to `0.82908`; the current Regression scores range from `11.16620` to `13.53473` MAE. The live rows, recipe links, local manifest hashes, and known local CSV hashes are recorded in [`REGISTRY.json`](REGISTRY.json).

The exact initial API receipts remain available: C4_HGB ref `57808225`, submitted `2026-10-30T14:27:27.577Z`, public Macro F1 `0.82641`; R1_D ref `57808231`, submitted `2026-10-30T14:27:38.327Z`, public MAE `13.53473`. All other current rows expose only relative age labels in the UI; submission refs and exact timestamps are marked unknown.

The previously ambiguous `11.71363` score is reconciled: it belongs to `lgb_text_inner_v1.csv`; it is a different model and CSV from the initial R1_D submission (`13.53473`). C16 and R22_ORIGINAL are also now confirmed `Complete` at `0.82844` and `11.30535`, respectively. Their prior local `PREPARED_NOT_SUBMITTED` / `NOT_SCORED` state is a stale preparation snapshot.

## Access and quota

The local Kaggle CLI command is unavailable, so this audit used the already signed-in Kaggle browser and made no submission. Both pages show `0/3` selected entries, explicitly described as final leaderboard selection slots. That counter is **not** the upload quota. Current upload quota is therefore `UNKNOWN`. A 31 October local preflight contains a dated quota snapshot, but it is not live and is not used as current evidence. The competition UI showed a deadline of 8 November 2026 at 23:55 WIB, with seven days remaining at observation.

## Hash binding and history limits

Local manifest/CSV SHA-256 values are recorded when available. The UI does not expose a server-side CSV hash or the manual-upload refs, so those hashes are **not** claimed to match the uploaded Kaggle bytes. For the two initial uploads, the local manifest links the CSV SHA-256 with a contemporaneous successful API receipt/ref; Kaggle still does not expose a server-byte hash. The stale C16/R22 status file is retained as historical preparation evidence and not overwritten.

No submission was created or modified during this reconciliation. No predictions were generated and no new test labels were accessed.
