# Luna overnight lane plan — V3 research

Status snapshot: 2026-11-01. This document turns the existing canonical V3 queue into a practical research order. It does not add candidate recipes or change queue gates. Recheck the repository and every card before acting; reports and frozen inputs outrank this snapshot.

## Scope and current guardrails

- Finish the already-started **R04** lane using its current runner and report. Check its actual report before doing anything: do not repeat a completed fit or verdict.
- After R04, follow the canonical queue order and each card's dependency. The first eligible card is C05, subject to its C04 condition.
- During each pilot and continuation, use official Train only. Do not open Test or SampleSubmission or access Kaggle in these stages. A qualifying finalist may proceed to full-Train refit, Test inference, or submission only after the handoff's full-six/support/source gates pass and the current V3 authorization, Kaggle team/rules/deadline/quota, global budget, manifest, and idempotency checks all pass.
- Preserve EDA, notebooks, prior experiments, frozen protocols, teammate work, and accepted submissions. Do not edit the canonical queue or relax a card's recipe, support, fit budget, or gate.
- The cards after R04 are marked `PLANNED_NOT_RUN`; no card-specific runner/adapter was present when this map was written. Do not invent a command such as `python ..._c05.py`. Build and test the required new runner before a card's pilot.

The shared handbook is `LUNA_HANDOFF.md`. Its B0 harness, report contract, freeze procedure, resume rules, and campaign limits apply. In particular, require end-to-end synthetic aggregation before a new card's official fit, reserve declared fits first, use at most one two-thread model-training job at a time, and stay under the existing budget of 180 candidate predictor fits and 24 control/verification refits. Check actual remaining budget from the canonical ledger; a planned fit count is not proof of remaining budget.

## Near-term candidates in canonical order

Every card below still needs an actual file/status check, its own preflight, implementation/tests if absent, a frozen recipe, and independent verification. The fit counts are the card's declared pilot predictor fits only; retries, control refits, and later continuation are separate budget items.

| Queue order | Lane | Pilot fits | Fixed question / useful next step | Conditions and fail-closed checks |
|---:|---|---:|---|---|
| 31 | **C05 — Rare-term retention** | 2 | Test the already-declared combined word TF-IDF vocabulary with `max_features=60000`, `min_df=2`; preserve the current routing and sparse expert. | C04 must be complete without resource failure. C04 is recorded `RETIRE`; both declared fits were saved and its fit accounting reports 0 failed attempts. Do not try additional cap or `min_df` values. |
| 32 | **R14 — EF1 sparse specialist** | 2 | Test a target-scale Regression expert only on parsed `entry_friction == 1`; other rows retain the exact incumbent. | Require at least 500 EF1 rows in each fit and at least 100 EF1 validation rows in each scored view. EF means `entry_friction`, never `telemetry_flag`. No within-subgroup rank mapping. |
| 33 | **C11 — Word + character sparse boosting** | 2 | Add the fixed `char_wb(3,5)` block to the existing word representation and sparse classifier. | Estimate RAM and vocabulary from Train folds first. Keep the design sparse; do not densify. |
| 34 | **R08 — Row-local token statistics** | 2 | Add token count, unique-token count, TTR, and character count to the existing structured + SVD32 Regression recipe. | TTR for an empty stream is fixed at 0. Omit `repeated_fraction` because it is exactly `1 - TTR`. This is a predictive ablation, not a causal claim about EDA A. |
| 35 | **C20 — Cached fixed-q80 recomposition** | 0 | Reconstruct the already-defined q80 route over all six cutoffs using existing sparse/C2_B/CB predictions. | No threshold search and no model fit. Verify every cache, row identity, source lineage, and complete six-cutoff coverage; if any required cache is absent or mismatched, record `BLOCKED_INPUT`. |
| 36 | **R19 — EF-stratified OOF median offsets** | 0 | Apply the fixed median residual offset by `entry_friction` using the latest eight complete prequential fit cycles. | Confirm exact OOF chronology and at least 200 OOF rows per EF group before using labels. This is LATE evidence (`W4+W6` / `H12_L`), not the standard `W1+W4` primary. If support/lineage fails, skip. |
| 37 | **C15 — Two-score prequential meta-head** | 2 | Fit the fixed LR meta-head from C2_B and sparse binary predictions only when their score blocks predate their labels. | Require at least 5,000 present-stream OOF rows and at least 1,000 per class; base cutoff must be earlier than each meta label. Fix fallback before scoring. This is LATE evidence. |
| 38 | **R22 — XGBoost pseudo-Huber** | 2 | Compare the already-fixed pseudo-Huber objective (`huber_slope=5`) under the R21 design. | Check package availability and resource limits first. Do not search alternate slope values. |

There are seven additional IDs after these first eight. Keep canonical order and only proceed when each dependency is true:

- **R20 (39, 2 fits):** select exactly 128 token identities inside each outer fit. Check that enough eligible tokens exist; validation labels must not enter selection.
- **C12 (40, 2 fits):** character-only ablation, and only if C11's pilot passes. It is not a rescue for a failed C11.
- **R06 (41, 2 fits):** character sparse Regression; do RAM/nnz checks before fitting and do not infer semantics from anonymous character fragments.
- **R07 (42, 2 fits):** NMF32 on the nonnegative text block only. Check RAM and convergence; never apply NMF to scaled structured values.
- **R10 (43, 2 fits):** currently **not eligible**. Its card requires a qualifying R09 result. The stored R09 report is `RETIRE`, with primary gain about −0.0364 and H12 gain about −0.0817; its predeclared checks fail. Record a condition-based skip unless the authoritative R09 evidence changes.
- **R13 (44, 2 fits):** currently **not eligible**. Its card requires R12 to improve average within-cycle Spearman at both origins while missing the MAE gate. Stored R12 evidence has positive primary/H12 MAE gains (+0.245801/+0.284490) but retires on supported missing-stream harm; its report does not establish the required paired Spearman result. Do not infer or fabricate that condition.
- **R11 (45, 18 fits):** keep last. It requires cheaper objective/ranking lanes complete, enough budget for all 18 heads, and at least 100 positive and negative fit examples for every threshold. Skip before any fit if one of these fails.

R09/R12 outcomes above are development evidence, not untouched holdout evidence. A valid gate failure retires that recipe; it does not justify changing the recipe. A card whose condition, data, cache, support, or resource preflight fails is skipped with exact evidence and the next independent eligible card proceeds.

## Repeatable per-card procedure

For each card, Luna must complete the whole card without asking for approval between its ordinary milestones:

1. Recheck repository root, remote, branch/HEAD, working tree, and ownership. Read the canonical queue entry, the complete card, relevant source/control manifests, `LUNA_HANDOFF.md`, current ledger, and current reports. Confirm that no other chat/worker owns the same card or source files.
2. Confirm its condition, official Train SHA, expected rows, views, support, exact predictor-fit count, dependencies, memory/RAM estimate, and remaining campaign budget. No Test/Kaggle access.
3. If no adapter/runner exists, implement it only in the card's new namespace. Reuse shared harness functions, preserve frozen files, use an explicit feature allowlist, and keep all learned preprocessing inside each temporal fit.
4. Run syntax checks, relevant tests, and a synthetic dry-run through adapter, aggregation, independent verification, and gate. Include string IDs, tied values/bins, missing or empty groups, zero support, and alignment. Do not start an official fit until this passes.
5. Freeze the exact source, recipe, Train/control/dependency hashes, folds, support/fallback rules, budget, and gates before scoring. Commit/push a coherent, validated protocol if required by the current repository procedure; never freeze after seeing candidate scores.
6. Reserve the complete card budget. Run only its declared fits, one at a time, with two threads. Preserve each successful fit, failed attempt, retry, and elapsed time distinctly. If a job takes more than 15 minutes, inspect progress and resources; an observation timeout alone is not a failure. Do not impose an arbitrary wall-clock stop.
7. Reconstruct predictions and metrics independently. Check IDs/rows, leakage boundaries, preprocessing fit rows, policy, metrics, subgroup support, and gates. Save a report even for RETIRE or SKIPPED, record the reason, update `BATCH_FINDINGS.md`/ledger idempotently, then continue to the next eligible card.
8. Keep card files and changes in an isolated branch/worktree when another chat is active. Workers may implement/review separate files, but MAIN owns final integration. Do not let two chats run model fits concurrently: a lock in a different worktree is not a shared lock. If the active fit belongs to another chat, do read-only/preparation work or wait for that fit to finish before reserving/running a new one.

For eligible candidates that pass their pilot gates, follow only the card/handbook's declared continuation rules and remaining budget. Do not use public scores for model search or threshold/weight probing. Finalist inference/submission is conditional on the exact V3 gates and fresh authorization checks described in `LUNA_HANDOFF.md`; it is not a routine next step after a pilot.

## Stop only for global blockers

Do not stop the whole queue for a card-local negative result. Retire or skip that card and continue. Stop official fits and preserve a resumable checkpoint only for a global source/control/Train hash mismatch, leakage or row-identity mismatch, corrupt/ambiguous frozen cache, unavailable or unaccounted budget, unresolved concurrent-training ownership, or a conflict in governing instructions. Report the exact evidence for a global blocker; never repair it by changing a frozen hash or relaxing a gate.

## Ready-to-paste Luna prompt

> Lanjutkan V3 research di repository `samindriano/tk2-kasdad` memakai Luna MAIN dan Orchestra yang terpasang. Baca `reports/optimization_v3/research_backlog/LUNA_HANDOFF.md`, dokumen `LUNA_OVERNIGHT_LANE_PLAN.md`, dan kartu canonical untuk setiap ID. Verifikasi HEAD, status Git, branch/worktree, ledger, report, cache, source hash, dan ownership aktual; dokumen lama hanya snapshot. Selesaikan lane R04 yang sedang berjalan berdasarkan status report sekarang—jangan mengulang fit atau verdict yang sudah tersimpan—lalu teruskan kartu yang benar-benar eligible dalam urutan queue.
>
> Untuk setiap kartu yang implementasinya belum ada, buat adapter dan runner baru, uji sintaks, jalankan tes dan synthetic end-to-end sampai report aggregation, independent verification, dan gate, lalu freeze recipe/source/input/dependency/gates sebelum official Train fit. Jalankan hanya fit yang dideklarasikan setelah resource, support, dependency, dan budget preflight lulus. Gunakan dua thread maksimal dan hanya satu job training global pada satu waktu. Jika chat/worktree lain sedang training, lanjutkan read-only/implementation/preparation atau tunggu; jangan mulai job kedua. Jangan pernah mengarang command script sebelum runner-nya dibuat dan diuji.
>
> Gunakan hanya official Train saat pilot dan continuation; jangan membuka Test/SampleSubmission atau Kaggle pada tahap tersebut. Setelah kandidat lolos full-six, supported-group, source-QA, dan gate finalist, lanjutkan full-Train refit/inference/submission hanya jika pemeriksaan live V3 authorization, team, rules, deadline, shared quota, global budget, manifest, dan idempotency semuanya lulus. Jangan probing leaderboard atau mengubah model berdasarkan public score. Jangan mengubah EDA/notebook, pekerjaan anggota lain, frozen protocol, queue, threshold, weight, atau gate setelah melihat skor. Saat gate lokal valid gagal, simpan bukti RETIRE/SKIPPED, perbarui ledger dan `BATCH_FINDINGS.md`, lalu lanjutkan ke lane eligible berikutnya tanpa meminta approval tiap milestone. Continue hanya sesuai continuation dan budget resmi. Jangan berhenti karena timeout pengamatan; cek progres/resource dan lanjutkan. Stop hanya untuk blocker global yang dicantumkan dalam runbook, simpan checkpoint, dan laporkan bukti. Commit/push hanya file lane yang sudah tervalidasi, tanpa force-push atau menggabungkan perubahan chat lain. Akhiri setelah antrean riset yang eligible selesai atau ada blocker global nyata, dengan semua ID, verdict, denominator, metrics, fits/attempts, source hashes, QA, keterbatasan, dan next action tercatat.
