# Sol final decision — MIX50 endgame complete

At 01:12 WIB on 4 November 2026, Kaggle confirmed submission **57956714**
`classification_mix50_e79.csv` as **COMPLETE**, with public Macro F1 **0.83144**.
This is +0.00021 over positional offset (0.83123), the previous best observed
public score. It is +0.00214 over A02 (0.82930). Private performance is unknown.

## Decision and evidence

Sol approved exactly one upload after reviewing the frozen policy implementation,
rerunning the independent policy verifier (PASS, 195 checks), and running the
focused tests (10 passed). Both policies qualified; MIX50 was selected by the
priority committed before their metrics were computed, rather than maximizing
their observed development scores. ADD_RESIDUAL was not exported or submitted.

MIX50 uses the original positional route pP unchanged for missing streams and
0.5*pP + 0.5*pA for present streams, with threshold >=0.50. E67 H12 Macro F1
improved from 0.806258 to 0.811937 (+0.005679); E75 W6 improved from 0.811443 to
0.816099 (+0.004657). All three E67 windows improved against pP, and all 40
supported group/view cells stayed within the frozen harm limits. E75 MIX50
still trails A02 (0.819604); the combination does not dominate every comparator.

Sol independently reconstructed the E79 formula from the immutable prior batch
probability bundle, checked its hash against the prior manifest, official Test
IDs/order/cycles/stream mask, bounded finite probabilities, historical POS/A02
labels, and the final CSV schema. There are 29,421 unique ordered IDs, 6,800
positive predictions, and 1,062 missing-stream rows. The new CSV differs on 352
labels from POS and 484 from A02. Live pre-upload history contained no MIX50
submission, and Luna's local label-vector scan found no identical existing CSV.

CSV SHA-256:
`48dcbd3f51fa4567e8f3f3a6838efdfaa50f4c2129e628e1b597436918c46ce7`.

The upload used the already authenticated official Kaggle CLI, once, with message
`Samuel MIX50 Positional A02 Frozen Endgame`. API scoring confirmation, upload
intent/response, pre-upload history, and Sol's independent candidate review are
stored beside this report. No credentials are included.

## Interpretation and stop point

The fixed combination improves the examined internal views and has a small
observed public improvement. This is consistent with useful complementarity,
but does not isolate positional information from learner/blending/calibration
effects. The experiment was chosen after seeing earlier public results;
development labels have been reused, E67/E75 views overlap, and public results
are not an untouched confirmation. No public confidence interval or guaranteed
private improvement is asserted.

Recommendation for review: retain MIX50 as the best-public-observed reference,
keep original POS and A02 artifacts as alternatives, and review final selection
separately. No final leaderboard slot was selected or changed. This is not an
instruction to submit all three or a determination of final competition rules.

No new fits, model changes, ledger events, EDA/notebook edits, or Test-label access
were required. Historical artifacts and failed verdicts remain intact. The one
authorized endgame upload is consumed. No further weight/threshold search,
policy export, or submission is triggered by this score. The handoff watchdog
was deleted after confirmed scoring; Luna receives the completed-result handoff
and stops. Changes are committed locally only; no push is made in this scope.

Evidence: `KAGGLE_SCORE_20261104.json`, `KAGGLE_UPLOAD_EVENTS_20261104.jsonl`,
`KAGGLE_HISTORY_PREUPLOAD_20261104.json`, `SOL_CANDIDATE_REVIEW.json`,
`E79_SELECTED_MANIFEST.json`, `E79_INDEPENDENT_QA.json`,
`INDEPENDENT_VERIFICATION.json`, and `FINAL_HANDOFF.md`.
