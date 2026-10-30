# Kaggle initial submission results

Queried 30 October 2026 after one submission per authorized competition. Kaggle history was checked first and returned no existing submissions. The CLI authenticated through the Kaggle user's existing token credential; no credential value is stored here.

| Track | Kaggle ref | Submitted (UTC) | Status | Public score | Message |
|---|---:|---|---|---:|---|
| Classification, C4_HGB route | 57808225 | 2026-10-30 14:27:27.577 | COMPLETE | 0.82641 | Samuel C4-HGB Routing Initial Benchmark |
| Regression, R1_D expanded HGB | 57808231 | 2026-10-30 14:27:38.327 | COMPLETE | 13.53473 | Samuel R1-D Expanded HGB Initial Benchmark |

Both requests were accepted and reached COMPLETE status. Kaggle's submission-history response contains the public scores shown above; private scores remain unavailable. No additional uploads or leaderboard-guided changes were made.

## Internal versus Kaggle evidence

These submissions are initial public-test benchmarks. For context, the already completed temporal-development Round 2 classification control-route result at fixed threshold 0.50 was primary Macro F1 0.825481 and pooled H12 Macro F1 0.796810; that metric and validation data are distinct from the Kaggle public score. Regression used the Round 1 R1_D recipe, whose Round 1 development primary MAE was 13.349836 and pooled H12 MAE 15.5772. The Kaggle score is recorded here as a separate initial benchmark, not a model-selection result.

The installed CLI followed Kaggle's supported API token file authentication described in the [official Kaggle CLI authentication guide](https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md). Manifest files contain input byte hashes, source commit, model/protocol identity, full-Train fit counts, output SHA-256, runtime identity, prediction ranges/counts and submission IDs/status. CSV files remain local and gitignored.
