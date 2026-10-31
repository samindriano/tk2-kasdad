# R09 Runtime Observations

- Frozen change: the existing SVD32 LightGBM recipe was retained, with objective `huber` and `alpha=0.9`; the fixed prediction route and all feature/preprocessing choices were unchanged.
- Both cutoff fits completed and were independently verified: 12,020 rows at cutoff 55 and 29,705 rows at cutoff 67; 400 trees and 32 SVD components per fit. Combined recorded fit time was 11.894 seconds. No adapter fit failed.
- LightGBM printed the same warning once per fit: `Met negative value in categorical features, will convert it to NaN`. The fitted-state receipt confirms the existing categorical sentinel `-1` is treated as missing. Both fits were saved successfully; this warning did not change the frozen recipe.
- Estimated peak memory was 2.08 GiB at cutoff 55 and 2.76 GiB at cutoff 67. These are conservative preflight estimates; peak process RSS was not observed.
- The primary view declined from 13.464396 to 13.500821 MAE (gain -0.036425), and H12 declined from 14.047533 to 14.129187 (gain -0.081655). W1 improved by 0.074223 MAE; W4 worsened by 0.128387. H12_E improved by 0.029750 while H12_L worsened by 0.175908.
- All 12 supported target-tail guard slices failed: `threat_rank <= 5` and `>= 95` across W1, W4, primary, H12_E, H12_L, and H12. Each had at least 205 rows. No supported missing-stream group appears among the recorded failures.
- Decision: **RETIRE** under the frozen gates. The primary gain missed the required +0.03, H12 exceeded the allowed 0.05 MAE harm, and supported target-tail groups were materially worse. The generic independent verifier passed. Regression Test, SampleSubmission, and Kaggle were not accessed.
