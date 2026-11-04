# C10 Trigram E79 exploratory benchmark

**Status:** authorized bounded benchmark; no Kaggle upload from this lane.

Sam asked Luna 1 to continue after the 45-card audit. Sol assigned one C10
trigram full-Train benchmark using the original frozen recipe. This is separate
from the already completed POS_MOMENTS E79 benchmark.

C10's historical pilot verdict remains **RETIRE**. Its primary gain was
`+0.000342846` versus Sparse25, below the frozen `+0.0005` gate; its H12 gain
was `+0.000878127`. The primary value is not a gain versus MIX50. This E79
artifact is an outcome-adaptive full-Train inference build, not fresh validation
evidence or model promotion.

The benchmark uses one full-Train C10 predictor, up to five Train-only
preprocessing states, exact existing Sparse25 C2_B and CatBoost missing-stream
probabilities, and the original fixed blend/threshold. It performs no C2_B or
CatBoost refit, no gate change, no recipe search, and no Kaggle contact. Sol
reviews the CSV and QA before any upload decision.
