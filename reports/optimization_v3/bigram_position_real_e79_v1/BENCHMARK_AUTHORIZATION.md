# CLS Bigram Position REAL E79 exploratory benchmark

The user authorized one Classification E79 exploratory benchmark: one new
real-position bigram gamma fit on all 50,608 official Classification Train rows,
followed by Test feature inference. The user authorized no Kaggle upload.

This is a new benchmark family and namespace. The completed
`POS_MOMENTS_REAL_E79_EXPLORATORY_BENCHMARK` is a different representation and
is not this fit. It is not reused as the new gamma and does not consume this
authorization. The six-fit `CLS_BIGRAM_POSITION_PILOT_V1` remains historical
development evidence with its original diagnostic-only gate failure intact.

The authorized recipe adds exactly one REAL three-bin known-bigram gamma to the
full-Train C2_B bag logit plus the saved original full-Train unigram-position
gamma. It mixes the resulting present-stream probability 50:50 with the original
A02 E79 probability, routes missing streams to the frozen C1_B HGB probability,
and uses threshold 0.50. The original Train-only TF-IDF, C2_B, original POS
gamma, and C1_B states may be reused only after receipt and artifact hashes,
fit populations, and feature compatibility pass. No BAG/SHUFFLED E79 fit,
recipe search, retry, model selection by Test, or Kaggle action is in scope.

The benchmark must complete synthetic lifecycle and persistence/resume QA before
the official fit. The official fit is charged once in the append-only shared
Research Next ledger. After the fitted-state freeze is committed, Test may be
opened for inference only. Test labels and SampleSubmission are not needed.
