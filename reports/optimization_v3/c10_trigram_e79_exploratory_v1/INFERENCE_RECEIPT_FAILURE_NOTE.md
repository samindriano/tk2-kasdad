# Inference receipt completion

The one Test inference completed after the trained-state freeze and wrote the
CSV and prediction NPZ. The original runner then stopped only while computing
the receipt because it passed an unsupported `dtype=` keyword to
`array_digest`. The existing Test predictions were not recomputed. A separate
receipt finalizer now checks the saved probabilities, exact blend/fallback,
IDs, CSV values, and file hashes, then writes the missing inference receipt.
No Kaggle API, CLI, browser, SampleSubmission, or Test labels were used.
