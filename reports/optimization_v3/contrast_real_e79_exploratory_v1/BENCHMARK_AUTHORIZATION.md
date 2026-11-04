# POS_CONTRAST REAL E79 exploratory benchmark

The user explicitly authorized exactly one Classification E79 benchmark after
reviewing the earlier positional pilot. This authorizes one new REAL contrast
theta fit on all 50,608 official Classification Train rows, followed by local
inference from official Test features. It does not authorize Kaggle upload.

The historical `POS_CONTRAST_V1` pilot remains `PILOT_GATE_FAILED`. This new
namespace does not revise that verdict, reuse the old released conditional
reservation, add a fit retry, or change any existing model/checkpoint. It uses a
new fit ID and a new reservation key under the current live budget.

The one fitted parameter is the contrast theta. Existing full-Train C2_B,
TF-IDF, A02, and C1_B states are reused only after their saved recipes, receipts,
fit populations, and artifact hashes pass verification. The exact formula,
optimizer, route, threshold, output schema, and boundaries are in `RECIPE.json`.

Test features are opened only after the trained-state freeze is committed. Test
labels are never read. SampleSubmission is used only for header and ID order
verification. The CSV stays local for Sol's later review and upload decision.
