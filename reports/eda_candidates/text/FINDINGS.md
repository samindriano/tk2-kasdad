# Text EDA candidate evidence

Inputs: Regression Train SHA256 756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082; Classification Train
SHA256 875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c. Both contain 50,608 unique IDs; all shared
feature columns were compared by ID before joining separate labels. This analysis
reads only the two official Train files. No raw token, identifier, model prediction,
or Kaggle score is exported.

## Candidate T1 — lexical variety at similar text length

**Question:** Pada stream dengan jumlah token yang sebanding, apakah proporsi
token unik berkaitan dengan proporsi artifact_signal?

The hypothesis concerns repetition versus variety, not token meaning.
Threat_rank is a secondary counter-outcome, not part of the main question. Five
token-count bands are chosen without labels. Within each band, compare the lower
and upper within-band average-rank tails (rank <= 0.25 vs >= 0.75), then inspect
per-cycle direction. In the shortest band, every row has the same unique-token
fraction; no high-diversity tail exists there, so no contrast is computed for
that band. The other 4 length bands have 37,562 rows.
Artifact-rate differences across them are all positive, ranging
0.6% to
24.2%.

A stricter check fixes the **exact token count** and compares the same within-count
average-rank tails. It requires at least 120 observations per count and at least
20 in each tail. For unique-token fraction, support is
71 counts/37,681 rows,
or 79.8% of stream-present and
74.5% of all Train. The high-diversity tail
has artifact prevalence +14.2%
9,011 lower-tail and
9,354 upper-tail observations are compared.
Their respective artifact prevalences are
24.2% and
38.4% when each exact-length
stratum is weighted by its full count. Across the individual compared tail
observations, prevalences are 24.3%
and 38.8%, a difference of
14.5%. Mean threat differs by
-2.75 points.
The artifact contrast is positive for
64/71
counts covering 92.8%
of the supported rows. Two related measures show the same direction at exact
length: within-row Shannon entropy +10.0%,
and lower repeated-token share +14.9%.
There is meaningful heterogeneity across token counts; the observed exact-count
artifact contrasts span -20.6% to
+55.3%.
Changing the minimum exact-length support threshold from 80 to 120 to 200
retains 84,
71, and
53 counts, respectively; each has at
least 20 observations in both tails. Their support-weighted artifact contrasts
are 14.6%,
14.2%, and
13.5%.

With broad label-blind length bands, the artifact contrast is positive in
30/33 cycle-specific estimates. Resampling the
33 cycle summaries gives a 90% sensitivity range of
10.2% to
15.3% for the
average cycle contrast, conditional on exchangeability. This is not a CI for
future cycles. Exact-length contrasts within cycle pass the stricter support
rule in only 5
cycles, so simultaneous exact-length and cycle adjustment remains unsupported.
All contrasts are exploratory; support and effect ranges are in stats.json
and length_stratified_diversity.csv.

## Candidate T2 — agreement of text_load with observed text size

**Question:** Seberapa baik text_load mewakili jumlah token dan karakter yang
benar-benar tersedia dalam signal_stream?

The observed Spearman association is 0.983
with regex token count and 0.983
with character count, suggesting a strong length proxy. The feature-level check
may be too obvious to merit the assigned EDA slot. See text_load_coherence in
stats.json, its quartile table and text_length_coherence.png.
Reported text_load quantiles are separately labeled for all Train rows and for
stream-present rows; text-load quartiles use only the latter (47,204).

## Initial overlap screen

Rousan studies entry_friction and impact_load for artifact_signal; those fields
are absent here. Abhi studies scope_pressure conditional on impact_load for
threat_rank; both are absent. Rizky studies telemetry_flag and module_spread
for artifact_signal; both are absent. Samuel's existing question centers on
impact_load–threat_rank across cycle periods; no such pair is analyzed here.
Cycle appears only as a robustness/support dimension. Sharing a target does not
by itself make these questions equivalent.

## Figures and interpretation limit

- text_length_coherence.png: numeric text_load versus realized regex token count,
  and how a simple unique-token fraction depends on stream length.
- diversity_by_length_and_targets.png: artifact-rate contrasts by
  label-blind length band plus per-cycle sensitivity; threat means are included
  as a secondary counter-outcome.

Regex is lowercased ASCII alphanumeric runs. These measures do not assign
meaning to anonymized tokens. All Train cycles are development data already
seen by prior work; findings remain exploratory rather than confirmatory.
