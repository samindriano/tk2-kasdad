# RNBR E79 observed-memory benchmark

Status: local full-Test CSV prepared for manual review; no Kaggle upload.

- Fit ID: `RNBR_FINAL_E79_OBSERVED` (one consumed continuation slot).
- Freeze: `9ce279a5b7dad7b47f0a70a41aaed490c1f824d214482a0b460027c0a2ccaa1e`.
- Donors: 36,010 present-stream Regression Train rows, cycles 56–79.
- Test rows: 29,421; exact SampleSubmission schema and ID order.
- Supported neighbor rows: 28,358; exact R06 fallback rows: 1,063.
- Changed R06 predictions: 28,355. No Test target was opened.
- CSV: `submissions/optimization_v3_rnbr_final_e79_v1/samuel_rnbr_final_e79.csv` SHA-256 `54316cd4806c1957a87d46806b5eacb59a6f35287d8f31411f12139f4114362c`.
- Independent verification: PASS; all 29,421 query geometries were recomputed. See `reports/optimization_v3/rnbr_final_e79_v1/verification.json`.
- CSV serialization: 17 significant digits; maximum raw-array round-trip difference `2.84e-14`, below the explicit `5e-14` QA tolerance. Raw prediction and CSV hashes match their manifests. See `reports/optimization_v3/rnbr_final_e79_v1/csv_serialization_audit.json`.
- This is an unscored local benchmark artifact; no Test MAE or Kaggle score is available, and no champion was changed.
