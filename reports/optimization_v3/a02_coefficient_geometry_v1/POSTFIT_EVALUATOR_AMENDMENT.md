# CGEOM_V1 — catatan amendment verifier pascafit

Status kandidat: `CLOSED_GATE_FAIL`. Gate tetap sesuai recipe asli; amendment tidak mengubah gate atau prediksi.

## Perbaikan terbatas

Verifier awal membandingkan `query_record_ids` hasil NPZ Unicode dengan array object yang berasal dari pandas. Semua ID sumber dan output memiliki shape, nilai, dan urutan yang sama. Adapter mengonversi hanya ID ekspektasi yang berasal dari source menjadi NumPy Unicode sebelum pemeriksaan shape/dtype/nilai frozen; larangan stored object arrays, pemeriksaan numerik, toleransi 1e-12, bank, donor, cycle, serta mass check tetap aktif.

Frozen training source: `ddd64736155a64e70c5453c921aedcbd910397f8`; amendment evaluator: `ea43b14e92ce7eca6cfedeb1ca212d7531c4f6ea`.
Adapter SHA-256: `ead86e38b56162b59fed527565383a199e88b06ed9764b87099f44e2fb77e4bf`; tests SHA-256: `f77e7301add53993149cebe39a0414b0e4593fc016be881bb75a425c9b6bbcd8`.
Original verifier SHA-256: `cf851381d124bef7f6bf478d3be967529171336650545ab6d18c9bdead7fd3c9`; freeze: `fae503bf65ab98ae4847249e808caafc514c34c1e2fbf688a1d4236c7f0b8eaa`.
Prelabel SHA-256: `7818eadefb9eef3c015d2da7c20d63847611398bfd8a7b7d6a19ba8aa60f6454`; independent score verifier: `eef28d7010a6bb01d5944985e63a0185075d94a9536847123daa8579eb1d4394`.
Completion event SHA-256: `fb84d5a27ccab83bc2d6b0b0f3dbf025fc96f19065c8166d3e3d9c544411f31b`; final ledger SHA-256: `5bd8aa218e17268913de9ecdac5b85ea576e611cde5566f2b3d242f371ff317a`.

## Batas dan status

Tidak ada fit, refit, predictor reconstruction baru, perubahan artefak prediksi, Test, Kaggle, atau notebook. Query labels dibuka hanya setelah rekonstruksi numerik/boundary independen PASS. Hasil memakai reused development labels dan berstatus diagnostic-only; hasil tidak mengubah champion atau memberi izin submission.

Laporan metrik dan gate lengkap ada di `RESULTS.md`; machine-readable details ada di `RESULTS.json`, `PRELABEL_VERIFICATION.json`, dan `SCORED_VERIFICATION.json`.

### Hash laporan utama

- `RESULTS.md`: `7cc8bb1049f08d515c42d1eeefb9c80e101cadb71a0783843910fd08c8f85b30`
- `RESULTS.json`: `db8aa20464aa2f273998e553aab967074372eff630734c7106311b807c203552`
- `PRELABEL_VERIFICATION.json`: `7818eadefb9eef3c015d2da7c20d63847611398bfd8a7b7d6a19ba8aa60f6454`
- `SCORED_VERIFICATION.json`: `eef28d7010a6bb01d5944985e63a0185075d94a9536847123daa8579eb1d4394`
- `COMPLETION_RECEIPT.json`: `d3ea32574e9696e91b8b47cb7767b27f44e89026596e611adcce94b8674f59cc`
