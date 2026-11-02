"""Synthetic-only E2E, save/readback/resume and corruption preflight."""
from __future__ import annotations

import tempfile
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_continuation as runner


def synthetic_case() -> tuple[dict, dict, dict, dict]:
    origin = 59
    donor_rows = np.arange(10, dtype=np.int64)
    query_rows = np.arange(10, 14, dtype=np.int64)
    ids = np.asarray([f"D{i}" for i in donor_rows] + [f"Q{i}" for i in query_rows], dtype="U8")
    cycles = np.asarray([56, 56, 57, 57, 58, 58, 59, 59, 59, 59, 60, 61, 62, 63], dtype=np.int64)
    present = np.ones(14, dtype=bool)
    present[-1] = False
    train = {"rows": np.arange(14, dtype=np.int64), "ids": ids, "cycles": cycles, "present": present}
    qbase = np.asarray([20., 30., 40., 50.])
    r06 = {
        55: {"rows": donor_rows.copy(), "prediction": np.linspace(18., 27., 10)},
        59: {"rows": query_rows.copy(), "prediction": qbase.copy()},
    }
    nr = np.full((4, 32), -1, dtype=np.int64)
    ni = np.full((4, 32), "", dtype="U16")
    sim = np.zeros((4, 32), dtype=np.float64)
    count = np.asarray([5, 5, 5, 0], dtype=np.int16)
    for i in range(3):
        source = np.arange(5, dtype=np.int64)
        nr[i, :5] = source
        ni[i, :5] = np.asarray([f"str_:D{x}" for x in source], dtype="U16")
        sim[i, :5] = np.asarray([.99, .9, .8, .7, .6])
    geom = {"score_rows": query_rows, "score_record_ids": ids[query_rows],
            "score_cycles": cycles[query_rows], "neighbor_count": count,
            "neighbor_source_rows": nr, "neighbor_ids": ni, "neighbor_similarities": sim}
    donor = {"rows": donor_rows, "ids": ids[donor_rows], "cycles": cycles[donor_rows],
             "target": np.asarray([24., 25., 26., 27., 28., 29., 30., 31., 32., 33.])}
    return train, r06, geom, donor


def run() -> dict:
    train, r06, geom, donor = synthetic_case()
    freeze = {"freeze_signature": "a" * 64}
    payload, meta = runner.build_payload(59, "observed", train, r06, geom, donor, freeze, "synthetic-commit")
    assert np.array_equal(payload["baseline_prediction"], np.asarray([20., 30., 40., 50.]))
    assert np.array_equal(payload["prediction"][-1:], payload["baseline_prediction"][-1:])
    # This makes origin-cache semantics observable: the donor OOF values are different.
    assert not np.array_equal(payload["baseline_prediction"][:3], r06[55]["prediction"][:3])
    with tempfile.TemporaryDirectory(prefix="rnbr-continuation-v1-") as temp:
        prior = runner.RUN_ROOT
        try:
            runner.RUN_ROOT = Path(temp)
            fit_id = "RNBR_CONT_E59_OBSERVED"
            receipt = runner.persist_bundle(fit_id, payload, meta)
            saved, saved_receipt = runner.load_bundle(fit_id, freeze, source_commit="synthetic-commit")
            assert saved_receipt["npz_sha256"] == receipt["npz_sha256"]
            assert np.array_equal(saved["prediction"], payload["prediction"])
            # Resume is read-only: a verified durable bundle is reused, never rebuilt.
            try:
                runner.persist_bundle(fit_id, payload, meta)
            except runner.ContinuationError:
                pass
            else:
                raise AssertionError("Existing bundle was overwritten on resume")
            with (Path(temp) / "bundles" / fit_id / "arrays.npz").open("ab") as f:
                f.write(b"corruption")
            try:
                runner.load_bundle(fit_id, freeze, source_commit="synthetic-commit")
            except runner.ContinuationError:
                pass
            else:
                raise AssertionError("Corrupt bundle was accepted")
        finally:
            runner.RUN_ROOT = prior
    report = {"status": "PASS", "dataset": "synthetic only", "origin": 59,
              "query_rows": 4, "donor_rows": 10,
              "checks": ["payload formula", "origin-cache query baseline distinct from donor OOF",
                         "missing fallback exact", "pickle-free safe Unicode save/readback",
                         "resume never overwrites", "corruption rejected"]}
    report_path = ROOT / "reports/optimization_v3/rnbr_continuation_v1/pre_execution_qa.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
