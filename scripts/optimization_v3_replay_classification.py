"""Fresh accepted V3 classifier refit from raw CSVs, without prediction caches."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.prepare_initial_submissions import _load_inputs
from src.models.classification_models import build_classifier
from src.models.classification_round2_models import stream_missing_mask
from src.models.campaign_classification_catboost import build_catboost_classifier
from src.models.optimization_v3_sparse_classification import SparseTextBoostClassifier
from src.models.round1_evaluation import array_digest


def replay(output: Path, receipt: Path) -> dict:
    output = output.resolve()
    receipt = receipt.resolve()
    output.relative_to((ROOT / "experiments/runs/optimization_v3_replay").resolve())
    receipt.relative_to(ROOT)
    assert not output.exists() and not receipt.exists(), "Exclusive replay outputs required"
    manifest_path = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["recipe"] == {
        "kind": "availability_present_blend",
        "present": [{"model": "C2_B", "weight": .75},
                    {"model": "SPARSE_BINARY", "weight": .25}],
        "missing": "CB_MISSING", "threshold": .5,
    }
    for path, expected in manifest["source_code_sha256"].items():
        actual = hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert actual == expected, path
    train, test, sample, hashes = _load_inputs("classification")
    assert hashes == manifest["input_sha256"]
    assert len(train) == 50608 and len(test) == 29421
    assert not {"artifact_signal", "threat_rank"}.intersection(test.columns)
    assert sample.record_id.equals(test.record_id)
    models = [build_classifier({"id": "C2_B"}), build_catboost_classifier(),
              SparseTextBoostClassifier()]
    scores = []
    state_checks = []
    start = time.perf_counter()
    for model, expected in zip(models, manifest["component_audits"]):
        fields = list(getattr(model, "input_feature_names", getattr(model, "feature_names", ())))
        assert fields and not {"record_id", "artifact_signal", "threat_rank"}.intersection(fields)
        with threadpool_limits(limits=2):
            model.fit(train[fields].copy(), train.artifact_signal.to_numpy(copy=True))
            state = model.audit_state()
            probability = model.predict_proba(test[fields].copy())
        assert state == model.audit_state(), "Prediction mutated fitted state"
        if expected["fit_component"]["model"] == "CB_MISSING":
            # CatBoost's binary container includes variable run metadata.
            keys = ["fit_rows", "fit_data_sha256", "parameters", "native_category_features",
                    "output_feature_names", "fit_only_preprocessor", "tree_count"]
            assert all(state[key] == expected["state"][key] for key in keys)
        else:
            assert state == expected["state"], "Fresh deterministic state differs"
        assert array_digest(probability) == expected["prediction_sha256"]
        scores.append(probability)
        state_checks.append({"model": expected["fit_component"]["model"],
                             "fit_lineage_and_preprocessing_match": True,
                             "prediction_bit_identical": True})
        print("FRESH FIT", expected["fit_component"]["model"], flush=True)
    missing = stream_missing_mask(test)
    probability = np.where(missing, scores[1], .75 * scores[0] + .25 * scores[2])
    prediction = (probability >= .5).astype(np.int8)
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    assert array_digest(probability) == manifest["probability_sha256"]
    assert array_digest(prediction) == manifest["prediction_sha256"]
    assert len(np.unique(test.record_id)) == 29421 and not test.record_id.isna().any()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"record_id": test.record_id, "artifact_signal": prediction}).to_csv(output, index=False)
    csv_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    assert csv_hash == manifest["csv_sha256"]
    result = {
        "status": "PASS", "at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "v3_cls_sparse25", "accepted_ref": "57837821",
        "accepted_source_commit": manifest["source_commit"],
        "fresh_full_train_fits": 3, "train_rows": len(train), "test_rows": len(test),
        "input_sha256": hashes, "component_checks": state_checks,
        "ignored_cache_or_prior_csv_read": False, "model_binary_read_or_written": False,
        "test_labels_used": False, "kaggle_API_called": False,
        "probability_bit_identical_to_accepted": True,
        "prediction_bit_identical_to_accepted": True,
        "csv_byte_identical_to_accepted": True,
        "probability_sha256": array_digest(probability),
        "prediction_sha256": array_digest(prediction), "csv_sha256": csv_hash,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "replay_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - start,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("PASS fresh raw-data replay; seconds", round(result["seconds"], 1), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    replay(args.output, args.receipt)
