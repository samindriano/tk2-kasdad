"""Row-matched R19 revision after the original late-view control failed closed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import scripts.optimization_v3_batch1_r19 as r19_v1
from src.models.optimization_v3_batch1 import BatchIntegrityError, file_sha256, json_sha256


ROOT = Path(__file__).resolve().parents[1]
REVISION_DIR = ROOT / "experiments/optimization_v3/batch1/R19/revision_02"
RECIPE_PATH = REVISION_DIR / "recipe.json"
FREEZE_PATH = REVISION_DIR / "freeze.json"
ATTEMPT1_PATH = ROOT / "reports/optimization_v3/batch1/R19/attempt1_control_mismatch.json"
RESULTS_PATH = ROOT / "reports/optimization_v3/batch1/R19/results.json"
VERIFY_PATH = ROOT / "scripts/optimization_v3_batch1_r19_revision2_verify.py"
HISTORICAL_REPORT = ROOT / "reports/optimization_v2/regression_repr/fullsix_verification_context.json"
CHAMPION_MANIFEST = ROOT / "reports/optimization_v2/kaggle/v2_reg_sparse_lgb25.json"
CHAMPION_CHECKPOINT = ROOT / "experiments/optimization_v3/champions_checkpoint1.json"
EXPECTED_METRICS: dict[str, float] = {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"Cannot read R19 revision 2 input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchIntegrityError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise BatchIntegrityError(f"Refusing to overwrite different R19 revision 2 artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _prepare() -> dict[str, Any]:
    global EXPECTED_METRICS
    context = r19_v1._prepare()
    historical = _read_json(HISTORICAL_REPORT)
    manifest = _read_json(CHAMPION_MANIFEST)
    checkpoint = _read_json(CHAMPION_CHECKPOINT)
    if historical.get("official_train_sha256") != r19_v1.TRAIN_SHA256:
        raise BatchIntegrityError("Historical full-six verification uses a different official Train")
    if manifest.get("candidate_id") != "v2_reg_sparse_lgb25":
        raise BatchIntegrityError("Pinned champion manifest is not the R19 sparse-LGB25 champion")
    regression_checkpoint = checkpoint.get("regression", {})
    if regression_checkpoint.get("manifest") != "reports/optimization_v2/kaggle/v2_reg_sparse_lgb25.json":
        raise BatchIntegrityError("Latest V3 checkpoint does not identify the R19 champion manifest")
    try:
        w4 = historical["recomputed_views"]["W4"]["lgbm_sparse_l1"]["blend25"]
        w6 = historical["recomputed_views"]["W6"]["lgbm_sparse_l1"]["blend25"]
        h12 = historical["recomputed_horizons"]["H12_L"]["lgbm_sparse_l1"]["blend25"]
        if (int(w4["n"]) != int(context["row_metadata"]["W4"]["rows"])
                or int(w6["n"]) != int(context["row_metadata"]["W6"]["rows"])
                or int(h12["n"]) != int(context["row_metadata"]["H12_L"]["rows"])):
            raise BatchIntegrityError("Historical champion metric denominators differ from R19 late windows")
        primary = (int(w4["n"]) * float(w4["mae"])
                   + int(w6["n"]) * float(w6["mae"])) / (int(w4["n"]) + int(w6["n"]))
        h12_l = float(h12["mae"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BatchIntegrityError(f"Cannot derive row-matched R19 champion metrics: {exc}") from exc
    EXPECTED_METRICS = {"primary": primary, "H12_L": h12_l}

    recipe = dict(context["recipe"])
    recipe.update({
        "revision": 2,
        "control_primary_mae_rounded": round(primary, 12),
        "control_H12_L_mae_rounded": round(h12_l, 12),
        "baseline_metric_provenance": {
            "current_champion": "v2_reg_sparse_lgb25",
            "champion_manifest": str(CHAMPION_MANIFEST.relative_to(ROOT)),
            "champion_checkpoint": str(CHAMPION_CHECKPOINT.relative_to(ROOT)),
            "historical_row_matched_report": str(HISTORICAL_REPORT.relative_to(ROOT)),
            "historical_report_sha256": file_sha256(HISTORICAL_REPORT),
            "late_W4_rows": int(w4["n"]), "late_W4_mae": float(w4["mae"]),
            "late_W6_rows": int(w6["n"]), "late_W6_mae": float(w6["mae"]),
            "late_primary_rows": int(w4["n"]) + int(w6["n"]),
            "late_H12_L_rows": int(h12["n"]), "late_H12_L_mae": h12_l,
            "superseded_card_reference": {
                "primary_mae": 11.559390, "H12_L_mae": 14.047533,
                "interpretation": "Ridge25/fullsix reference; not row-matched to the late W4+W6 champion comparator",
            },
            "selection_note": (
                "Metrics are from the pre-existing independent full-six verification report for the accepted sparse-LGB25 blend. "
                "The first R19 freeze compared late windows with fullsix Ridge25 values and stopped before fitting offsets or scoring a candidate."
            ),
        },
    })
    identity = dict(context["identity"])
    input_hashes = dict(identity["input_sha256"])
    for path in (HISTORICAL_REPORT, CHAMPION_MANIFEST, CHAMPION_CHECKPOINT):
        input_hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    identity["input_sha256"] = input_hashes
    source_hashes = dict(identity["source_sha256"])
    for path in (Path(__file__).resolve(), VERIFY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"R19 revision 2 source is missing: {path}")
        source_hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    identity["source_sha256"] = source_hashes
    identity["recipe"] = recipe
    context.update({"recipe": recipe, "identity": identity,
                    "signature": json_sha256(identity)})
    return context


def _record_superseded_attempt() -> None:
    _write_once(ATTEMPT1_PATH, {
        "card_id": "R19", "attempt": 1,
        "status": "INVALID_CONTROL_WINDOW_MISMATCH_NOT_SCORED",
        "freeze_signature": "453651c3df7e66af43eed3434fa67434483512bb36360ab403dfb2a9fda6dadf",
        "reconstructed_late_primary_mae": 10.477525111,
        "incorrect_fullsix_control_primary_mae": 11.559390,
        "candidate_offsets_fitted": False,
        "candidate_metrics_computed": False,
        "test_or_sample_access": False, "kaggle_access": False,
        "explanation": (
            "The original freeze incorrectly required the fullsix Ridge25 primary value for a late W4+W6 current-champion comparison. "
            "The scorer failed closed at baseline validation before fitting offsets or computing candidate metrics. "
            "Revision 2 pins the pre-existing row-matched sparse-LGB25 blend metrics and preserves the original freeze unchanged."
        ),
    })


def _verify_frozen(context: dict[str, Any]) -> None:
    frozen = _read_json(FREEZE_PATH)
    if (frozen.get("status") != "FROZEN_BEFORE_R19_CALIBRATION_AND_SCORING"
            or frozen.get("signature") != context["signature"]
            or json_sha256(frozen.get("identity", {})) != context["signature"]):
        raise BatchIntegrityError("R19 revision 2 freeze is missing or does not match current inputs")
    normalized = json.loads(json.dumps(context["recipe"], sort_keys=True, allow_nan=False))
    if _read_json(RECIPE_PATH) != normalized:
        raise BatchIntegrityError("R19 revision 2 recipe differs from its frozen identity")


def _freeze(context: dict[str, Any]) -> dict[str, Any]:
    _write_once(RECIPE_PATH, context["recipe"])
    _write_once(FREEZE_PATH, {
        "status": "FROZEN_BEFORE_R19_CALIBRATION_AND_SCORING",
        "identity": context["identity"], "signature": context["signature"],
        "target_read": False, "prediction_arrays_read": False,
        "candidate_metric_computed": False, "test_read": False,
    })
    return {"card_id": "R19", "revision": 2, "status": "FROZEN_READY",
            "freeze_signature": context["signature"],
            "row_matched_control_metrics": EXPECTED_METRICS,
            "candidate_metric_computed": False, "test_read": False}


def _run(context: dict[str, Any]) -> dict[str, Any]:
    _verify_frozen(context)
    old_verify = r19_v1._verify_frozen
    old_expected = r19_v1.EXPECTED_BASELINE_MAE
    old_results = r19_v1.RESULTS_PATH
    try:
        r19_v1._verify_frozen = _verify_frozen
        r19_v1.EXPECTED_BASELINE_MAE = EXPECTED_METRICS
        r19_v1.RESULTS_PATH = RESULTS_PATH
        report = r19_v1._score(context)
    finally:
        r19_v1._verify_frozen = old_verify
        r19_v1.EXPECTED_BASELINE_MAE = old_expected
        r19_v1.RESULTS_PATH = old_results
    return {"card_id": "R19", "revision": 2, "status": report["status"],
            "decision": report["decision"],
            "results_path": str(RESULTS_PATH.relative_to(ROOT)),
            "fit_count": 0, "test_data_used": False, "kaggle_access": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args()
    _record_superseded_attempt()
    context = _prepare()
    if args.freeze_only:
        result = _freeze(context)
    elif args.run_frozen_pilot:
        result = _run(context)
    else:
        result = {
            "card_id": "R19", "revision": 2, "status": "PREFLIGHT_PASS",
            "freeze_signature": context["signature"],
            "row_matched_control_metrics": EXPECTED_METRICS,
            "historical_report_support": context["recipe"]["baseline_metric_provenance"],
            "feature_only_support": context["support"],
            "target_read": False, "prediction_arrays_read": False,
            "candidate_metric_computed": False, "test_read": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
