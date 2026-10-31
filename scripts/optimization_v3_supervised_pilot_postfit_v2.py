"""No-refit adapter for the frozen supervised pilot's post-fit aggregation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts import optimization_v3_supervised_pilot as pilot
from src.models.round1_evaluation import code_digest, digest_json

CONFIG = pilot.CONFIG
OUT = pilot.OUT
CACHE = pilot.CACHE
AMENDMENT = CONFIG / "postfit_amendment_v2.json"
ADAPTER = Path(__file__).relative_to(ROOT).as_posix()
TEST = "tests/test_optimization_v3_supervised_pilot_postfit_v2.py"
FROZEN_SOURCES = [
    "scripts/optimization_v3_supervised_pilot.py",
    "scripts/optimization_v3_verify_supervised_pilot.py",
    "tests/test_optimization_v3_supervised_pilot.py",
    "src/models/optimization_v3_supervised.py",
    "src/models/optimization_v3_token_stats.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _expected_files() -> list[Path]:
    return [CACHE / f"{task}_fit{end}.{ext}"
            for task in pilot.TASKS for end in pilot.ENDPOINTS for ext in ["npz", "json"]]


def _assert_output_boundary() -> None:
    future_reports = [OUT / "pilot.json", OUT / "FULLPILOT.md", OUT / "verification.json",
                      OUT / "postfit_adapter_run_v2.json"]
    present = [str(path) for path in future_reports if path.exists()]
    if present:
        raise FileExistsError(f"Post-fit report outputs already exist: {present}")
    missing = [str(path) for path in _expected_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"All four frozen fit cache pairs are required: {missing}")


def make_amendment_identity() -> dict:
    frozen = pilot.check_freeze()
    _assert_output_boundary()
    input_hashes = frozen["identity"]["input_sha256_exact_bytes"]
    for relative, expected in input_hashes.items():
        actual = sha(ROOT / Path(relative))
        if actual != expected:
            raise AssertionError(f"Frozen input changed before post-fit adapter: {relative}")
    fit_pairs = {}
    for task in pilot.TASKS:
        for end in pilot.ENDPOINTS:
            array_path = CACHE / f"{task}_fit{end}.npz"
            sidecar_path = array_path.with_suffix(".json")
            meta = read_json(sidecar_path)
            if meta.get("freeze_signature") != frozen["signature"]:
                raise AssertionError(f"{task}@{end} cache has a different pilot freeze")
            if meta.get("npz_sha256") != sha(array_path):
                raise AssertionError(f"{task}@{end} cache sidecar hash mismatch")
            if meta.get("task") != task or meta.get("cutoff") != end:
                raise AssertionError(f"{task}@{end} cache metadata identity mismatch")
            if meta.get("test_read") is not False or meta.get("outer_labels_used_for_fit") is not False:
                raise AssertionError(f"{task}@{end} cache scope metadata is invalid")
            fit_pairs[f"{task}_{end}"] = {
                "npz_sha256_exact_bytes": sha(array_path),
                "sidecar_sha256_exact_bytes": sha(sidecar_path),
                "fit_rows": int(meta["fit_rows"]),
                "fit_data_sha256": meta["fit_data_sha256"],
                "model_state_sha256": meta["model_state_sha256"],
            }
    if len(fit_pairs) != 4:
        raise AssertionError("Post-fit amendment must bind exactly the four frozen fits")
    source_hashes = {path: code_digest(ROOT / path) for path in FROZEN_SOURCES}
    source_hashes[ADAPTER] = code_digest(Path(__file__))
    source_hashes[TEST] = code_digest(ROOT / TEST)
    return {
        "pilot_freeze_signature": frozen["signature"],
        "pilot_freeze_receipt_sha256": sha(CONFIG / "freeze.json"),
        "pilot_protocol_sha256": sha(CONFIG / "protocol.json"),
        "frozen_source_code_sha256_lf_utf8": source_hashes,
        "frozen_input_sha256_exact_bytes": input_hashes,
        "four_fit_output_sha256_exact_bytes": fit_pairs,
        "new_official_fits_in_adapter": 0,
        "test_read": False,
        "kaggle_access": False,
        "amendment": {
            "issue": "Frozen runner completed all four fits and wrote cache pairs, then post-fit view aggregation requested record['policies'] although fit records expose route arrays at top level.",
            "repair": "Analysis-only adapter reconstructs the expected nested policy mapping from the exact four preserved fit caches; no model source, predictions, fit configuration, controls, or labels are changed.",
            "source_freeze_preserved": True,
            "refit_authorized_or_performed": False,
        },
    }


def freeze_amendment() -> dict:
    identity = make_amendment_identity()
    signature = digest_json(identity)
    receipt = {
        "status": "FROZEN_POSTFIT_ADAPTER_BEFORE_METRIC_AGGREGATION",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "signature": signature,
    }
    if AMENDMENT.exists():
        old = read_json(AMENDMENT)
        if old.get("identity") != identity or old.get("signature") != signature:
            raise AssertionError("Existing post-fit amendment differs; preserve and use a new version")
        return old
    write_new(AMENDMENT, receipt)
    return receipt


def check_amendment() -> dict:
    receipt = read_json(AMENDMENT)
    current = make_amendment_identity()
    if receipt.get("identity") != current or receipt.get("signature") != digest_json(current):
        raise AssertionError("Post-fit adapter source/input/output identity changed")
    if receipt.get("status") != "FROZEN_POSTFIT_ADAPTER_BEFORE_METRIC_AGGREGATION":
        raise AssertionError("Unexpected post-fit amendment status")
    return receipt


def adapt_classification_fit(end: int, arrays: dict, state: dict,
                             seconds: float, cache_sha256: str) -> dict:
    required = {"rows", "cycles", "raw_probability", "initial_control", "latest_control",
                "direct", "blend25", "c2b_present", "cb_missing", "missing"}
    if set(arrays) != required:
        raise ValueError(f"Class fit cache schema mismatch: {sorted(set(arrays) ^ required)}")
    routes = {
        "initial_control": np.asarray(arrays["initial_control"], dtype=float),
        "latest_sparse25": np.asarray(arrays["latest_control"], dtype=float),
        "direct": np.asarray(arrays["direct"], dtype=float),
        "blend25": np.asarray(arrays["blend25"], dtype=float),
    }
    return {
        "end": int(end),
        "rows": np.asarray(arrays["rows"], dtype=np.int64),
        "cycles": np.asarray(arrays["cycles"], dtype=int),
        "raw_probability": np.asarray(arrays["raw_probability"], dtype=float),
        "initial_control": routes["initial_control"],
        "latest_sparse25": routes["latest_sparse25"],
        "direct": routes["direct"],
        "blend25": routes["blend25"],
        "policies": routes,
        "missing": np.asarray(arrays["missing"], dtype=bool),
        "state": state,
        "seconds": float(seconds),
        "cache_sha256": cache_sha256,
        "source": "preserved_frozen_fit_cache",
    }


def adapt_regression_fit(end: int, arrays: dict, state: dict,
                         seconds: float, cache_sha256: str) -> dict:
    required = {"rows", "cycles", "raw", "rank50", "blend25", "champion"}
    if set(arrays) != required:
        raise ValueError(f"Regression fit cache schema mismatch: {sorted(set(arrays) ^ required)}")
    return {
        "end": int(end),
        "rows": np.asarray(arrays["rows"], dtype=np.int64),
        "cycles": np.asarray(arrays["cycles"], dtype=int),
        "raw": np.asarray(arrays["raw"], dtype=float),
        "rank50": np.asarray(arrays["rank50"], dtype=float),
        "blend25": np.asarray(arrays["blend25"], dtype=float),
        "champion": np.asarray(arrays["champion"], dtype=float),
        "state": state,
        "seconds": float(seconds),
        "cache_sha256": cache_sha256,
        "source": "preserved_frozen_fit_cache",
    }


def _load_records():
    class_frame, class_y, _, reg_frame, _ = pilot.load_official_data()
    reg_y = pd.to_numeric(reg_frame.threat_rank, errors="raise").to_numpy(dtype=float)
    reg_primary, reg_h12, group_audits = pilot.get_regression_views(reg_frame)
    class_records, reg_records = {}, {}
    cache_sha = {}
    for task, destination in [("classification", class_records), ("regression", reg_records)]:
        for end in pilot.ENDPOINTS:
            path = CACHE / f"{task}_fit{end}.npz"
            meta = read_json(path.with_suffix(".json"))
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            if task == "classification":
                # Drop output-only arrays that are not part of the policy-record interface.
                view_arrays = {name: arrays[name] for name in [
                    "rows", "cycles", "raw_probability", "initial_control", "latest_control",
                    "direct", "blend25", "c2b_present", "cb_missing", "missing",
                ]}
                record = adapt_classification_fit(end, view_arrays, meta["full_model_state"],
                                                  meta["seconds"], sha(path))
            else:
                view_arrays = {name: arrays[name] for name in [
                    "rows", "cycles", "raw", "rank50", "blend25", "champion",
                ]}
                record = adapt_regression_fit(end, view_arrays, meta["full_model_state"],
                                              meta["seconds"], sha(path))
            destination[end] = record
            cache_sha[f"{task}_{end}"] = sha(path)
    return (class_frame, class_y, reg_frame, reg_y, class_records, reg_records,
            reg_primary, reg_h12, group_audits, cache_sha)


def _gates(class_views: dict, reg_views: dict) -> tuple[dict, dict]:
    class_window_deltas = [
        class_views[name]["comparisons"]["blend25_vs_latest_sparse25"]["macro_f1_delta"]
        for name in ["W1", "W4", "H12_E", "H12_L"]
    ]
    class_primary = class_views["primary_pilot"]["comparisons"]["blend25_vs_latest_sparse25"]["macro_f1_delta"]
    class_h12 = class_views["H12"]["comparisons"]["blend25_vs_latest_sparse25"]["macro_f1_delta"]
    class_gate = {
        "primary_or_h12_improves": bool(class_primary > 0 or class_h12 > 0),
        "no_view_harm_over_0_01": bool(min(class_window_deltas) >= -0.01),
        "primary_gain_vs_latest_sparse25": float(class_primary),
        "h12_gain_vs_latest_sparse25": float(class_h12),
        "minimum_view_delta_vs_latest_sparse25": float(min(class_window_deltas)),
        "views_nonnegative_vs_latest_sparse25": int(sum(delta >= 0 for delta in class_window_deltas)),
    }
    reg_harms = [-reg_views[name]["delta_vs_champion"]["blend25"]["mae_gain_champion_minus_candidate"]
                 for name in ["W1", "W4", "H12_E", "H12_L"]]
    reg_primary = reg_views["primary_pilot"]["delta_vs_champion"]["blend25"]["mae_gain_champion_minus_candidate"]
    reg_h12_gain = reg_views["H12"]["delta_vs_champion"]["blend25"]["mae_gain_champion_minus_candidate"]
    reg_gate = {
        "primary_or_h12_improves": bool(reg_primary > 0 or reg_h12_gain > 0),
        "max_view_harm_at_most_0_50": bool(max(reg_harms) <= 0.50),
        "primary_gain_vs_three_component_champion": float(reg_primary),
        "h12_gain_vs_three_component_champion": float(reg_h12_gain),
        "maximum_W1_W4_H12_view_harm": float(max(reg_harms)),
        "view_harms": {name: float(value) for name, value in zip(["W1", "W4", "H12_E", "H12_L"], reg_harms)},
    }
    return class_gate, reg_gate


def aggregate() -> dict:
    amendment = check_amendment()
    frozen = pilot.check_freeze()
    (class_frame, class_y, reg_frame, reg_y, class_records, reg_records,
     reg_primary, reg_h12, group_audits, cache_hashes) = _load_records()
    class_views = pilot.build_class_views(class_frame, class_y, class_records)
    reg_views = pilot.build_reg_views(reg_frame, reg_y, reg_records, reg_primary, reg_h12)
    class_gate, reg_gate = _gates(class_views, reg_views)
    model_audit = {}
    for task, records in [("classification", class_records), ("regression", reg_records)]:
        for end, record in records.items():
            stats = record["state"]["ordered_tokens"]
            model_audit[f"{task}_{end}"] = {
                "fit_rows": int(record["state"]["fit_rows"]),
                "fit_population": "present_stream_only" if task == "classification" else "all_train_rows",
                "fit_data_sha256": record["state"]["fit_data_sha256"],
                "fit_target_sum": float(stats["fit_target_sum"]),
                "dictionary_size": int(stats["dictionary_size"]),
                "dictionary_sha256": stats["dictionary_sha256"],
                "training_features_sha256": stats["training_features_sha256"],
                "lineage": stats["lineage"],
                "validation_rows": int(len(record["rows"])),
                "fit_seconds": float(record["seconds"]),
                "cache_sha256": record["cache_sha256"],
            }
    result = {
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "freeze_signature": frozen["signature"],
        "source_head_contextual": frozen["source_head_contextual"],
        "classification_train_sha256": frozen["identity"]["classification_official_train_sha256"],
        "regression_train_sha256": frozen["identity"]["regression_official_train_sha256"],
        "official_train_rows_per_task": 50608,
        "new_official_fits": 4,
        "fit_plan": [f"{task}@{end}" for end in pilot.ENDPOINTS for task in pilot.TASKS],
        "test_read": False,
        "kaggle_access": False,
        "outer_labels_used_for_fit": False,
        "classification_threshold": pilot.THRESHOLD,
        "classification_fixed_new_blend_weight_present": pilot.BLEND_NEW_WEIGHT,
        "regression_rank_weight": 0.50,
        "regression_fixed_rank50_blend_weight": 0.25,
        "classification": {"views": class_views, "advance_gate_vs_latest_sparse25": class_gate},
        "regression": {"views": reg_views, "advance_gate_vs_current_champion": reg_gate,
                       "fit_defined_control_group_audits": group_audits},
        "fit_audits": model_audit,
        "postfit_adapter": {
            "version": "postfit_v2",
            "amendment_signature": amendment["signature"],
            "official_refits": 0,
            "frozen_runner_source_edited": False,
        },
        "interpretation_limits": [
            "Classification comparisons jointly change representation and fit population; metric changes cannot be attributed to token statistics alone.",
            "Training features encode each row using labels from strictly earlier cycles; validation rows use all labels available through their fit cutoff.",
            "These are observed past token-target associations, not claims about anonymous token semantics or causation.",
            "The temporal views are reused known Train outcomes and not an untouched holdout.",
            "Regression target-tail groups are descriptive only; the external Test and Kaggle were not accessed.",
        ],
    }
    pilot.write_new(OUT / "pilot.json", result)
    pilot.write_markdown(result)
    run_receipt = {
        "status": "AGGREGATED_FROM_PRESERVED_FOUR_FIT_CACHES",
        "amendment_signature": amendment["signature"],
        "pilot_json_sha256_exact_bytes": sha(OUT / "pilot.json"),
        "fullpilot_markdown_sha256_exact_bytes": sha(OUT / "FULLPILOT.md"),
        "fit_cache_hashes_rechecked": cache_hashes,
        "new_official_fits": 0,
        "test_read": False,
        "kaggle_access": False,
    }
    write_new(OUT / "postfit_adapter_run_v2.json", run_receipt)
    print(json.dumps({"status": result["status"], "amendment_signature": amendment["signature"],
                      "classification_gate": class_gate, "regression_gate": reg_gate,
                      "official_refits": 0}, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-amendment", action="store_true")
    group.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.freeze_amendment:
        receipt = freeze_amendment()
        print(receipt["signature"])
    else:
        aggregate()
