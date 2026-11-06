"""Audit all-view supported-group harm for the frozen missing-offset result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
LANE = "REG_MISSING_MEDIAN_OFFSET_V1"
REPORT_ROOT = ROOT / "reports" / "optimization_v3" / LANE
SCORE_PATH = REPORT_ROOT / "SCORE_R1.json"
METRIC_QA_PATH = REPORT_ROOT / "METRIC_QA_R2.json"
PRELABEL_PATH = REPORT_ROOT / "PRELABEL_QA_R1.json"
SOURCE_PATH = REPORT_ROOT / "SOURCE_FREEZE_R1.json"
RECIPE_PATH = REPORT_ROOT / "RECIPE_R1.json"
OUTPUT_PATH = REPORT_ROOT / "GROUP_GUARD_SCOPE_QA_R3.json"
METRIC_VERIFIER_PATH = ROOT / "scripts" / "verify_reg_missing_median_offset_metric_r2.py"
MIN_SUPPORT = 100
GROUP_LIMIT = 0.25


class GuardAuditError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GuardAuditError(f"Expected JSON object: {path}")
    return value


def _verify_signature(value: Mapping[str, Any], key: str, label: str) -> None:
    body = {name: item for name, item in value.items() if name != key}
    if value.get(key) != _sha_bytes(_canonical(body)):
        raise GuardAuditError(f"Signature mismatch: {label}")


def _same_tree(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same_tree(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_tree(a, b, tolerance) for a, b in zip(left, right))
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def _extract_view(name: str, groups: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(groups, Mapping) or not groups:
        raise GuardAuditError(f"Missing fixed groups for view {name}")
    supported = []
    unknown = []
    violations = []
    group_rows = {}
    for group_name in sorted(groups):
        item = groups[group_name]
        if not isinstance(item, Mapping):
            raise GuardAuditError(f"Malformed group {name}.{group_name}")
        status = item.get("status")
        rows = item.get("rows")
        harm = item.get("harm_candidate_minus_champion")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
            raise GuardAuditError(f"Invalid group support {name}.{group_name}")
        if status == "MEASURED":
            if rows < MIN_SUPPORT or not isinstance(harm, (int, float)) or isinstance(harm, bool):
                raise GuardAuditError(f"Measured group below support or without harm: {name}.{group_name}")
            if not float("-inf") < float(harm) < float("inf"):
                raise GuardAuditError(f"Nonfinite harm: {name}.{group_name}")
            record = {"rows": rows, "harm_candidate_minus_champion": float(harm)}
            supported.append({"group": group_name, **record})
            if float(harm) > GROUP_LIMIT:
                violations.append({"group": group_name, **record,
                                   "excess_over_limit": float(harm) - GROUP_LIMIT})
        elif status == "UNKNOWN":
            if rows >= MIN_SUPPORT or harm is not None:
                raise GuardAuditError(f"UNKNOWN support incorrectly suppressed or filled: {name}.{group_name}")
            unknown.append({"group": group_name, "rows": rows})
        else:
            raise GuardAuditError(f"Unexpected group status {name}.{group_name}: {status}")
        group_rows[group_name] = {"status": status, "rows": rows, "harm_candidate_minus_champion": harm}
    maximum = max(supported, key=lambda row: row["harm_candidate_minus_champion"], default=None)
    return {
        "view": name,
        "status": "PASS" if not violations else "FAIL",
        "minimum_support": MIN_SUPPORT,
        "groups_total": len(groups),
        "supported_group_count": len(supported),
        "unknown_group_count": len(unknown),
        "maximum_supported_group_harm": maximum,
        "limit": GROUP_LIMIT,
        "violations": violations,
        "unknown_groups": unknown,
        "fixed_groups": group_rows,
    }


def build() -> dict[str, Any]:
    score = _read_object(SCORE_PATH)
    qa = _read_object(METRIC_QA_PATH)
    prelabel = _read_object(PRELABEL_PATH)
    source = _read_object(SOURCE_PATH)
    recipe = _read_object(RECIPE_PATH)
    _verify_signature(score, "score_signature", "score R1")
    _verify_signature(qa, "qa_signature", "metric QA R2")
    _verify_signature(prelabel, "qa_signature", "prelabel QA R1")
    _verify_signature(source, "source_signature", "source freeze R1")
    _verify_signature(recipe, "recipe_signature", "recipe R1")

    bindings = {
        "source_signature": source.get("source_signature"),
        "score_signature": score.get("score_signature"),
        "score_sha256": _sha_file(SCORE_PATH),
        "metric_qa_signature": qa.get("qa_signature"),
        "metric_qa_sha256": _sha_file(METRIC_QA_PATH),
        "prelabel_qa_sha256": _sha_file(PRELABEL_PATH),
        "recipe_sha256": _sha_file(RECIPE_PATH),
        "source_freeze_sha256": _sha_file(SOURCE_PATH),
        "metric_verifier_r2_sha256": _sha_file(METRIC_VERIFIER_PATH),
        "guard_verifier_r3_sha256": _sha_file(Path(__file__)),
    }
    if (source.get("status") != "SOURCE_FROZEN_PRELABEL" or score.get("status") != "SCORED"
            or qa.get("status") != "PASS" or prelabel.get("status") != "PASS"
            or qa.get("source_signature") != source.get("source_signature")
            or score.get("source_signature") != source.get("source_signature")
            or qa.get("score_signature") != score.get("score_signature")
            or qa.get("score_sha256") != bindings["score_sha256"]
            or qa.get("prelabel_qa_sha256") != bindings["prelabel_qa_sha256"]
            or prelabel.get("source_signature") != source.get("source_signature")
            or qa.get("metric_verifier_sha256") != bindings["metric_verifier_r2_sha256"]
            or score.get("maximum_supported_group_harm") is None):
        raise GuardAuditError("Frozen score/QA/source bindings failed")

    score_views = {
        "D1": score["folds"]["D1"]["fixed_groups"],
        "D2": score["folds"]["D2"]["fixed_groups"],
        "pooled": score["pooled_fixed_groups"],
    }
    qa_views = {
        "D1": qa["D1"]["fixed_groups"],
        "D2": qa["D2"]["fixed_groups"],
        "pooled": qa["pooled_fixed_groups"],
    }
    expected_names = set(score_views["pooled"])
    if len(expected_names) != 14 or any(set(groups) != expected_names for groups in score_views.values()):
        raise GuardAuditError("Fixed-group definition drifted across score views")
    if not _same_tree(score_views, qa_views):
        raise GuardAuditError("R2 independent metric QA differs from the stored score group trees")

    views = {name: _extract_view(name, score_views[name]) for name in ("D1", "D2", "pooled")}
    max_records = [value["maximum_supported_group_harm"] for value in views.values()
                   if value["maximum_supported_group_harm"] is not None]
    all_view_maximum = max(max_records, key=lambda row: row["harm_candidate_minus_champion"], default=None)
    pooled_only_maximum = views["pooled"]["maximum_supported_group_harm"]
    if (pooled_only_maximum is None
            or score.get("maximum_supported_group_harm") != pooled_only_maximum["harm_candidate_minus_champion"]
            or qa.get("maximum_supported_group_harm") != pooled_only_maximum["harm_candidate_minus_champion"]):
        raise GuardAuditError("Prior top-level group-harm fields are not the pooled-only maximum")

    gate_pass = all(view["status"] == "PASS" for view in views.values())
    body = {
        "schema_version": 1,
        "experiment_id": LANE,
        "addendum_id": "GROUP_GUARD_SCOPE_QA_R3",
        "status": "PASS",
        "qualification_status": "FAIL",
        "gate_definition": "maximum harm_candidate_minus_champion across every supported fixed group (n>=100) in EACH of D1, D2, and pooled views must be <=0.25",
        "minimum_group_support": MIN_SUPPORT,
        "group_harm_limit": GROUP_LIMIT,
        "source_inventory": {
            "inputs": {
                "score": {"path": SCORE_PATH.relative_to(ROOT).as_posix(), "sha256": bindings["score_sha256"],
                          "signature": score["score_signature"]},
                "independent_metric_qa": {"path": METRIC_QA_PATH.relative_to(ROOT).as_posix(),
                                          "sha256": bindings["metric_qa_sha256"], "signature": qa["qa_signature"]},
                "prelabel_qa": {"path": PRELABEL_PATH.relative_to(ROOT).as_posix(),
                                "sha256": bindings["prelabel_qa_sha256"], "signature": prelabel["qa_signature"]},
                "source_freeze": {"path": SOURCE_PATH.relative_to(ROOT).as_posix(),
                                  "sha256": bindings["source_freeze_sha256"], "signature": source["source_signature"]},
                "recipe": {"path": RECIPE_PATH.relative_to(ROOT).as_posix(),
                           "sha256": bindings["recipe_sha256"], "signature": recipe["recipe_signature"]},
                "metric_verifier_r2": {"path": METRIC_VERIFIER_PATH.relative_to(ROOT).as_posix(),
                                       "sha256": bindings["metric_verifier_r2_sha256"]},
                "guard_verifier_r3": {"path": Path(__file__).relative_to(ROOT).as_posix(),
                                       "sha256": bindings["guard_verifier_r3_sha256"]},
            },
            "source_labels_read_by_addendum": False,
            "prediction_arrays_read_by_addendum": False,
            "evidence_used": "stored signed D1, D2, and pooled fixed-group metrics only",
        },
        "prior_field_scope": {
            "score_R1_maximum_supported_group_harm": score["maximum_supported_group_harm"],
            "metric_QA_R2_maximum_supported_group_harm": qa["maximum_supported_group_harm"],
            "interpretation": "Both existing top-level fields are pooled-view-only maxima; they are not the maximum over D1, D2, and pooled views.",
        },
        "views": views,
        "maximum_supported_group_harm_all_views": all_view_maximum,
        "all_view_group_guard_pass": gate_pass,
        "pooled_only_group_guard_pass": pooled_only_maximum["harm_candidate_minus_champion"] <= GROUP_LIMIT,
        "original_scientific_qualification": score["scientific_gate_without_independent_metric_qa"] is True,
        "original_exploratory_qualification": score["exploratory_gate_without_independent_metric_qa"] is True,
        "scientific_qualification": False,
        "exploratory_eligibility": False,
        "verdict": "NOT_ELIGIBLE; D1 missing group harm exceeds the all-view 0.25 guard. Original score and R2 QA remain unchanged.",
        "mutations": {
            "score_R1_rewritten": False,
            "metric_QA_R2_rewritten": False,
            "new_fit_or_application": False,
            "new_score_pass": False,
            "target_labels_read": False,
        },
    }
    result = dict(body)
    result["addendum_signature"] = _sha_bytes(_canonical(result))
    return result


def verify_saved() -> dict[str, Any]:
    report = _read_object(OUTPUT_PATH)
    _verify_signature(report, "addendum_signature", "guard scope R3")
    expected = build()
    if not _same_tree(report, expected):
        raise GuardAuditError("Saved guard scope R3 differs from recomputed stored-metric audit")
    return {"status": report["status"], "qualification_status": report["qualification_status"],
            "all_view_group_guard_pass": report["all_view_group_guard_pass"],
            "maximum_supported_group_harm_all_views": report["maximum_supported_group_harm_all_views"],
            "addendum_signature": report["addendum_signature"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        if args.build:
            if OUTPUT_PATH.exists():
                raise GuardAuditError("Refusing to overwrite immutable R3 addendum")
            report = build()
            OUTPUT_PATH.write_bytes(_canonical(report) + b"\n")
            result = {"status": report["status"], "qualification_status": report["qualification_status"],
                      "all_view_group_guard_pass": report["all_view_group_guard_pass"],
                      "maximum_supported_group_harm_all_views": report["maximum_supported_group_harm_all_views"],
                      "addendum_signature": report["addendum_signature"]}
        else:
            result = verify_saved()
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
