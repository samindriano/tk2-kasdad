"""Independent zero-fit verifier for the frozen R06/R22 midpoint check.

This verifier reads the frozen protocol, official Regression Train, existing
R06/R22 NPZ bundles and the runner's result.json/groups.csv. It does not import
the runner or any project metric/composition helper, and never opens Test.
Run only after the frozen runner has completed and published its artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT


PROTOCOL_PATH = PROJECT_ROOT / "experiments/optimization_v3/measured_benchmark_v1/reg_midpoint_protocol.json"
RESULT_DIR = PROJECT_ROOT / "reports/optimization_v3/measured_benchmark_v1/REG_MIDPOINT"
RESULT_PATH = RESULT_DIR / "result.json"
GROUPS_PATH = RESULT_DIR / "groups.csv"
OUTPUT_PATH = RESULT_DIR / "verification.json"
EXPECTED_PROTOCOL_SHA256 = "dd1cf7febdbf8350481e41dbba5fb2cf8679673e5512eb61f8f0c59f43ecc6c2"
METRIC_TOL = 1e-9
PREDICTION_TOL = 1e-10


class VerificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Independent stable-sort average ranks, one-based as in rankdata."""
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), "rank input must be finite 1D")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        # Sorted positions [start, end) have one-based ranks start+1 through end.
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def rank50_by_full_cycle(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles)
    require(raw.ndim == cycles.ndim == 1 and raw.size == cycles.size, "rank input shapes differ")
    require(np.isfinite(raw).all(), "raw component contains non-finite values")
    mapped = np.empty(raw.size, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        rank = average_ranks(raw[mask])
        percentile = 100.0 * rank / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    return np.clip(mapped, 0.0, 100.0)


def mae(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    require(y.ndim == prediction.ndim == 1 and y.size == prediction.size, "MAE input shapes differ")
    require(y.size > 0 and np.isfinite(y).all() and np.isfinite(prediction).all(), "MAE inputs empty/non-finite")
    return float(np.abs(y - prediction).sum(dtype=np.float64) / y.size)


def read_protocol() -> tuple[dict[str, Any], str]:
    require(PROTOCOL_PATH.is_file(), f"frozen protocol missing: {PROTOCOL_PATH}")
    protocol_sha = sha256_file(PROTOCOL_PATH)
    require(protocol_sha == EXPECTED_PROTOCOL_SHA256, "protocol SHA does not match frozen parent receipt")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    require(protocol.get("id") == "MEASURED_REG_MIDPOINT_V1", "unexpected protocol id")
    require(protocol.get("status") == "FROZEN_BEFORE_NEW_MIDPOINT_OUTCOMES", "protocol not in frozen pre-outcome state")
    require(protocol.get("recipe") == ".5 * R06_ORIGINAL + .5 * R22_ORIGINAL; all rows; no routing change or reranking of mapped predictions", "midpoint recipe differs from reviewed protocol")
    require(protocol.get("official_new_fits") == 0, "protocol must remain zero-fit")
    return protocol, protocol_sha


def verify_input_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    expected = protocol.get("input_sha256")
    require(isinstance(expected, dict) and len(expected) == 25, "expected 25 frozen input hashes")
    observed: dict[str, str] = {}
    for relative, expected_sha in expected.items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"frozen input missing: {relative}")
        actual = sha256_file(path)
        require(actual == expected_sha, f"frozen input hash mismatch: {relative}")
        observed[relative] = actual
    return observed


def load_train(protocol: dict[str, Any]) -> pd.DataFrame:
    train_path = PROJECT_ROOT / "data/raw/regression/Train.csv"
    expected_sha = protocol["input_sha256"]["data/raw/regression/Train.csv"]
    require(sha256_file(train_path) == expected_sha, "official Regression Train hash mismatch")
    frame = pd.read_csv(
        train_path,
        usecols=["record_id", "cycle_code", "threat_rank", "signal_stream"],
        dtype={"record_id": str},
    )
    require(frame["record_id"].is_unique, "official Train record_id is not unique")
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy()
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    require(np.isfinite(target).all(), "official Train target is non-finite")
    require(np.isfinite(cycle).all(), "official Train cycle_code is non-finite")
    frame["cycle_code"] = cycle.astype(np.int64)
    frame["threat_rank"] = target
    return frame


def load_bundle(card: str, cutoff: int, relative_npz: str, train: pd.DataFrame) -> dict[str, Any]:
    npz_path = PROJECT_ROOT / relative_npz
    sidecar_path = npz_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    require(sidecar.get("schema_version") is not None, f"{card}/{cutoff}: missing sidecar schema")
    require(sidecar.get("npz_sha256") == sha256_file(npz_path), f"{card}/{cutoff}: NPZ sidecar hash mismatch")
    require(
        sidecar.get("identity_sha256") == sha256_json(sidecar.get("identity")),
        f"{card}/{cutoff}: identity digest mismatch",
    )
    integrity_payload = {key: value for key, value in sidecar.items() if key != "integrity_sha256"}
    require(
        sidecar.get("integrity_sha256") == sha256_json(integrity_payload),
        f"{card}/{cutoff}: sidecar integrity digest mismatch",
    )
    identity = sidecar.get("identity", {})
    metadata = sidecar.get("metadata", {})
    require(identity.get("card_id") == card and identity.get("cutoff") == cutoff, f"{card}/{cutoff}: identity mismatch")
    require(metadata.get("card_id", card) == card and metadata.get("cutoff", cutoff) == cutoff, f"{card}/{cutoff}: metadata mismatch")

    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for key in ("score_rows", "score_record_ids", "raw_candidate"):
        require(key in arrays, f"{card}/{cutoff}: missing {key}")
    candidate_key = f"{card}_ORIGINAL" if f"{card}_ORIGINAL" in arrays else "prediction"
    baseline_key = "champion_prediction" if "champion_prediction" in arrays else "baseline_prediction"
    require(candidate_key in arrays and baseline_key in arrays, f"{card}/{cutoff}: missing policy/baseline output")

    rows = np.asarray(arrays["score_rows"])
    ids = np.asarray(arrays["score_record_ids"]).astype(str)
    require(rows.ndim == ids.ndim == 1 and rows.size == ids.size, f"{card}/{cutoff}: row/ID shape mismatch")
    require(np.issubdtype(rows.dtype, np.integer) and np.all(rows >= 0), f"{card}/{cutoff}: invalid row indices")
    require(rows.size == np.unique(rows).size and ids.size == np.unique(ids).size, f"{card}/{cutoff}: duplicate row or ID")
    require(int(rows.max(initial=-1)) < len(train), f"{card}/{cutoff}: row index outside Train")
    train_ids = train["record_id"].to_numpy(dtype=str)[rows]
    require(np.array_equal(train_ids, ids), f"{card}/{cutoff}: source row/record_id alignment mismatch")
    cycles = train["cycle_code"].to_numpy(dtype=np.int64)[rows]
    target = train["threat_rank"].to_numpy(dtype=np.float64)[rows]
    require(np.isfinite(target).all(), f"{card}/{cutoff}: non-finite target")

    policy = np.asarray(arrays[candidate_key], dtype=np.float64)
    legacy = np.asarray(arrays[baseline_key], dtype=np.float64)
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    require(policy.shape == legacy.shape == raw.shape == rows.shape, f"{card}/{cutoff}: prediction shape mismatch")
    require(np.isfinite(policy).all() and np.isfinite(legacy).all() and np.isfinite(raw).all(), f"{card}/{cutoff}: non-finite prediction")
    require(np.all((policy >= 0) & (policy <= 100)), f"{card}/{cutoff}: policy output outside [0,100]")
    require(np.all((legacy >= 0) & (legacy <= 100)), f"{card}/{cutoff}: legacy output outside [0,100]")

    # Ensure every scored cycle is complete before validating the within-cycle mapping.
    for cycle in np.unique(cycles):
        got_rows = np.sort(rows[cycles == cycle])
        expected_rows = np.flatnonzero(train["cycle_code"].to_numpy(dtype=np.int64) == cycle)
        require(np.array_equal(got_rows, expected_rows), f"{card}/{cutoff}: scored cycle {cycle} is incomplete")
    reconstructed = np.clip(0.75 * legacy + 0.25 * rank50_by_full_cycle(raw, cycles), 0.0, 100.0)
    require(np.allclose(policy, reconstructed, atol=PREDICTION_TOL, rtol=0.0), f"{card}/{cutoff}: policy/rank50 reconstruction mismatch")

    order = np.argsort(rows, kind="mergesort")
    return {
        "card": card,
        "cutoff": cutoff,
        "npz_sha256": sha256_file(npz_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "rows": rows[order],
        "ids": ids[order],
        "cycles": cycles[order],
        "target": target[order],
        "legacy": legacy[order],
        "policy": policy[order],
    }


def build_views(protocol: dict[str, Any], bundles: dict[tuple[str, int], dict[str, Any]], train: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    streams = train["signal_stream"]
    stream_missing_all = (streams.isna() | streams.fillna("").astype(str).str.strip().eq("")).to_numpy(dtype=bool)

    def part(origin: int, start: int, end: int) -> dict[str, np.ndarray]:
        b22 = bundles[("R22", origin)]
        b06 = bundles[("R06", origin)]
        require(np.array_equal(b22["rows"], b06["rows"]), f"origin {origin}: component row indices differ")
        require(np.array_equal(b22["ids"], b06["ids"]), f"origin {origin}: component IDs differ")
        require(np.array_equal(b22["cycles"], b06["cycles"]), f"origin {origin}: component cycle arrays differ")
        require(np.allclose(b22["legacy"], b06["legacy"], atol=PREDICTION_TOL, rtol=0.0), f"origin {origin}: legacy baseline differs")
        mask = (b22["cycles"] >= start) & (b22["cycles"] <= end)
        rows = b22["rows"][mask]
        require(rows.size > 0, f"origin {origin}: empty requested cycle block")
        return {
            "rows": rows,
            "ids": b22["ids"][mask],
            "cycles": b22["cycles"][mask],
            "y": b22["target"][mask],
            "stream_missing": stream_missing_all[rows],
            "legacy": b22["legacy"][mask],
            "R22": b22["policy"][mask],
            "R06": b06["policy"][mask],
            "midpoint": 0.5 * b06["policy"][mask] + 0.5 * b22["policy"][mask],
        }

    views: dict[str, dict[str, np.ndarray]] = {}
    windows: list[dict[str, np.ndarray]] = []
    for index, (origin, start, end) in enumerate(protocol["primary_blocks"], start=1):
        view = part(int(origin), int(start), int(end))
        require(np.array_equal(view["cycles"], train["cycle_code"].to_numpy(dtype=np.int64)[view["rows"]]), f"W{index}: cycle mismatch")
        views[f"W{index}"] = view
        windows.append(view)

    def concatenate(pieces: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        rows = np.concatenate([piece["rows"] for piece in pieces])
        require(rows.size == np.unique(rows).size, "view concatenation contains duplicate official Train rows")
        order = np.argsort(rows, kind="mergesort")
        return {key: np.concatenate([piece[key] for piece in pieces])[order] for key in pieces[0]}

    views["primary"] = concatenate(windows)
    views["late_primary"] = concatenate([views["W5"], views["W6"]])

    h12_pieces: list[dict[str, np.ndarray]] = []
    for origin, start, end in protocol["H12_blocks"]:
        key = f"H12_{int(origin)}"
        view = part(int(origin), int(start), int(end))
        views[key] = view
        h12_pieces.append(view)
    views["H12_pooled"] = concatenate(h12_pieces)

    expected_rows = {
        "primary": int(protocol["primary_rows"]),
        "late_primary": int(protocol["late_rows"]),
        "H12_pooled": int(protocol["H12_rows"]),
    }
    for name, expected_n in expected_rows.items():
        require(len(views[name]["rows"]) == expected_n, f"{name}: denominator differs from frozen protocol")
    require(len(views) == 11, "expected exactly 11 frozen views")
    return views


def summarize_views(views: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for name, view in views.items():
        metrics = {key: mae(view["y"], view[key]) for key in ("legacy", "R06", "R22", "midpoint")}
        summaries[name] = {
            "n": int(view["y"].size),
            "metrics": metrics,
            "gains_vs": {baseline: metrics[baseline] - metrics["midpoint"] for baseline in ("legacy", "R06", "R22")},
        }
    return summaries


def summarize_groups(views: dict[str, dict[str, np.ndarray]], minimum_n: int, maximum_harm: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_masks = {
        "stream_missing": lambda view: view["stream_missing"],
        "target_le5_evaluation_only": lambda view: view["y"] <= 5.0,
        "target_ge95_evaluation_only": lambda view: view["y"] >= 95.0,
    }
    for view_name, view in views.items():
        for field, mask_fn in group_masks.items():
            mask = np.asarray(mask_fn(view), dtype=bool)
            n = int(mask.sum())
            supported = n >= minimum_n
            harms = {
                baseline: (mae(view["y"][mask], view["midpoint"][mask]) - mae(view["y"][mask], view[baseline][mask])) if n else None
                for baseline in ("legacy", "R06", "R22")
            }
            guard_pass = (not supported) or all(harm is not None and harm <= maximum_harm for harm in harms.values())
            rows.append({"view": view_name, "field": field, "n": n, "supported": supported, **{f"harm_vs_{key}": value for key, value in harms.items()}, "guard_pass": bool(guard_pass)})
    return rows


def compare_root_outputs(protocol: dict[str, Any], summaries: dict[str, dict[str, Any]], group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(RESULT_PATH.is_file(), f"runner result not ready: {RESULT_PATH}")
    require(GROUPS_PATH.is_file(), f"runner groups CSV not ready: {GROUPS_PATH}")
    result_sha = sha256_file(RESULT_PATH)
    groups_sha = sha256_file(GROUPS_PATH)
    root_result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    root_views = root_result.get("views")
    require(isinstance(root_views, dict) and set(root_views) == set(summaries), "runner view names differ")
    require(root_result.get("official_fits", root_result.get("official_new_fits")) == 0, "runner reports nonzero official fits")
    require(root_result.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "runner result protocol hash mismatch")
    require(root_result.get("Test_access") is False, "runner result does not attest Test access=false")
    require(root_result.get("Kaggle_access") is False, "runner result does not attest Kaggle access=false")
    require(root_result.get("ledger_unchanged") is True, "runner result does not attest unchanged fit ledger")

    metric_checks: list[dict[str, Any]] = []
    for view_name, expected in summaries.items():
        got = root_views[view_name]
        require(int(got.get("n", -1)) == expected["n"], f"runner denominator mismatch: {view_name}")
        for metric_name, expected_value in expected["metrics"].items():
            actual = float(got["metrics"][metric_name])
            require(np.isclose(actual, expected_value, atol=METRIC_TOL, rtol=0.0), f"runner MAE mismatch {view_name}/{metric_name}: {actual} vs {expected_value}")
            metric_checks.append({"view": view_name, "metric": metric_name, "expected": expected_value, "actual": actual, "pass": True})
        for baseline, expected_gain in expected["gains_vs"].items():
            actual_gain = float(got["gains_vs"][baseline])
            require(np.isclose(actual_gain, expected_gain, atol=METRIC_TOL, rtol=0.0), f"runner gain mismatch {view_name}/{baseline}: {actual_gain} vs {expected_gain}")

    root_groups = pd.read_csv(GROUPS_PATH)
    key_cols = ["view", "field"]
    required_cols = set(key_cols + ["n", "supported", "harm_vs_legacy", "harm_vs_R06", "harm_vs_R22", "guard_pass"])
    require(required_cols.issubset(root_groups.columns), "runner groups CSV missing required columns")
    require(not root_groups.duplicated(key_cols).any(), "runner groups CSV has duplicate view/field rows")
    require(len(root_groups) == len(group_rows), "runner group row count mismatch")
    group_index = {(str(row["view"]), str(row["field"])): row for _, row in root_groups.iterrows()}
    group_checks = []
    for expected in group_rows:
        key = (expected["view"], expected["field"])
        require(key in group_index, f"runner group missing {key}")
        actual = group_index[key]
        require(int(actual["n"]) == expected["n"], f"runner subgroup denominator mismatch: {key}")
        actual_supported = str(actual["supported"]).strip().lower() in ("true", "1")
        actual_guard = str(actual["guard_pass"]).strip().lower() in ("true", "1")
        require(actual_supported == expected["supported"], f"runner subgroup support mismatch: {key}")
        require(actual_guard == expected["guard_pass"], f"runner subgroup guard mismatch: {key}")
        for baseline in ("legacy", "R06", "R22"):
            col = f"harm_vs_{baseline}"
            actual_harm = None if pd.isna(actual[col]) else float(actual[col])
            expected_harm = expected[col]
            require((actual_harm is None and expected_harm is None) or (actual_harm is not None and expected_harm is not None and np.isclose(actual_harm, expected_harm, atol=METRIC_TOL, rtol=0.0)), f"runner subgroup harm mismatch: {key}/{baseline}")
        group_checks.append({"view": key[0], "field": key[1], "n": expected["n"], "pass": True})
    return {
        "runner_result_sha256": result_sha,
        "runner_groups_sha256": groups_sha,
        "metric_comparisons": metric_checks,
        "group_comparisons": group_checks,
    }


def compute_gates(protocol: dict[str, Any], summaries: dict[str, dict[str, Any]], groups: list[dict[str, Any]]) -> dict[str, Any]:
    g = protocol["gates"]
    primary = summaries["primary"]
    late = summaries["late_primary"]
    windows = [summaries[f"W{i}"] for i in range(1, 7)]
    h12_views = [summaries["H12_55"], summaries["H12_67"], summaries["H12_pooled"]]
    window_gains = [window["gains_vs"]["R06"] for window in windows]
    subgroup_failures = [
        {"view": row["view"], "field": row["field"], "n": row["n"], "failed_baselines": [baseline for baseline in ("legacy", "R06", "R22") if row["supported"] and row[f"harm_vs_{baseline}"] > g["subgroup_harm_vs_each_baseline_max"]]}
        for row in groups
        if row["supported"] and not row["guard_pass"]
    ]
    checks = {
        "primary_gain_vs_R06_min": primary["gains_vs"]["R06"] >= g["primary_gain_vs_R06_min"],
        "primary_gain_vs_R22_min": primary["gains_vs"]["R22"] >= g["primary_gain_vs_R22_min"],
        "H12_harm_vs_each_component_max": all(
            (view["metrics"]["midpoint"] - view["metrics"][baseline]) <= g["H12_harm_vs_each_component_max"]
            for view in h12_views for baseline in ("R06", "R22")
        ),
        "late_gain_vs_R06_min": late["gains_vs"]["R06"] >= g["late_gain_vs_R06_min"],
        "nonnegative_primary_windows_vs_R06_min": sum(gain >= 0.0 for gain in window_gains) >= g["nonnegative_primary_windows_vs_R06_min"],
        "individual_window_harm_vs_R06_max": max(-gain for gain in window_gains) <= g["individual_window_harm_vs_R06_max"],
        "supported_subgroups_harm_vs_each_baseline_max": not subgroup_failures,
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": "CANDIDATE_FOR_REVIEW" if all(checks.values()) else "NO_GO",
        "primary_gain_vs_R06": primary["gains_vs"]["R06"],
        "primary_gain_vs_R22": primary["gains_vs"]["R22"],
        "late_gain_vs_R06": late["gains_vs"]["R06"],
        "primary_window_gains_vs_R06": {f"W{i}": window_gains[i - 1] for i in range(1, 7)},
        "nonnegative_primary_windows_vs_R06": sum(gain >= 0.0 for gain in window_gains),
        "subgroup_failures": subgroup_failures,
    }


def run(output_path: Path) -> dict[str, Any]:
    require(not output_path.exists(), f"refusing to overwrite verifier output: {output_path}")
    protocol, protocol_sha = read_protocol()
    input_hashes = verify_input_hashes(protocol)
    train = load_train(protocol)
    mapping = {
        relative: expected
        for relative, expected in protocol["input_sha256"].items()
        if relative.endswith(".npz")
    }
    bundles: dict[tuple[str, int], dict[str, Any]] = {}
    for relative_npz in mapping:
        filename = Path(relative_npz).name
        card = filename.split("_fit", 1)[0]
        cutoff = int(filename.split("_fit", 1)[1].split(".", 1)[0])
        require(card in ("R06", "R22"), f"unexpected frozen cache card: {card}")
        key = (card, cutoff)
        require(key not in bundles, f"duplicate frozen cache identity: {key}")
        bundles[key] = load_bundle(card, cutoff, relative_npz, train)
    expected_keys = {(card, cutoff) for card in ("R06", "R22") for cutoff in protocol["origins"]}
    require(set(bundles) == expected_keys, "frozen cache identities do not match six protocol origins")

    views = build_views(protocol, bundles, train)
    summaries = summarize_views(views)
    group_rows = summarize_groups(
        views,
        minimum_n=int(protocol["gates"]["minimum_subgroup_n"]),
        maximum_harm=float(protocol["gates"]["subgroup_harm_vs_each_baseline_max"]),
    )
    root_comparison = compare_root_outputs(protocol, summaries, group_rows)
    gates = compute_gates(protocol, summaries, group_rows)

    root_result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    require(bool(root_result.get("gate", {}).get("passed")) == gates["passed"], "runner overall gate differs from independent gate")
    root_failed = root_result.get("gate", {}).get("failed_groups", [])
    root_failed_keys = set()
    for row in root_failed:
        if isinstance(row, dict):
            root_failed_keys.add((row.get("view"), row.get("field")))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            root_failed_keys.add((row[0], row[1]))
    expected_failed_keys = {(row["view"], row["field"]) for row in gates["subgroup_failures"]}
    require(root_failed_keys == expected_failed_keys, "runner failed-group identities differ from independent subgroup guards")

    return {
        "schema_version": 1,
        "status": "PASS",
        "verifier": "independent direct-NPZ rank/metric/group reconstruction; no runner helper imported",
        "protocol_sha256": protocol_sha,
        "source_head": protocol.get("source_head"),
        "official_fits": 0,
        "test_opened": False,
        "kaggle_called": False,
        "input_files_verified": len(input_hashes),
        "bundle_count": len(bundles),
        "bundle_checks": [
            {"card": card, "cutoff": cutoff, "npz_sha256": bundle["npz_sha256"], "sidecar_sha256": bundle["sidecar_sha256"], "pass": True}
            for (card, cutoff), bundle in sorted(bundles.items())
        ],
        "view_count": len(views),
        "view_summaries": summaries,
        "group_rows": group_rows,
        "gates": gates,
        "runner_artifact_comparison": root_comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    try:
        report = run(output)
    except Exception as exc:  # fail closed; preserve error in a separate report artifact
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "official_fits": 0,
            "test_opened": False,
            "kaggle_called": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        output.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "view_count": report["view_count"], "bundle_count": report["bundle_count"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
