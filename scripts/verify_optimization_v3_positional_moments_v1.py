"""Independent prelabel and scored-result checks for POS_MOMENTS_V1.

The prelabel path reads only official Train features and safe named prediction
arrays. The scored path is separate and opens only the temporal Train query
rows after the runner's prelabel receipt is present.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)

from src.models.positional_text_features import deterministic_order_shuffle


CARD_ID = "POS_MOMENTS_V1"
FIT_IDS = tuple(f"{CARD_ID}_E{o}_{arm}" for o in (67, 75) for arm in ("REAL", "SHUFFLED"))
GATE_TOLERANCE = 1e-12
TRAIN_REL = "data/raw/classification/Train.csv"
PILOT_REL = "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
BAG_REL = "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
A02_REL = "experiments/runs/optimization_v3_a02_neighbors_v1"
LEDGER_REL = "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"


class VerificationError(RuntimeError):
    pass


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(_canonical(value), sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: Any) -> str:
    array = np.asarray(value)
    header = _json_hash({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _csr_hash(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256(_json_hash({"shape": matrix.shape, "dtype": str(matrix.dtype)}).encode("ascii"))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object at {path}")
    return value


def _named_npz(path: Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    # A02 archives contain historical labels; only the explicit safe keys below
    # are materialized from the archive.
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(keys) - set(archive.files))
        if missing:
            raise VerificationError(f"Safe NPZ keys missing at {path}: {missing}")
        return {key: np.asarray(archive[key]).copy() for key in keys}


def _normalize_docs(values: pd.Series) -> tuple[list[str], np.ndarray]:
    docs: list[str] = []
    missing = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values.to_numpy(dtype=object)):
        if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
            docs.append("")
            missing[i] = True
        else:
            text = str(value).strip()
            docs.append(text)
            missing[i] = not bool(text)
    return docs, missing


def _rebuild_moments(documents: Sequence[str], vectorizer: Any, *, shuffled: bool) -> sparse.csr_matrix:
    """Independent literal implementation of the frozen three-channel formula."""
    vocab = {str(token): int(index) for token, index in vectorizer.vocabulary_.items()}
    unigram_original = {token: index for token, index in vocab.items() if " " not in token}
    unigram = {token: compact for compact, token in enumerate(
        sorted(unigram_original, key=unigram_original.__getitem__))}
    idf = np.asarray(vectorizer.idf_, dtype=np.float64)
    preprocess, tokenizer = vectorizer.build_preprocessor(), vectorizer.build_tokenizer()
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    d = len(unigram)
    for row_no, value in enumerate(documents):
        tokens = tuple(tokenizer(preprocess(str(value).strip())))
        if shuffled:
            tokens = tuple(deterministic_order_shuffle(tokens, seed=42)["tokens"])
        n = len(tokens)
        if n <= 1:
            position1 = np.zeros(n, dtype=np.float64)
            position2 = np.zeros(n, dtype=np.float64)
        else:
            u = 2.0 * np.arange(n, dtype=np.float64) / (n - 1) - 1.0
            position1 = u
            raw = (3.0 * u**2 - 1.0) / 2.0
            position2 = raw - raw.mean()
        occurrences: dict[str, list[int]] = {}
        for i, token in enumerate(tokens):
            if token in unigram:
                counts = occurrences.setdefault(token, [])
                counts.append(i)
        raw_bag: dict[str, float] = {
            token: (1.0 + math.log(len(pos))) * float(idf[unigram_original[token]])
            for token, pos in occurrences.items()
        }
        norm_bag = math.sqrt(sum(value * value for value in raw_bag.values()))
        if norm_bag == 0.0:
            continue
        row_features: list[tuple[int, float]] = []
        for token, positions in occurrences.items():
            j = unigram[token]
            b = raw_bag[token] / norm_bag
            m1 = float(position1[positions].mean())
            m2 = float(position2[positions].mean())
            row_features.extend(((j, b), (d + j, b * m1), (2 * d + j, b * m2)))
        joint_norm = math.sqrt(sum(x * x for _, x in row_features))
        if not math.isfinite(joint_norm) or joint_norm <= 0:
            raise VerificationError("Independent moment reconstruction encountered invalid row norm")
        for j, x in row_features:
            if x != 0.0:
                rows.append(row_no)
                cols.append(j)
                data.append(x / joint_norm)
    out = sparse.csr_matrix((np.asarray(data, dtype=np.float64),
                             (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
                            shape=(len(documents), 3 * d), dtype=np.float64)
    out.sum_duplicates()
    out.sort_indices()
    return out


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "runner": root / "scripts/optimization_v3_positional_moments_v1.py",
        "verifier": root / "scripts/verify_optimization_v3_positional_moments_v1.py",
        "moments_model": root / "src/models/positional_moments_v1.py",
        "offset_model": root / "src/models/positional_bag_offset_v1.py",
        "pilot_builder": root / "src/models/optimization_v3_positional_text_pilot_v1.py",
        "classification_models": root / "src/models/classification_models.py",
        "position_shuffle": root / "src/models/positional_text_features.py",
        "offset_runner": root / "scripts/optimization_v3_positional_bag_offset_v1.py",
        "contrast_runner": root / "scripts/optimization_v3_positional_contrast_v1.py",
        "tests": root / "tests/test_optimization_v3_positional_moments_v1.py",
        "recipe": root / "reports/optimization_v3/positional_moments_v1/RECIPE.json",
        "protocol": root / "reports/optimization_v3/positional_moments_v1/SOL_EXECUTION_HANDOFF.md",
    }


def _verify_freeze(root: Path, freeze: Mapping[str, Any]) -> None:
    unsigned = dict(freeze)
    signature = unsigned.pop("freeze_signature", None)
    if signature != _json_hash(unsigned):
        raise VerificationError("Freeze signature is invalid")
    for name, expected in freeze["source_hashes"].items():
        path = _source_paths(root).get(name)
        if path is None or not path.is_file() or _file_hash(path) != expected:
            raise VerificationError(f"Frozen source hash differs: {name}")
    if _file_hash(root / TRAIN_REL) != freeze.get("input_sha256"):
        raise VerificationError("Official Train source hash differs from prefit freeze")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", freeze["source_commit"], head], cwd=root)
    if ancestry.returncode != 0:
        raise VerificationError("Frozen source commit is not an ancestor of current HEAD")


def _origin_inputs(root: Path, origin: int, frame: pd.DataFrame, docs: list[str]) -> dict[str, Any]:
    pilot = root / PILOT_REL
    bag_run = root / BAG_REL
    a02_root = root / A02_REL / f"E{origin}/observed"
    pred_path = pilot / f"E{origin}/predictions_E{origin}.npz"
    pred = _named_npz(pred_path, ["origin", "fit_rows", "score_rows", "score_record_ids", "score_cycles",
                                  "present_mask", "matrix_manifest_sha256", "prob_bag", "prob_missing_hgb"])
    manifest_path = pilot / f"MATRIX_MANIFEST_E{origin}.json"
    manifest = _read_json(manifest_path)
    parent_ready = _read_json(pilot / "PREDICTIONS_READY.json")
    if (_file_hash(pred_path) != parent_ready["predictions"][f"E{origin}"]["sha256"]
            or str(pred["matrix_manifest_sha256"].reshape(-1)[0]) != _file_hash(manifest_path)):
        raise VerificationError(f"E{origin} parent manifest/prediction binding mismatch")
    fit_rows = pred["fit_rows"].astype(np.int64)
    score_rows = pred["score_rows"].astype(np.int64)
    ids_all = frame["record_id"].astype(str).to_numpy()
    cycles_all = frame["cycle_code"].to_numpy(dtype=np.int64)
    if (not np.array_equal(fit_rows, np.flatnonzero(cycles_all <= origin))
            or not np.array_equal(score_rows, np.flatnonzero(cycles_all > origin))
            or not np.array_equal(ids_all[score_rows], pred["score_record_ids"].astype(str))
            or not np.array_equal(cycles_all[score_rows], pred["score_cycles"].astype(np.int64))):
        raise VerificationError(f"E{origin} parent rows, IDs, or cycle identities differ")
    present = pred["present_mask"].astype(bool)
    if not np.array_equal(frame["missing_stream"].to_numpy(dtype=bool)[score_rows], ~present):
        raise VerificationError(f"E{origin} stream availability differs from official Train")
    matrix_meta: dict[str, sparse.csr_matrix] = {}
    for part in ("fit_base", "score_base"):
        item = manifest["matrices"][part]
        path = root / item["path"]
        if not path.is_file() or _file_hash(path) != item["sha256"]:
            raise VerificationError(f"E{origin} parent base-matrix file hash mismatch: {part}")
        matrix = sparse.load_npz(path).tocsr().astype(np.float64)
        if list(matrix.shape) != item["shape"] or _csr_hash(matrix) != item["logical_sha256"]:
            raise VerificationError(f"E{origin} parent base-matrix logical hash mismatch: {part}")
        matrix_meta[part] = matrix

    accounting = _read_json(root / "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json")
    fit_units = {item["fit_id"]: item for item in accounting["fit_units"]}
    bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
    vec_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_C2B_TFIDF"
    bag_path, bag_receipt_path = pilot / "units" / f"{bag_id}.joblib", pilot / "receipts" / f"{bag_id}.json"
    vec_path, vec_receipt_path = pilot / "units" / f"{vec_id}.joblib", pilot / "receipts" / f"{vec_id}.json"
    for path, unit, digest_key in ((bag_path, fit_units[bag_id], "artifact_sha256"),
                                   (bag_receipt_path, fit_units[bag_id], "receipt_sha256"),
                                   (vec_path, fit_units[vec_id], "artifact_sha256"),
                                   (vec_receipt_path, fit_units[vec_id], "receipt_sha256")):
        if _file_hash(path) != unit[digest_key]:
            raise VerificationError(f"E{origin} frozen parent component hash mismatch: {path.name}")
    bag_state = joblib.load(bag_path)
    bag = bag_state.get("estimator") if isinstance(bag_state, dict) else bag_state
    vec_state = joblib.load(vec_path)
    vectorizer = vec_state.get("vectorizer") if isinstance(vec_state, dict) else vec_state
    if (list(np.asarray(bag.classes_).tolist()) != [0, 1]
            or np.asarray(bag.coef_).shape != (1, matrix_meta["fit_base"].shape[1])
            or not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_")):
        raise VerificationError(f"E{origin} parent estimator/vectorizer state dimensions are invalid")
    if (fit_units[bag_id]["fit_metadata"].get("coef_sha256") != _array_hash(bag.coef_)
            or fit_units[bag_id]["fit_metadata"].get("intercept_sha256") != _array_hash(bag.intercept_)):
        raise VerificationError(f"E{origin} BAG_LR parameters differ from their signed fit metadata")
    idf = np.asarray(vectorizer.idf_, dtype=np.float64)
    vocabulary = {str(token): int(index) for token, index in vectorizer.vocabulary_.items()}
    parameter_values = {}
    for key, value in vectorizer.get_params(deep=False).items():
        if isinstance(value, type) and value.__module__.startswith("numpy"):
            parameter_values[key] = f"{value.__module__}.{value.__qualname__}"
        elif isinstance(value, tuple):
            parameter_values[key] = list(value)
        else:
            parameter_values[key] = value
    vectorizer_signature = {
        "parameters": parameter_values,
        "vocabulary_idf_sha256": _json_hash({
            "vocabulary": sorted(vocabulary.items()), "idf_sha256": _array_hash(idf),
        }),
        "vocabulary_size": len(vocabulary),
        "unigram_size": sum(" " not in token for token in vocabulary),
    }
    z0 = np.asarray(matrix_meta["score_base"] @ bag.coef_[0], dtype=np.float64).reshape(-1) + float(bag.intercept_[0])
    if not np.allclose(expit(z0), pred["prob_bag"], rtol=0.0, atol=1e-12):
        raise VerificationError(f"E{origin} frozen BAG_LR probability reconstruction mismatch")

    bag_ready = _read_json(bag_run / "PREDICTIONS_READY.json")
    bag_bundle_path = bag_run / f"E{origin}/predictions_E{origin}.npz"
    if _file_hash(bag_bundle_path) != bag_ready["predictions"][f"E{origin}"]["sha256"]:
        raise VerificationError(f"E{origin} original pP bundle hash mismatch")
    bag_pred = _named_npz(bag_bundle_path, ["score_rows", "score_record_ids", "score_cycles", "present_mask",
                                            "prob_missing_hgb", "prob_offset_real"])
    p_old = bag_pred["prob_offset_real"].astype(np.float64)
    if (not np.array_equal(bag_pred["score_rows"], score_rows)
            or not np.array_equal(bag_pred["score_record_ids"].astype(str), ids_all[score_rows])
            or not np.array_equal(bag_pred["score_cycles"].astype(np.int64), cycles_all[score_rows])
            or not np.array_equal(p_old[~present], bag_pred["prob_missing_hgb"][~present])):
        raise VerificationError(f"E{origin} pP identity or missing fallback mismatch")

    a02_meta = _read_json(a02_root / "predictions.json")
    a02_receipt = _read_json(a02_root / "fit_receipt.json")
    a02_path = a02_root / "predictions.npz"
    if (a02_meta.get("fit_id") != f"A02_E{origin}_OBSERVED"
            or a02_meta.get("query_labels_passed_to_prediction") is not False
            or a02_receipt.get("query_labels_passed_to_fit") is not False
            or a02_receipt.get("test_data_used") is not False):
        raise VerificationError(f"E{origin} A02 source-boundary receipt mismatch")
    a02 = _named_npz(a02_path, ["score_rows", "score_record_ids", "score_cycles", "prediction", "missing_mask"])
    if (not np.array_equal(a02["score_rows"], score_rows)
            or not np.array_equal(a02["score_record_ids"].astype(str), ids_all[score_rows])
            or not np.array_equal(a02["score_cycles"].astype(np.int64), cycles_all[score_rows])
            or not np.array_equal(a02["missing_mask"].astype(bool), ~present)):
        raise VerificationError(f"E{origin} A02 rows, IDs, cycles, or availability differ")
    p_a02 = a02["prediction"].astype(np.float64)
    if not np.isfinite(p_a02).all() or np.any((p_a02 < 0) | (p_a02 > 1)):
        raise VerificationError(f"E{origin} A02 probabilities are invalid")

    text_fit = pd.to_numeric(frame.iloc[fit_rows]["text_load"], errors="raise").to_numpy(dtype=np.float64)
    edges = np.quantile(text_fit, [0.25, 0.5, 0.75], method="linear")
    group = np.searchsorted(edges, pd.to_numeric(frame.iloc[score_rows]["text_load"], errors="raise"), side="right").astype(np.int8)
    entry = pd.to_numeric(frame.iloc[score_rows]["entry_friction"], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(entry, [0, 1]).all():
        raise VerificationError(f"E{origin} entry-friction subgroup is not binary")
    return {
        "origin": origin, "fit_rows": fit_rows, "score_rows": score_rows,
        "score_ids": ids_all[score_rows], "score_cycles": cycles_all[score_rows],
        "present": present, "p_old": p_old, "p_a02": p_a02, "p_bag": pred["prob_bag"].astype(np.float64),
        "z0_score": z0, "vectorizer": vectorizer, "docs": [docs[int(i)] for i in score_rows],
        "fit_target_sha256": fit_units[bag_id]["fit_population"]["fit_target_sha256"],
        "tokenizer_unigram_size": sum(" " not in str(token) for token in vectorizer.vocabulary_),
        "tokenizer_signature": vectorizer_signature,
        "entry_friction": entry, "text_load_group": group,
        "manifest": manifest, "matrices": matrix_meta,
        "bag_ready_sha256": _file_hash(bag_run / "PREDICTIONS_READY.json"),
        "a02_metadata_sha256": _file_hash(a02_root / "predictions.json"),
        "a02_receipt_sha256": _file_hash(a02_root / "fit_receipt.json"),
        "a02_probability_sha256": _array_hash(p_a02),
    }


def verify_prelabel_outputs(root: Path, run_root: Path, freeze_data: Mapping[str, Any]) -> dict[str, Any]:
    root, run_root = Path(root), Path(run_root)
    _verify_freeze(root, freeze_data)
    train = pd.read_csv(root / TRAIN_REL, usecols=["record_id", "cycle_code", "signal_stream", "text_load", "entry_friction"],
                        dtype={"record_id": "string"})
    if len(train) != 50608 or train["record_id"].isna().any() or train["record_id"].duplicated().any():
        raise VerificationError("Official classification Train feature table identity is invalid")
    train["cycle_code"] = pd.to_numeric(train["cycle_code"], errors="raise").astype(np.int64)
    docs, missing = _normalize_docs(train["signal_stream"])
    train["missing_stream"] = missing
    ready_path = run_root / "PREDICTIONS_READY.json"
    ready_report = root / "reports/optimization_v3/positional_moments_v1/PREDICTIONS_READY.json"
    ready = _read_json(ready_path)
    if ready != _read_json(ready_report):
        raise VerificationError("Tracked and run PREDICTIONS_READY receipts differ")
    if (ready.get("status") != "PASS_PREDICTIONS_DURABLE_PRELABEL_CLOSED"
            or ready.get("card_id") != CARD_ID or ready.get("fit_ids") != list(FIT_IDS)
            or ready.get("freeze_signature") != freeze_data["freeze_signature"]
            or ready.get("query_labels_read") is not False or ready.get("test_or_sample_access") is not False):
        raise VerificationError("PREDICTIONS_READY boundary or identity receipt is invalid")
    ledger = []
    for line in (root / LEDGER_REL).read_text(encoding="utf-8").splitlines():
        if line.strip():
            ledger.append(json.loads(line))
    starts = [e for e in ledger if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    saves = [e for e in ledger if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    failures = [e for e in ledger if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    if ([e.get("fit_id") for e in starts] != list(FIT_IDS)
            or [e.get("fit_id") for e in saves] != list(FIT_IDS) or failures):
        raise VerificationError("Official ledger does not show four ordered successful POS_MOMENTS fits")
    amendment = [e for e in ledger if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == "POS_MOMENTS_V1_CAP142_2026_11_04"]
    reservation = [e for e in ledger if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == "POS_MOMENTS_V1:PILOT_4"]
    if (len(amendment) != 1 or len(reservation) != 1 or len(ledger) != 356
            or len([e for e in ledger if e.get("event") == "FIT_STARTED"]) != 137
            or len([e for e in ledger if e.get("event") == "FIT_SAVED"]) != 131
            or len([e for e in ledger if e.get("event") == "FIT_FAILED"]) != 6
            or amendment[0].get("global_cap") != 142 or amendment[0].get("effective_caps") !=
               {"continuation": 51, "control_replay": 2, "pilot": 88, "technical_retry": 1}
            or amendment[0].get("source_commit") != freeze_data["source_commit"]
            or amendment[0].get("freeze_signature") != freeze_data["freeze_signature"]
            or reservation[0].get("fit_ids") != list(FIT_IDS)
            or reservation[0].get("freeze_signature") != freeze_data["freeze_signature"]):
        raise VerificationError("Shared cap amendment, reservation, or pilot fit budget counts differ")
    start_by_id = {event["fit_id"]: event for event in starts}
    for event in [*starts, *saves]:
        if (event.get("card_id") != CARD_ID or event.get("reservation_key") != "POS_MOMENTS_V1:PILOT_4"
                or event.get("amendment_id") != "POS_MOMENTS_V1_CAP142_2026_11_04"
                or event.get("source_commit") != freeze_data["source_commit"]
                or event.get("freeze_signature") != freeze_data["freeze_signature"]):
            raise VerificationError("A pilot fit ledger event is not bound to the frozen reservation")

    origins: dict[str, Any] = {}
    for origin in (67, 75):
        key = f"E{origin}"
        state = _origin_inputs(root, origin, train, docs)
        frozen_origin = freeze_data["origins"][key]
        if (len(state["fit_rows"]) != frozen_origin["fit_rows"]
                or len(state["score_rows"]) != frozen_origin["score_rows"]
                or state["fit_target_sha256"] != frozen_origin["fit_target_sha256"]
                or state["tokenizer_signature"] != frozen_origin["tokenizer_signature"]
                or state["bag_ready_sha256"] != frozen_origin["bag_prediction_ready_sha256"]
                or state["a02_metadata_sha256"] != frozen_origin["a02_metadata_sha256"]
                or state["a02_receipt_sha256"] != frozen_origin["a02_receipt_sha256"]
                or state["a02_probability_sha256"] != frozen_origin["a02_probability_sha256"]):
            raise VerificationError(f"E{origin} frozen fit population or unigram identity differs")
        parent_pred_path = root / PILOT_REL / f"E{origin}/predictions_E{origin}.npz"
        parent_manifest_path = root / PILOT_REL / f"MATRIX_MANIFEST_E{origin}.json"
        if (_file_hash(parent_pred_path) != frozen_origin["prediction_bundle_sha256"]
                or _file_hash(parent_manifest_path) != frozen_origin["manifest_sha256"]):
            raise VerificationError(f"E{origin} parent prediction/manifest differs from preflight freeze")
        out_path = root / ready["predictions"][key]["path"]
        if not out_path.is_file() or _file_hash(out_path) != ready["predictions"][key]["sha256"]:
            raise VerificationError(f"E{origin} output bundle hash mismatch")
        bundle_keys = ["score_rows", "score_record_ids", "score_cycles", "present_mask", "entry_friction",
                       "text_load_group", "prob_pP", "prob_a02", "prob_bag", "prob_mix50"]
        for arm in ("real", "shuffled"):
            bundle_keys += [f"prob_q_{arm}", f"prob_candidate_{arm}", f"fit_receipt_sha256_{arm}"]
        bundle = _named_npz(out_path, bundle_keys)
        if (not np.array_equal(bundle["score_rows"], state["score_rows"])
                or not np.array_equal(bundle["score_record_ids"].astype(str), state["score_ids"])
                or not np.array_equal(bundle["score_cycles"].astype(np.int64), state["score_cycles"])
                or not np.array_equal(bundle["present_mask"].astype(bool), state["present"])
                or not np.array_equal(bundle["entry_friction"].astype(np.int8), state["entry_friction"])
                or not np.array_equal(bundle["text_load_group"].astype(np.int8), state["text_load_group"])
                or not np.array_equal(bundle["prob_pP"], state["p_old"])
                or not np.array_equal(bundle["prob_a02"], state["p_a02"])
                or not np.array_equal(bundle["prob_bag"], state["p_bag"])):
            raise VerificationError(f"E{origin} prediction bundle identity/parent probabilities differ")
        reconstructed: dict[str, Any] = {}
        for arm, shuffled in (("REAL", False), ("SHUFFLED", True)):
            arm_key = arm.lower()
            fit_id = f"{CARD_ID}_E{origin}_{arm}"
            fit_receipt_path = run_root / "receipts" / f"{fit_id}.json"
            fit_artifact_path = run_root / "units" / f"{fit_id}.npz"
            receipt = _read_json(fit_receipt_path)
            if (receipt.get("fit_id") != fit_id or receipt.get("freeze_signature") != freeze_data["freeze_signature"]
                    or receipt.get("source_commit") != freeze_data["source_commit"]
                    or receipt.get("source_hashes") != freeze_data["source_hashes"]
                    or receipt.get("input_sha256") != freeze_data["input_sha256"]
                    or receipt.get("fit_population", {}).get("fit_target_sha256") != state["fit_target_sha256"]
                    or receipt.get("optimizer", {}).get("c") != 1.0
                    or receipt.get("optimizer", {}).get("options") != {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}
                    or receipt.get("optimizer", {}).get("success") is not True):
                raise VerificationError(f"{fit_id} fit receipt/frozen optimizer contract mismatch")
            saved_event = next(event for event in saves if event.get("fit_id") == fit_id)
            if (receipt.get("fit_start_event_sha256") != _json_hash(start_by_id[fit_id])
                    or saved_event.get("receipt_sha256") != _file_hash(fit_receipt_path)):
                raise VerificationError(f"{fit_id} start/save ledger receipt binding mismatch")
            if _file_hash(fit_artifact_path) != receipt.get("artifact_sha256"):
                raise VerificationError(f"{fit_id} gamma artifact hash differs from receipt")
            gamma = _named_npz(fit_artifact_path, ["gamma"])["gamma"].astype(np.float64)
            if _array_hash(gamma) != receipt.get("gamma_sha256"):
                raise VerificationError(f"{fit_id} gamma logical hash differs from receipt")
            moment_ref = frozen_origin["moments"][arm_key]["score"]
            matrix_path = run_root / f"E{origin}/matrices/score_{arm_key}.npz"
            matrix = sparse.load_npz(matrix_path).tocsr().astype(np.float64)
            rebuilt = _rebuild_moments(state["docs"], state["vectorizer"], shuffled=shuffled)
            if (_csr_hash(matrix) != moment_ref["sha256"] or _csr_hash(rebuilt) != moment_ref["sha256"]
                    or matrix.shape != tuple(moment_ref["shape"]) or len(gamma) != matrix.shape[1]
                    or receipt.get("matrix", {}).get("sha256") != frozen_origin["moments"][arm_key]["fit"]["sha256"]
                    or receipt.get("matrix", {}).get("shape") != frozen_origin["moments"][arm_key]["fit"]["shape"]):
                raise VerificationError(f"E{origin} {arm} W matrix or gamma dimension differs from preflight")
            q = expit(state["z0_score"] + np.asarray(rebuilt @ gamma, dtype=np.float64).reshape(-1))
            candidate = np.where(state["present"], 0.5 * q + 0.5 * state["p_a02"], state["p_old"])
            if (not np.isfinite(q).all() or np.any((q < 0) | (q > 1))
                    or not np.array_equal(bundle[f"prob_q_{arm_key}"], q)
                    or not np.array_equal(bundle[f"prob_candidate_{arm_key}"], candidate)
                    or str(bundle[f"fit_receipt_sha256_{arm_key}"].reshape(-1)[0]) != _file_hash(fit_receipt_path)):
                raise VerificationError(f"E{origin} {arm} prediction is not independently reproducible")
            reconstructed[arm_key] = {"gamma_sha256": _array_hash(gamma), "receipt_sha256": _file_hash(fit_receipt_path),
                                      "score_matrix_sha256": _csr_hash(rebuilt), "prediction_rows": len(q)}
        expected_mix = np.where(state["present"], 0.5 * state["p_old"] + 0.5 * state["p_a02"], state["p_old"])
        if not np.array_equal(bundle["prob_mix50"], expected_mix):
            raise VerificationError(f"E{origin} fixed MIX50 comparator formula differs")
        origins[key] = {
            "rows": len(state["score_rows"]), "present": int(state["present"].sum()),
            "missing": int((~state["present"]).sum()), "score_ids_sha256": _array_hash(state["score_ids"]),
            "score_cycles_sha256": _array_hash(state["score_cycles"]), "fits": reconstructed,
            "prediction_bundle_sha256": _file_hash(out_path),
        }
    return {
        "schema_version": 1, "status": "PASS", "card_id": CARD_ID,
        "freeze_signature": freeze_data["freeze_signature"],
        "predictions_ready_sha256": _file_hash(ready_path), "origins": origins,
        "query_labels_read": False, "test_or_sample_access": False,
        "checks": ["source and Train hashes", "parent source lineage", "safe named arrays", "row/ID/cycle alignment",
                   "fitted gamma and receipt binding", "independent W reconstruction", "fixed routing and MIX50"] ,
    }


def verify_synthetic_prelabel(
    matrix: sparse.spmatrix, gamma: Any, z0: Any, p_a02: Any, p_old: Any, present: Any,
    probability: Any, candidate: Any, mix50: Any,
) -> dict[str, Any]:
    """Exercise label-closed prediction and fixed routing on synthetic rows."""
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    offset = np.asarray(z0, dtype=np.float64).reshape(-1)
    x = sparse.csr_matrix(matrix, dtype=np.float64)
    expected_q = expit(offset + np.asarray(x @ coef, dtype=np.float64).reshape(-1))
    a02 = np.asarray(p_a02, dtype=np.float64).reshape(-1)
    old = np.asarray(p_old, dtype=np.float64).reshape(-1)
    mask = np.asarray(present, dtype=bool).reshape(-1)
    expected_candidate = np.where(mask, 0.5 * expected_q + 0.5 * a02, old)
    expected_mix = np.where(mask, 0.5 * old + 0.5 * a02, old)
    values = [np.asarray(probability, dtype=np.float64).reshape(-1),
              np.asarray(candidate, dtype=np.float64).reshape(-1),
              np.asarray(mix50, dtype=np.float64).reshape(-1)]
    if (not all(np.isfinite(value).all() for value in [expected_q, expected_candidate, expected_mix, *values])
            or not np.array_equal(values[0], expected_q)
            or not np.array_equal(values[1], expected_candidate)
            or not np.array_equal(values[2], expected_mix)
            or not np.array_equal(expected_candidate[~mask], old[~mask])):
        raise VerificationError("Synthetic prelabel q/fixed-policy/missing route failed independent reconstruction")
    return {"status": "PASS", "rows": len(offset), "query_labels_read": False,
            "fixed_blend": True, "missing_route_exact": True}


def _metric(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.int8).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    pred = (p >= 0.5).astype(np.int8)
    two = len(np.unique(y)) == 2
    return {
        "n": len(y), "negative": int((y == 0).sum()), "positive": int((y == 1).sum()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).astype(int).tolist(),
        "roc_auc": float(roc_auc_score(y, p)) if two else None,
        "average_precision": float(average_precision_score(y, p)) if two else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _view_masks(origin: int, cycles: np.ndarray) -> dict[str, np.ndarray]:
    if origin == 67:
        return {"H12_E67": (cycles >= 68) & (cycles <= 79),
                "W4": (cycles >= 68) & (cycles <= 71), "W5": (cycles >= 72) & (cycles <= 75),
                "W6": (cycles >= 76) & (cycles <= 79)}
    return {"W6": (cycles >= 76) & (cycles <= 79)}


def _recompute_origin(origin: int, bundle: Mapping[str, np.ndarray], y: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cycles = bundle["score_cycles"].astype(np.int64)
    present = bundle["present_mask"].astype(bool)
    entry = bundle["entry_friction"].astype(np.int8)
    load = bundle["text_load_group"].astype(np.int8)
    policies = {
        "moment_real": bundle["prob_candidate_real"], "moment_shuffled": bundle["prob_candidate_shuffled"],
        "MIX50": bundle["prob_mix50"], "q_real": bundle["prob_q_real"],
        "q_shuffled": bundle["prob_q_shuffled"], "pP_oldPOS": bundle["prob_pP"],
        "A02": bundle["prob_a02"], "bag": bundle["prob_bag"],
    }
    views: dict[str, Any] = {}
    gates: list[dict[str, Any]] = []
    subgroups: list[dict[str, Any]] = []
    per_cycle: list[dict[str, Any]] = []
    groups = [("availability_present", present), ("availability_missing", ~present),
              ("entry_friction_0", entry == 0), ("entry_friction_1", entry == 1)]
    groups.extend((f"text_load_Q{i+1}", load == i) for i in range(4))
    for view, mask in _view_masks(origin, cycles).items():
        metrics = {key: _metric(y[mask], value[mask]) for key, value in policies.items()}
        delta_rm = metrics["moment_real"]["macro_f1"] - metrics["MIX50"]["macro_f1"]
        delta_rs = metrics["moment_real"]["macro_f1"] - metrics["moment_shuffled"]["macro_f1"]
        delta_sm = metrics["moment_shuffled"]["macro_f1"] - metrics["MIX50"]["macro_f1"]
        views[view] = {"n": int(mask.sum()), "metrics": metrics,
                       "deltas": {"real_minus_MIX50": delta_rm, "real_minus_shuffled": delta_rs,
                                  "shuffled_minus_MIX50": delta_sm}}
        if origin == 67 and view in {"W4", "W5", "W6"}:
            gates.append({"gate": f"E67_{view}_real_harm_vs_MIX50_max_0.003", "delta": delta_rm,
                          "limit": -0.003, "pass": bool(delta_rm >= -0.003 - GATE_TOLERANCE)})
        if (origin == 67 and view == "H12_E67") or (origin == 75 and view == "W6"):
            gates.extend([
                {"gate": f"E{origin}_{view}_real_gain_vs_MIX50_min_0.0005", "delta": delta_rm,
                 "limit": 0.0005, "pass": bool(delta_rm >= 0.0005 - GATE_TOLERANCE)},
                {"gate": f"E{origin}_{view}_real_gain_vs_shuffled_min_0", "delta": delta_rs,
                 "limit": 0.0, "pass": bool(delta_rs >= -GATE_TOLERANCE)},
            ])
        for group_name, gmask in groups:
            use = mask & gmask
            n = int(use.sum())
            pos = int(y[use].sum())
            neg = n - pos
            if n >= 200 and pos >= 30 and neg >= 30:
                real = _metric(y[use], policies["moment_real"][use])
                mix = _metric(y[use], policies["MIX50"][use])
                df = real["macro_f1"] - mix["macro_f1"]
                dr = real["positive_recall"] - mix["positive_recall"]
                gates.append({"gate": f"E{origin}_{view}_{group_name}_guard", "delta_macro_f1": df,
                              "delta_positive_recall": dr, "support": n, "positive": pos, "negative": neg,
                              "pass": bool(df >= -0.01 - GATE_TOLERANCE and dr >= -0.02 - GATE_TOLERANCE)})
                subgroups.append({"view": view, "group": group_name, "status": "SUPPORTED", "n": n,
                                  "positive": pos, "negative": neg, "real_macro_f1": real["macro_f1"],
                                  "mix50_macro_f1": mix["macro_f1"], "delta_macro_f1": df,
                                  "real_positive_recall": real["positive_recall"],
                                  "mix50_positive_recall": mix["positive_recall"], "delta_positive_recall": dr})
            else:
                subgroups.append({"view": view, "group": group_name, "status": "UNKNOWN_UNSUPPORTED",
                                  "n": n, "positive": pos, "negative": neg})
        for cycle in sorted(np.unique(cycles[mask])):
            one = mask & (cycles == cycle)
            mr = _metric(y[one], policies["moment_real"][one])
            mm = _metric(y[one], policies["MIX50"][one])
            per_cycle.append({"view": view, "cycle": int(cycle), "n": int(one.sum()),
                              "real_macro_f1": mr["macro_f1"], "mix50_macro_f1": mm["macro_f1"],
                              "delta": mr["macro_f1"] - mm["macro_f1"]})
    return {"rows": len(y), "views": views, "subgroups": subgroups, "per_cycle": per_cycle}, gates


def _query_targets(root: Path, rows_needed: set[int]) -> dict[int, tuple[str, int, int]]:
    result: dict[int, tuple[str, int, int]] = {}
    with (root / TRAIN_REL).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        id_i, cycle_i, y_i = header.index("record_id"), header.index("cycle_code"), header.index("artifact_signal")
        for row_idx, row in enumerate(reader):
            if row_idx in rows_needed:
                result[row_idx] = (row[id_i], int(row[cycle_i]), int(row[y_i]))
    if set(result) != rows_needed:
        raise VerificationError("Could not locate every temporal Train query row")
    return result


def verify_scored_results(root: Path, run_root: Path, freeze_data: Mapping[str, Any], core: Mapping[str, Any]) -> dict[str, Any]:
    root, run_root = Path(root), Path(run_root)
    if (core.get("card_id") != CARD_ID or core.get("freeze_signature") != freeze_data.get("freeze_signature")
            or core.get("query_labels_opened") is not True or core.get("test_or_sample_access") is not False
            or core.get("qualified_candidate") is not False or core.get("champion_change") is not False):
        raise VerificationError("Scored result identity/status/source boundary mismatch")
    ready = _read_json(run_root / "PREDICTIONS_READY.json")
    needed: set[int] = set()
    bundles: dict[int, dict[str, np.ndarray]] = {}
    for origin in (67, 75):
        bundle_path = root / ready["predictions"][f"E{origin}"]["path"]
        keys = ["score_rows", "score_record_ids", "score_cycles", "present_mask", "entry_friction", "text_load_group",
                "prob_pP", "prob_a02", "prob_bag", "prob_q_real", "prob_q_shuffled", "prob_candidate_real",
                "prob_candidate_shuffled", "prob_mix50"]
        bundle = _named_npz(bundle_path, keys)
        bundles[origin] = bundle
        needed.update(bundle["score_rows"].astype(int).tolist())
    target_rows = _query_targets(root, needed)
    verified_origins: dict[str, Any] = {}
    gates_all: list[dict[str, Any]] = []
    for origin in (67, 75):
        bundle = bundles[origin]
        rows = bundle["score_rows"].astype(np.int64)
        ids = np.asarray([target_rows[int(i)][0] for i in rows], dtype=str)
        cycles = np.asarray([target_rows[int(i)][1] for i in rows], dtype=np.int64)
        y = np.asarray([target_rows[int(i)][2] for i in rows], dtype=np.int8)
        if (not np.array_equal(ids, bundle["score_record_ids"].astype(str))
                or not np.array_equal(cycles, bundle["score_cycles"].astype(np.int64))
                or not np.isin(y, [0, 1]).all()):
            raise VerificationError(f"E{origin} scored Train query rows do not match frozen IDs/cycles")
        recomputed, gates = _recompute_origin(origin, bundle, y)
        saved = core.get("origins", {}).get(f"E{origin}")
        if saved != {**recomputed, "query_labels_opened": True}:
            raise VerificationError(f"E{origin} metrics/subgroups/gates differ from independent recomputation")
        verified_origins[f"E{origin}"] = {"query_rows": len(rows), "query_rows_sha256": _array_hash(rows),
                                          "target_sha256": _array_hash(y), "views": list(recomputed["views"])}
        gates_all.extend(gates)
    if gates_all != core.get("gate_checks"):
        raise VerificationError("Gate ledger differs from independent metric reconstruction")
    all_pass = all(gate.get("pass") is True for gate in gates_all)
    if core.get("all_gates_pass") is not all_pass:
        raise VerificationError("Overall gate verdict differs from independent gate reconstruction")
    clean_core = dict(core)
    clean_core.pop("independent_verification_sha256", None)
    return {
        "schema_version": 1, "status": "PASS", "card_id": CARD_ID,
        "results_core_sha256": _json_hash(clean_core), "origins": verified_origins,
        "gates": gates_all, "all_gates_pass": all_pass,
        "query_labels_read": True, "test_or_sample_access": False,
        "checks": ["query-only Train target reads", "ID/cycle alignment", "all views and policies",
                   "subgroup support and guards", "gate thresholds and aggregate verdict"],
    }
