"""Independent replay/metrics verifier for CLS POS frequency-ridge evaluation.

This verifier does not import the evaluator's metric helpers. Its result mode
reads only committed prediction arrays plus official Train identity/targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from src.models.optimization_v3_positional_text_pilot_v1 import sha256_csr

FREEZE = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_FREEZE_R5.json"
FREEZE_PARENT = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_FREEZE_R4.json"
R4_ATTEMPT = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_ATTEMPT_R4_PRELABEL.json"
PRED = "experiments/runs/optimization_v3_cls_pos_frequency_ridge_eval_v1/predictions_r1.npz"
PRED_MANIFEST = "reports/optimization_v3/cls_pos_frequency_ridge_v1/PREDICTIONS_MANIFEST_R1.json"
RESULTS = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_RESULTS_R1.json"
TRAIN = "data/raw/classification/Train.csv"
TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EPS = 1e-15
TOL = 2e-12
VIEWS = {
    "E67_H12": (67, 68, 79), "E67_W4": (67, 68, 71), "E67_W5": (67, 72, 75),
    "E67_W6": (67, 76, 79), "E75_W6": (75, 76, 79),
}
GROUPS = ("availability", "entry_friction", "text_load")
OPS_THRESHOLD = 0.53230976129477
PILOT_ROOT = "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
POS_BAG_ROOT = "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
A02 = "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
A02_KEYS = tuple(
    f"E{o}_{key}" for o in (67,75)
    for key in ("rows","ids","cycles","pA","pP","missing","MIX50",
                "group_availability","group_entry_friction","group_text_load")
)
POS_KEYS = ("score_rows","score_record_ids","score_cycles","present_mask",
            "prob_bag","prob_offset_real","prob_missing_hgb")
PILOT_KEYS = ("score_rows","score_record_ids","score_cycles","present_mask",
             "prob_bag","prob_missing_hgb","prob_route_bag")


class VerifyError(RuntimeError):
    pass


def p(rel: str) -> Path:
    return ROOT / rel


def sha_file(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(file: Path) -> dict[str, Any]:
    data = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VerifyError(f"Expected JSON object: {file}")
    return data


def git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode:
        raise VerifyError(proc.stderr.strip())
    return proc.stdout.strip()


def committed(rel: str) -> str:
    blob = git("rev-parse", f"HEAD:{rel}")
    if blob != git("hash-object", rel):
        raise VerifyError(f"Git blob differs from worktree file: {rel}")
    return git("log", "-1", "--format=%H", "--", rel)


def verify_freeze_and_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    committed(FREEZE); committed(PRED_MANIFEST)
    freeze = read_json(p(FREEZE)); manifest = read_json(p(PRED_MANIFEST))
    if freeze.get("status") != "FROZEN_BEFORE_QUERY_TARGET_LOAD":
        raise VerifyError("Source freeze is not pre-label")
    if (freeze.get("freeze_round") != "R5"
            or freeze.get("supersedes_freeze") != FREEZE_PARENT
            or freeze.get("parent_freeze_sha256") != sha_file(p(FREEZE_PARENT))
            or freeze.get("parent_freeze_commit") != committed(FREEZE_PARENT)
            or freeze.get("preserved_failed_attempt_sha256") != sha_file(p(R4_ATTEMPT))
            or freeze.get("preserved_failed_attempt_commit") != committed(R4_ATTEMPT)):
        raise VerifyError("R5 source freeze parent/failed-attempt lineage is invalid")
    parent = read_json(p(FREEZE_PARENT))
    if any(freeze.get(name) != parent.get(name)
           for name in ("windows", "unique_pool", "recipe", "gates", "incident_disclosure", "incident_status")):
        raise VerifyError("R5 changed the scientific recipe, views, gates, or incident disclosure")
    if manifest.get("status") != "PREDICTIONS_FROZEN_BEFORE_QUERY_TARGET_READ":
        raise VerifyError("Predictions manifest is not pre-label")
    if manifest.get("source_freeze_sha256") != sha_file(p(FREEZE)):
        raise VerifyError("Prediction manifest source-freeze hash mismatch")
    if manifest.get("prediction_sha256") != sha_file(p(PRED)):
        raise VerifyError("Prediction archive hash mismatch")
    for rel, expected in freeze.get("source_hashes", {}).items():
        if sha_file(p(rel)) != expected:
            raise VerifyError(f"Evaluator/predecessor source hash changed: {rel}")
    for rel, expected in freeze.get("input_hashes", {}).items():
        if sha_file(p(rel)) != expected:
            raise VerifyError(f"Frozen input hash changed: {rel}")
    if manifest.get("query_target_labels_read") is not False or manifest.get("query_target_members_loaded") != []:
        raise VerifyError("Predictions manifest does not prove zero query-target read")
    resource = manifest.get("resource_snapshot", {})
    samples = resource.get("samples", [])
    if (resource.get("policy_id") != "CLS_EVALUATION_QUERY_REPLAY_R4"
            or resource.get("initial_floor_bytes") != 3 * 1024**3
            or resource.get("incremental_memory_estimate_bytes") != 512 * 1024**2
            or resource.get("projected_free_floor_bytes") != 2 * 1024**3
            or resource.get("samples_are_peak_measurements") is not False
            or not samples or samples[0].get("pass_both_initial_gates") is not True
            or any(item.get("pass") is not True for item in samples[1:])):
        raise VerifyError("R4 resource sample guards are incomplete or failed")
    return freeze, manifest


def load_predictions(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    schema = manifest.get("prediction_schema")
    if not isinstance(schema, list) or any(str(k).endswith("_y") for k in schema):
        raise VerifyError("Prediction schema is malformed or contains target arrays")
    # Request only the frozen explicit schema; target-key rejection precedes indexing.
    if set(schema) & {"E67_y", "E75_y"}:
        raise VerifyError("Target key is forbidden in prediction archive")
    with np.load(p(PRED), allow_pickle=False) as archive:
        if set(schema) != set(archive.files):
            raise VerifyError("Prediction archive member set differs from manifest")
        arrays = {key: np.asarray(archive[key]) for key in schema}
    for key, expected in manifest.get("prediction_array_hashes", {}).items():
        array = arrays[key]
        header = json.dumps({"shape": array.shape, "dtype": str(array.dtype)},
                            sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(header.encode("ascii"))
        if array.dtype.kind in "OUS":
            digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
        if digest.hexdigest() != expected:
            raise VerifyError(f"Independent saved array hash mismatch: {key}")
    return arrays


def strict_npz(file: Path, requested: tuple[str, ...], allowlist: tuple[str, ...]) -> dict[str, np.ndarray]:
    if any(k.endswith("_y") for k in requested) or set(requested) - set(allowlist):
        raise VerifyError("Forbidden or non-allowlisted archive field requested")
    with np.load(file, allow_pickle=False) as archive:
        absent = set(requested)-set(archive.files)
        if absent:
            raise VerifyError(f"Allowlisted NPZ fields missing: {sorted(absent)}")
        return {key: np.asarray(archive[key]) for key in requested}


def array_sha256(value: Any) -> str:
    a=np.asarray(value)
    header=json.dumps({"shape":a.shape,"dtype":str(a.dtype)},sort_keys=True,separators=(",",":"))
    digest=hashlib.sha256(header.encode("ascii"))
    if a.dtype.kind in "OUS":
        digest.update(json.dumps(a.astype(str).tolist(),ensure_ascii=False,separators=(",",":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()


def load_gamma(artifact_rel: str, receipt_rel: str) -> np.ndarray:
    receipt=read_json(p(receipt_rel)); artifact=p(artifact_rel)
    if receipt.get("artifact_sha256")!=sha_file(artifact):
        raise VerifyError(f"Gamma bytes differ from receipt: {artifact_rel}")
    with np.load(artifact,allow_pickle=False) as z:
        if set(z.files)!={"gamma"}:
            raise VerifyError("Gamma unit archive has unexpected fields")
        gamma=np.asarray(z["gamma"],dtype=np.float64)
    if array_sha256(gamma)!=receipt.get("gamma_sha256") or not np.isfinite(gamma).all():
        raise VerifyError(f"Gamma content differs from receipt: {artifact_rel}")
    return gamma


def load_frozen_matrix(manifest: dict[str, Any], key: str) -> sparse.csr_matrix:
    item=manifest["matrices"][key]; file=p(str(item["path"]))
    if sha_file(file)!=item["sha256"]:
        raise VerifyError(f"Frozen matrix byte hash mismatch: {key}")
    matrix=sparse.load_npz(file).tocsr()
    if (list(matrix.shape)!=item["shape"] or str(matrix.dtype)!=item["dtype"]
            or sha256_csr(matrix)!=item["logical_sha256"]):
        raise VerifyError(f"Frozen matrix logical identity mismatch: {key}")
    return matrix


def independently_replay_saved_predictions(arrays: dict[str,np.ndarray]) -> dict[str,Any]:
    a02=strict_npz(p(A02),A02_KEYS,A02_KEYS)
    audit={}
    for origin in (67,75):
        k=f"E{origin}"; pos=strict_npz(
            p(f"{POS_BAG_ROOT}/{k}/predictions_{k}.npz"),POS_KEYS,POS_KEYS)
        pilot=strict_npz(
            p(f"{PILOT_ROOT}/{k}/predictions_{k}.npz"),PILOT_KEYS,PILOT_KEYS)
        manifest=read_json(p(f"{PILOT_ROOT}/MATRIX_MANIFEST_E{origin}.json"))
        base=load_frozen_matrix(manifest,"score_base")
        augmented=load_frozen_matrix(manifest,"score_position")
        if (augmented[:,:base.shape[1]]!=base).nnz:
            raise VerifyError(f"{k} position matrix no longer contains the exact frozen C2_B base")
        position=augmented[:,base.shape[1]:].tocsr()
        bag_id=f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
        bag_unit=joblib.load(p(f"{PILOT_ROOT}/units/{bag_id}.joblib"))
        estimator=bag_unit.get("estimator")
        if estimator is None or list(np.asarray(estimator.classes_).tolist())!=[0,1]:
            raise VerifyError(f"{k} frozen C2_B estimator invalid")
        z0=np.asarray(base@np.asarray(estimator.coef_,dtype=np.float64)[0]).reshape(-1)
        z0=z0+float(np.asarray(estimator.intercept_,dtype=np.float64).reshape(-1)[0])
        bag_p=expit(z0)
        if not np.allclose(bag_p,pilot["prob_bag"],rtol=0.0,atol=2e-12):
            raise VerifyError(f"{k} frozen C2_B probability replay differs")

        missing=a02[f"E{origin}_missing"].astype(bool); present=~missing
        p_a=a02[f"E{origin}_pA"].astype(np.float64)
        fallback=pos["prob_missing_hgb"].astype(np.float64)
        pilot_hgb=pilot["prob_missing_hgb"].astype(np.float64)
        pilot_raw_bag=pilot["prob_bag"].astype(np.float64)
        pilot_routed_bag=pilot["prob_route_bag"].astype(np.float64)
        if not np.array_equal(fallback,pilot_hgb):
            raise VerifyError(f"{k} POS and pilot HGB fallback vectors differ")
        if not np.array_equal(pos["prob_bag"],pilot_routed_bag):
            raise VerifyError(f"{k} POS route and pilot prob_route_bag differ")
        if (not np.array_equal(pilot_routed_bag[missing],pilot_hgb[missing])
                or not np.array_equal(pilot_routed_bag[present],pilot_raw_bag[present])):
            raise VerifyError(f"{k} pilot raw/routed/HGB cache semantics fail")
        gamma_specs={
            "UNIFORM":(
                f"{POS_BAG_ROOT}/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz",
                f"{POS_BAG_ROOT}/receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json"),
            "ALIGNED":(
                f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/units/E{origin}_ALIGNED.npz",
                f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/receipts/E{origin}_ALIGNED.json"),
            "PERMUTED":(
                f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/units/E{origin}_PERMUTED.npz",
                f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/receipts/E{origin}_PERMUTED.json"),
        }
        errors={}
        probabilities={}
        for arm,(art,rec) in gamma_specs.items():
            gamma=load_gamma(art,rec)
            if len(gamma)!=position.shape[1]:
                raise VerifyError(f"{k}/{arm} gamma and score-position width differ")
            logits=z0+np.asarray(position@gamma).reshape(-1)
            value=fallback.copy(); value[present]=expit(logits[present])
            if not np.array_equal(value[missing],fallback[missing]):
                raise VerifyError(f"{k}/{arm} changed HGB missing fallback")
            saved=arrays[f"{k}_pP_{arm}"] if arm!="UNIFORM" else arrays[f"{k}_pP_UNIFORM"]
            errors[arm]=float(np.max(np.abs(value-saved)))
            if errors[arm]>2e-10:
                raise VerifyError(f"{k}/{arm} independent gamma replay differs from saved predictions")
            probabilities[arm]=value
        for key,value in (
            (f"{k}_rows",a02[f"E{origin}_rows"]),
            (f"{k}_ids",a02[f"E{origin}_ids"]),
            (f"{k}_cycles",a02[f"E{origin}_cycles"]),
            (f"{k}_pA",p_a),
            (f"{k}_missing",missing),
        ):
            if not np.array_equal(arrays[key],value):
                raise VerifyError(f"Saved prediction identity/A02 mismatch: {key}")
        for family in GROUPS:
            if not np.array_equal(arrays[f"{k}_group_{family}"],
                                  a02[f"E{origin}_group_{family}"].astype(str)):
                raise VerifyError(f"Saved subgroup values differ from A02 frozen groups: {k}/{family}")
        expected_mix=0.5*probabilities["UNIFORM"]+0.5*p_a
        expected_mix[missing]=p_a[missing]
        if not np.allclose(arrays[f"{k}_MIX50_REFERENCE"],expected_mix,rtol=0.0,atol=2e-15):
            raise VerifyError(f"Saved original MIX50 reference does not replay: {k}")
        audit[k]={"c2b_max_abs_error":float(np.max(np.abs(bag_p-pilot_raw_bag))),
                  "gamma_probability_max_abs_errors":errors,
                  "route_cache_fields_bit_exact":True,
                  "missing_fallback_bit_exact":True,"a02_alignment":"PASS"}
    return {"status":"PASS_INDEPENDENT_PREDICTION_REPLAY","origins":audit}


def calc(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    # Deliberately simple, independently implemented confusion/F1/calibration math.
    y = [int(v) for v in np.asarray(y).tolist()]
    prob = [float(v) for v in np.asarray(prob).tolist()]
    if len(y) != len(prob) or any(v not in (0, 1) for v in y):
        raise VerifyError("Metric inputs do not align or contain non-binary targets")
    pred = [v >= threshold for v in prob]
    tn = sum(1 for a,b in zip(y,pred) if a == 0 and not b)
    fp = sum(1 for a,b in zip(y,pred) if a == 0 and b)
    fn = sum(1 for a,b in zip(y,pred) if a == 1 and not b)
    tp = sum(1 for a,b in zip(y,pred) if a == 1 and b)
    f0d = 2*tn+fp+fn; f1d = 2*tp+fp+fn
    f0 = 2*tn/f0d if f0d else 0.0
    f1 = 2*tp/f1d if f1d else 0.0
    clipped = [min(1-EPS,max(EPS,x)) for x in prob]
    brier = sum((x-a)**2 for a,x in zip(y,prob))/len(y)
    loss = -sum(a*np.log(x)+(1-a)*np.log1p(-x) for a,x in zip(y,clipped))/len(y)
    return {
        "n": len(y), "class_0": y.count(0), "class_1": y.count(1),
        "confusion_matrix": [[tn,fp],[fn,tp]], "macro_f1": (f0+f1)/2,
        "positive_precision": tp/(tp+fp) if tp+fp else 0.0,
        "positive_recall": tp/(tp+fn) if tp+fn else 0.0,
        "brier": brier, "logloss": loss,
    }


def calc_with_missing_cut(y: np.ndarray, pval: np.ndarray, missing: np.ndarray,
                          p_a: np.ndarray, threshold: float) -> dict[str, Any]:
    yv = np.asarray(y, dtype=np.int8); pv = np.asarray(pval, dtype=np.float64)
    mv = np.asarray(missing, dtype=bool); pa = np.asarray(p_a, dtype=np.float64)
    pred = pv >= 0.5
    pred[mv] = pa[mv] >= threshold
    # calibration metrics use unchanged probabilities; only missing hard decisions change.
    base = calc(yv, pv, threshold=2.0)
    tn = int(np.sum((yv == 0) & ~pred)); fp = int(np.sum((yv == 0) & pred))
    fn = int(np.sum((yv == 1) & ~pred)); tp = int(np.sum((yv == 1) & pred))
    f0d=2*tn+fp+fn; f1d=2*tp+fp+fn
    base.update({
        "confusion_matrix": [[tn,fp],[fn,tp]],
        "macro_f1": 0.5*((2*tn/f0d if f0d else 0)+(2*tp/f1d if f1d else 0)),
        "positive_precision": tp/(tp+fp) if tp+fp else 0.0,
        "positive_recall": tp/(tp+fn) if tp+fn else 0.0,
        "present_threshold": 0.5, "missing_threshold": threshold,
    })
    return base


def close(a: Any, b: Any, label: str) -> None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not np.isclose(float(a), float(b), rtol=0.0, atol=TOL):
            raise VerifyError(f"{label}: {a} != {b}")
    elif a != b:
        raise VerifyError(f"{label}: {a!r} != {b!r}")


def compare_metric(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key in ("n", "class_0", "class_1", "confusion_matrix"):
        close(actual[key], expected[key], f"{label}/{key}")
    for key in ("macro_f1", "positive_precision", "positive_recall", "brier", "logloss"):
        close(actual[key], expected[key], f"{label}/{key}")


def verify_results() -> dict[str, Any]:
    _, manifest = verify_freeze_and_manifest()
    arrays = load_predictions(manifest)
    results = read_json(p(RESULTS))
    if results.get("status") != "SCORED_DISCLOSED_DEVELOPMENT_EVALUATION":
        raise VerifyError("Result status does not disclose development evaluation")
    if results.get("incident_disclosure", {}).get("pre_exposure_cannot_be_claimed") is not True:
        raise VerifyError("Result omits the access-order incident limitation")
    if sha_file(p(TRAIN)) != TRAIN_SHA:
        raise VerifyError("Official Train raw-data hash mismatch before independent target read")
    frame = pd.read_csv(p(TRAIN), usecols=["record_id","artifact_signal"],
                        dtype={"record_id":"string"})
    ids = frame.record_id.astype(str).to_numpy()
    labels: dict[int,np.ndarray] = {}
    for origin in (67,75):
        rows = arrays[f"E{origin}_rows"].astype(np.int64)
        if not np.array_equal(ids[rows], arrays[f"E{origin}_ids"].astype(str)):
            raise VerifyError(f"E{origin} label row IDs differ from saved prediction IDs")
        y = pd.to_numeric(frame.iloc[rows]["artifact_signal"],errors="raise").to_numpy(dtype=np.int8)
        if not np.isin(y,[0,1]).all():
            raise VerifyError(f"E{origin} target values are not binary")
        labels[origin]=y

    checks = 0
    gain_by_view: dict[str,float] = {}
    subgroup_failures: list[dict[str,Any]] = []
    subgroup_unknown: list[str] = []
    for view_name,(origin,first,last) in VIEWS.items():
        cyc=arrays[f"E{origin}_cycles"].astype(np.int64)
        mask=(cyc>=first)&(cyc<=last); y=labels[origin]
        view=results["views"][view_name]
        present=arrays[f"E{origin}_present"].astype(bool)
        p_a=arrays[f"E{origin}_pA"].astype(np.float64)
        component={
            "ALIGNED": arrays[f"E{origin}_pP_ALIGNED"],
            "PERMUTED": arrays[f"E{origin}_pP_PERMUTED"],
            "UNIFORM": arrays[f"E{origin}_pP_UNIFORM"],
        }
        for arm,prob in component.items():
            compare_metric(view["component_full"][arm],calc(y[mask],prob[mask]),f"{view_name}/component/{arm}")
            compare_metric(view["component_present_only"][arm],
                           calc(y[mask&present],prob[mask&present]),f"{view_name}/present/{arm}")
            checks += 2
        base=arrays[f"E{origin}_MIX50_REFERENCE"]
        aligned=0.5*arrays[f"E{origin}_pP_ALIGNED"]+0.5*p_a
        aligned=aligned.copy(); aligned[~present]=p_a[~present]
        compare_metric(view["competition_full"]["MIX50_ORIGINAL"],calc(y[mask],base[mask]),f"{view_name}/mix/base")
        compare_metric(view["competition_full"]["MIX50_ALIGNED"],calc(y[mask],aligned[mask]),f"{view_name}/mix/aligned")
        compare_metric(view["competition_present_only"]["MIX50_ORIGINAL"],
                       calc(y[mask&present],base[mask&present]),f"{view_name}/mix-present/base")
        compare_metric(view["competition_present_only"]["MIX50_ALIGNED"],
                       calc(y[mask&present],aligned[mask&present]),f"{view_name}/mix-present/aligned")
        gain_by_view[view_name]=calc(y[mask],aligned[mask])["macro_f1"]-calc(y[mask],base[mask])["macro_f1"]
        checks += 4
        op=view["post_selected_missing_threshold_descriptive"]
        compare_metric(op["MIX50_ORIGINAL"],
                       calc_with_missing_cut(y[mask],base[mask],~present[mask],p_a[mask],OPS_THRESHOLD),
                       f"{view_name}/op/base")
        compare_metric(op["MIX50_ALIGNED"],
                       calc_with_missing_cut(y[mask],aligned[mask],~present[mask],p_a[mask],OPS_THRESHOLD),
                       f"{view_name}/op/aligned")
        checks += 2

        for family in GROUPS:
            vals=arrays[f"E{origin}_group_{family}"].astype(str)
            for level,cell in view["subgroups"][family].items():
                gm=mask&(vals==level); gy=y[gm]
                n0=int((gy==0).sum()); n1=int((gy==1).sum())
                support={"n":int(gm.sum()),"class_0":n0,"class_1":n1}
                close(cell["support"],support,f"{view_name}/{family}/{level}/support")
                supported=support["n"]>=200 and n0>=30 and n1>=30
                close(cell["status"],"SUPPORTED" if supported else "UNKNOWN_SUPPORT",
                      f"{view_name}/{family}/{level}/status")
                if supported:
                    bm=calc(gy,base[gm]); cm=calc(gy,aligned[gm])
                    compare_metric(cell["MIX50_ORIGINAL"],bm,f"{view_name}/{family}/{level}/base")
                    compare_metric(cell["MIX50_ALIGNED"],cm,f"{view_name}/{family}/{level}/aligned")
                    close(cell["harm_macro_f1"],bm["macro_f1"]-cm["macro_f1"],"subgroup F1 harm")
                    close(cell["harm_positive_recall"],bm["positive_recall"]-cm["positive_recall"],
                          "subgroup recall harm")
                    harm_f1=bm["macro_f1"]-cm["macro_f1"]
                    harm_recall=bm["positive_recall"]-cm["positive_recall"]
                    if harm_f1>0.01 or harm_recall>0.02:
                        subgroup_failures.append({"group":f"{view_name}/{family}/{level}",
                                                  "harm_macro_f1":harm_f1,
                                                  "harm_positive_recall":harm_recall})
                    checks += 2
                else:
                    subgroup_unknown.append(f"{view_name}/{family}/{level}")

        for cycle,d in view["per_cycle"].items():
            cm=mask&(cyc==int(cycle))
            bm=calc(y[cm],base[cm]); cmx=calc(y[cm],aligned[cm])
            close(d["n"],int(cm.sum()),f"{view_name}/cycle/{cycle}/n")
            close(d["gain_macro_f1"],cmx["macro_f1"]-bm["macro_f1"],f"{view_name}/cycle/{cycle}/gain")
            expected_direction="POSITIVE" if cmx["macro_f1"]>bm["macro_f1"] else (
                "NEGATIVE" if cmx["macro_f1"]<bm["macro_f1"] else "TIE")
            close(d["direction"],expected_direction,f"{view_name}/cycle/{cycle}/direction")
            checks += 1

    # Independently rebuild the disjoint pooled view.
    selected=[]
    for origin,names_for_origin in ((67,("E67_W4","E67_W5")),(75,("E75_W6",))):
        c=arrays[f"E{origin}_cycles"].astype(np.int64)
        mask=np.zeros(len(c),bool)
        for name in names_for_origin:
            _,a,b=VIEWS[name]; mask |= (c>=a)&(c<=b)
        selected.append((origin,mask))
    ypool=np.concatenate([labels[o][m] for o,m in selected])
    ppool={k:np.concatenate([arrays[f"E{o}_{k}"][m] for o,m in selected])
           for k in ("pA","pP_UNIFORM","pP_ALIGNED","pP_PERMUTED","MIX50_REFERENCE","present")}
    pa=ppool["pA"]; present=ppool["present"].astype(bool)
    aligned=0.5*ppool["pP_ALIGNED"]+0.5*pa; aligned[~present]=pa[~present]
    pool=results["views"]["UNIQUE_E67_W4_W5_PLUS_E75_W6"]
    close(pool["n"],20903,"unique pool n")
    for arm,key in (("ALIGNED","pP_ALIGNED"),("PERMUTED","pP_PERMUTED"),("UNIFORM","pP_UNIFORM")):
        compare_metric(pool["component_full"][arm],calc(ypool,ppool[key]),f"pool/component/{arm}")
        compare_metric(pool["component_present_only"][arm],
                       calc(ypool[present],ppool[key][present]),f"pool/component-present/{arm}")
        checks += 2
    for key,prob in (("MIX50_ORIGINAL",ppool["MIX50_REFERENCE"]),("MIX50_ALIGNED",aligned)):
        compare_metric(pool["competition_full"][key],calc(ypool,prob),f"pool/{key}")
        compare_metric(pool["competition_present_only"][key],
                       calc(ypool[present],prob[present]),f"pool-present/{key}")
        checks += 2
    gain_by_view["UNIQUE_E67_W4_W5_PLUS_E75_W6"]=(
        calc(ypool,aligned)["macro_f1"]-calc(ypool,ppool["MIX50_REFERENCE"])["macro_f1"])
    pool_missing=~present
    for key,prob in (("MIX50_ORIGINAL",ppool["MIX50_REFERENCE"]),("MIX50_ALIGNED",aligned)):
        compare_metric(pool["post_selected_missing_threshold_descriptive"][key],
                       calc_with_missing_cut(ypool,prob,pool_missing,pa,OPS_THRESHOLD),
                       f"pool/op/{key}")
        checks += 1
    for family in GROUPS:
        vals=np.concatenate([arrays[f"E{o}_group_{family}"][m].astype(str) for o,m in selected])
        for level,cell in pool["subgroups"][family].items():
            gm=vals==level; gy=ypool[gm]
            support={"n":int(gm.sum()),"class_0":int((gy==0).sum()),"class_1":int((gy==1).sum())}
            close(cell["support"],support,f"pool/{family}/{level}/support")
            supported=support["n"]>=200 and support["class_0"]>=30 and support["class_1"]>=30
            close(cell["status"],"SUPPORTED" if supported else "UNKNOWN_SUPPORT",
                  f"pool/{family}/{level}/status")
            if supported:
                bm=calc(gy,ppool["MIX50_REFERENCE"][gm]); cm=calc(gy,aligned[gm])
                compare_metric(cell["MIX50_ORIGINAL"],bm,f"pool/{family}/{level}/base")
                compare_metric(cell["MIX50_ALIGNED"],cm,f"pool/{family}/{level}/aligned")
                hf=bm["macro_f1"]-cm["macro_f1"]; hr=bm["positive_recall"]-cm["positive_recall"]
                close(cell["harm_macro_f1"],hf,f"pool/{family}/{level}/F1-harm")
                close(cell["harm_positive_recall"],hr,f"pool/{family}/{level}/recall-harm")
                if hf>0.01 or hr>0.02:
                    subgroup_failures.append({"group":f"UNIQUE_E67_W4_W5_PLUS_E75_W6/{family}/{level}",
                                              "harm_macro_f1":hf,"harm_positive_recall":hr})
                checks+=2
            else:
                subgroup_unknown.append(f"UNIQUE_E67_W4_W5_PLUS_E75_W6/{family}/{level}")
    pool_cycles=np.concatenate([arrays[f"E{o}_cycles"][m] for o,m in selected]).astype(np.int64)
    for cycle,d in pool["per_cycle"].items():
        cm=pool_cycles==int(cycle)
        bm=calc(ypool[cm],ppool["MIX50_REFERENCE"][cm]); cmx=calc(ypool[cm],aligned[cm])
        gain=cmx["macro_f1"]-bm["macro_f1"]
        close(d["n"],int(cm.sum()),f"pool/cycle/{cycle}/n")
        close(d["gain_macro_f1"],gain,f"pool/cycle/{cycle}/gain")
        direction="POSITIVE" if gain>0 else "NEGATIVE" if gain<0 else "TIE"
        close(d["direction"],direction,f"pool/cycle/{cycle}/direction")
        checks+=1

    # Confirm gates were computed from the independently reconstructed views.
    mechanism = {}
    for name in ("E67_H12","E75_W6"):
        values=results["views"][name]["component_full"]
        a=values["ALIGNED"]["macro_f1"]
        mechanism[name]=(a-values["PERMUTED"]["macro_f1"]>1e-12
                         and a-values["UNIFORM"]["macro_f1"]>1e-12)
    if results["mechanism_gate"]["pass"] != all(mechanism.values()):
        raise VerifyError("Mechanism gate result does not independently recompute")
    for name in ("E67_H12","E75_W6"):
        m=results["views"][name]["component_full"]
        a=m["ALIGNED"]["macro_f1"]
        close(results["mechanism_gate"]["checks"][name]["aligned_vs_permuted_gain"],
              a-m["PERMUTED"]["macro_f1"],f"mechanism/{name}/permuted gain")
        close(results["mechanism_gate"]["checks"][name]["aligned_vs_uniform_gain"],
              a-m["UNIFORM"]["macro_f1"],f"mechanism/{name}/uniform gain")
        close(results["mechanism_gate"]["checks"][name]["pass"],mechanism[name],
              f"mechanism/{name}/pass")
    cg=results["competition_gate"]
    gains={name:gain_by_view[name] for name in
           ("E67_H12","E75_W6","UNIQUE_E67_W4_W5_PLUS_E75_W6")}
    harms={name:-gain_by_view[name] for name in ("E67_W4","E67_W5","E67_W6")}
    gate_checks={
        "E67_H12_gain_at_least_0.0003":gains["E67_H12"]>=0.0003,
        "E75_W6_gain_nonnegative":gains["E75_W6"]>=0.0,
        "unique_pool_gain_at_least_0.0003":gains["UNIQUE_E67_W4_W5_PLUS_E75_W6"]>=0.0003,
        "each_E67_window_harm_at_most_0.003":all(v<=0.003 for v in harms.values()),
        "supported_subgroups_within_guards":not subgroup_failures,
    }
    if cg.get("checks")!=gate_checks or cg.get("pass")!=all(gate_checks.values()):
        raise VerifyError("Competition gate checks do not independently recompute")
    if set(cg.get("gains",{}))!=set(gains) or set(cg.get("E67_window_harms",{}))!=set(harms):
        raise VerifyError("Competition gate view keys differ")
    for key,value in gains.items():
        close(cg["gains"][key],value,f"competition gain/{key}")
    for key,value in harms.items():
        close(cg["E67_window_harms"][key],value,f"competition harm/{key}")
    actual_failures=cg.get("supported_subgroup_failures",[])
    if [x.get("group") for x in actual_failures] != [x["group"] for x in subgroup_failures]:
        raise VerifyError("Supported subgroup failure groups differ from independent recomputation")
    for actual,expected in zip(actual_failures,subgroup_failures):
        close(actual.get("harm_macro_f1"),expected["harm_macro_f1"],"subgroup failure F1")
        close(actual.get("harm_positive_recall"),expected["harm_positive_recall"],"subgroup failure recall")
    if sorted(cg.get("unsupported_subgroups_unknown",[]))!=sorted(subgroup_unknown):
        raise VerifyError("Unsupported subgroup UNKNOWN ledger differs from independent recomputation")
    if int(results.get("target_rows_read",-1)) != sum(len(y) for y in labels.values()):
        raise VerifyError("Recorded target row count differs from independent read")
    return {"status":"PASS_INDEPENDENT_METRIC_AND_ROW_RECONSTRUCTION",
            "metric_comparisons":checks,"target_rows":sum(len(y) for y in labels.values()),
            "test_or_samplesubmission_accessed":False,
            "incident_disclosure_present":True}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-predictions",action="store_true")
    group.add_argument("--verify-results",action="store_true")
    args=parser.parse_args()
    if args.verify_predictions:
        freeze,manifest=verify_freeze_and_manifest(); arrays=load_predictions(manifest)
        replay=independently_replay_saved_predictions(arrays)
        result={"status":"PASS_PRELABEL_PREDICTION_HASH_AND_SCHEMA",
                "prediction_fields":len(arrays),"source_freeze_sha256":sha_file(p(FREEZE)),
                "target_labels_loaded":False,"independent_prediction_replay":replay}
    else:
        result=verify_results()
    print(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
