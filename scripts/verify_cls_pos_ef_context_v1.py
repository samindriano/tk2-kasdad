"""Independent target-free output and metric arithmetic verifier for POS x EF."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts import cls_pos_ef_context_v1 as pilot


def _independent_shuffle(ef: Sequence[int], cycles: Sequence[int], present: Sequence[bool]) -> tuple[np.ndarray,int]:
    value=np.asarray(ef,dtype=np.int8).reshape(-1)
    cyc=np.asarray(cycles,dtype=np.int64).reshape(-1)
    avail=np.asarray(present,dtype=bool).reshape(-1)
    if not (len(value)==len(cyc)==len(avail)) or not np.isin(value,(0,1)).all():
        raise ValueError("Independent shuffle inputs are invalid")
    shuffled=value.copy(); rng=np.random.default_rng(42)
    for cycle in sorted(set(cyc.tolist())):
        for is_present in (False,True):
            idx=np.flatnonzero((cyc==cycle)&(avail==is_present))
            if len(idx)>1: shuffled[idx]=value[idx][rng.permutation(len(idx))]
    moved=int(np.count_nonzero((shuffled!=value)&avail))
    return shuffled,moved


def _independent_q(ef: Sequence[int], mu: float, scale: float) -> np.ndarray:
    raw=np.asarray(ef,dtype=np.int8).reshape(-1)
    if not np.isin(raw,(0,1)).all() or not (0<mu<1 and scale>0):
        raise ValueError("Independent context state invalid")
    return (raw.astype(np.float64)-mu)/scale


def _independent_interaction(x: Any, q: Sequence[float]):
    from scipy import sparse
    matrix=sparse.csr_matrix(x,dtype=np.float64)
    context=np.asarray(q,dtype=np.float64).reshape(-1)
    if matrix.shape[0]!=len(context) or not np.isfinite(matrix.data).all():
        raise ValueError("Independent interaction inputs do not align")
    out=matrix.multiply(context[:,None]).tocsr(); out.sum_duplicates(); out.eliminate_zeros(); out.sort_indices()
    return out


def _independent_objective(gamma: Sequence[float], x: Any, z: Sequence[float], y: Sequence[int]):
    from scipy import sparse
    from scipy.special import expit
    matrix=sparse.csr_matrix(x,dtype=np.float64)
    coef=np.asarray(gamma,dtype=np.float64).reshape(-1)
    offset=np.asarray(z,dtype=np.float64).reshape(-1)
    target=np.asarray(y,dtype=np.int8).reshape(-1).astype(np.float64)
    if matrix.shape!=(len(target),len(coef)) or len(offset)!=len(target):
        raise ValueError("Independent objective inputs do not align")
    logits=offset+np.asarray(matrix@coef,dtype=np.float64).reshape(-1)
    losses=np.logaddexp(0.0,logits)-target*logits
    objective=float(losses.sum()+0.5*np.dot(coef,coef))
    gradient=np.asarray(matrix.T@(expit(logits)-target),dtype=np.float64).reshape(-1)+coef
    return objective,gradient


def _independent_threshold(probability: Sequence[float], present: Sequence[bool]) -> np.ndarray:
    p=np.asarray(probability,dtype=np.float64).reshape(-1)
    avail=np.asarray(present,dtype=bool).reshape(-1)
    threshold=np.full(len(p),0.53230976129477,dtype=np.float64)
    threshold[avail]=0.50
    return (p>=threshold).astype(np.int8)


def _independent_route(p_a02: Sequence[float], p_pos: Sequence[float], present: Sequence[bool]) -> np.ndarray:
    p_a=np.asarray(p_a02,dtype=np.float64).reshape(-1).copy()
    p_pos=np.asarray(p_pos,dtype=np.float64).reshape(-1)
    avail=np.asarray(present,dtype=bool).reshape(-1)
    p_a[avail]=0.5*p_a[avail]+0.5*p_pos[avail]
    return p_a


def verify_prelabel_outputs(repository: Path, *, write: bool=True) -> dict[str,Any]:
    """Reconstruct source, optimizer objective, and predictions without query labels."""
    from scipy.special import expit
    from src.models import cls_pos_ef_context_v1 as core
    root=Path(repository).resolve(); freeze=pilot.verify_freeze(root)
    ready=pilot._verified_ready(freeze)
    frame=pilot.read_features(root/"data/raw/classification/Train.csv")
    parent={origin:pilot.load_parent_origin(root,origin,frame) for origin in (67,75)}
    fit_checks={}
    for fit_id in freeze["fit_ids"]:
        origin=67 if "E67" in fit_id else 75
        arm="REAL" if fit_id.endswith("REAL") else "SHUFFLED"
        p=parent[origin]; fit_present=p["fit_present"]
        fit_rows=p["fit_rows"][fit_present]; fit_ids=p["fit_ids"][fit_present]
        ef=p["fit_ef"][fit_present]
        fit_shuffled,moved=_independent_shuffle(p["fit_ef"],p["fit_cycles"],p["fit_present"])
        if sha256_array(fit_shuffled)!=sha256_array(p["fit_ef_shuffled"]) or moved!=p["fit_moved_present"]:
            raise RuntimeError(f"Independent fit shuffle reconstruction failed: {fit_id}")
        ef_use=ef if arm=="REAL" else fit_shuffled[fit_present]
        st=p["context_state"]; selected=p["fit_ef"][p["fit_present"]]
        mu=float(np.mean(selected)); scale=float(np.sqrt(mu*(1-mu)))
        if mu!=st["mu"] or scale!=st["scale"]:
            raise RuntimeError(f"Independent fit-only context state differs: {fit_id}")
        q=_independent_q(ef_use,mu,scale)
        x=_independent_interaction(p["x_fit"][fit_present],q)
        y=pilot._selected_targets(root/"data/raw/classification/Train.csv",fit_rows,fit_ids)
        fit=pilot.verify_fit_artifact(freeze,fit_id); gamma=fit["gamma"]; rec=fit["receipt"]
        objective,gradient=_independent_objective(gamma,x,p["z_pos_fit"][fit_present],y)
        if (sha256_array(y)!=rec["fit_population"]["fit_target_sha256"]
            or sha256_array(fit_rows)!=rec["fit_population"]["fit_rows_sha256"]
            or sha256_array(fit_ids)!=rec["fit_population"]["fit_ids_sha256"]
            or sha256_csr(x)!=rec["fit_population"]["interaction_sha256"]
            or abs(objective-rec["optimizer"]["loss_final"])>1e-8
            or abs(float(np.max(np.abs(gradient)))-rec["optimizer"]["gradient_inf_norm"])>1e-9
            or rec["optimizer"]["success"] is not True
            or objective>rec["optimizer"]["loss_gamma0"]+1e-8):
            raise RuntimeError(f"Independent objective/receipt check failed: {fit_id}")
        fit_checks[fit_id]={"receipt_sha256":fit["receipt_sha256"],"artifact_sha256":rec["artifact_sha256"],
            "rows":int(len(fit_rows)),"present_fit_rows":int(len(y)),"fit_target_sha256":sha256_array(y),
            "loss_reconstructed":objective,"gradient_inf_norm_reconstructed":float(np.max(np.abs(gradient))),
            "query_labels_read":False}
    outputs={}
    for origin in (67,75):
        suffix=f"E{origin}"; item=ready["predictions"][suffix]
        bundle=pilot._load_saved_predictions(pilot.AUDIT_ROOT/item["path"])
        p=parent[origin]; present=p["present"]; st=p["context_state"]
        expected={"origin":np.asarray([origin],dtype=np.int64),"row_positions":p["score_rows"].astype(np.int64),
            "record_ids":p["score_ids"].astype(str),"cycles":p["cycles"].astype(np.int64),"present":present.astype(bool),
            "ef1":p["ef"].astype(np.int8),"text_load_group":p["text_load_group"].astype(str),
            "p_a02":p["p_a02"].astype(np.float64),"p_pos_original":p["p_pos_original"].astype(np.float64),
            "p_champion":p["p_champion"].astype(np.float64),"label_champion":p["label_champion"].astype(np.int8)}
        deltas={}
        for arm in ("REAL","SHUFFLED"):
            fit_id=f"CLS_POS_EF_CONTEXT_V1_E{origin}_{arm}"; gamma=pilot.verify_fit_artifact(freeze,fit_id)["gamma"]
            if arm=="REAL": ef_query=p["ef"]
            else:
                shuffled,moved=_independent_shuffle(p["ef"],p["cycles"],present)
                if sha256_array(shuffled)!=sha256_array(p["query_ef_shuffled"]) or moved!=p["query_moved_present"]:
                    raise RuntimeError(f"Independent query shuffle reconstruction failed: E{origin}")
                ef_query=shuffled
            q=_independent_q(ef_query,p["context_state"]); q=q[present]
            x=_independent_interaction(p["x_score"][present],q)
            delta=np.asarray(x@gamma,dtype=np.float64).reshape(-1)
            p_context=np.asarray(expit(p["z_pos_score"][present]+delta),dtype=np.float64).reshape(-1)
            full_pos=p["p_pos_original"].copy(); full_pos[present]=p_context
            routed=_independent_route(p["p_a02"],full_pos,present)
            labels=_independent_threshold(routed,present)
            full_delta=np.zeros(len(present),dtype=np.float64); full_delta[present]=delta
            expected[f"p_{arm.lower()}"]=routed
            expected[f"label_{arm.lower()}"]=labels
            expected[f"delta_{arm.lower()}"]=full_delta
            deltas[arm.lower()]={"n":int(len(delta)),"l2_norm":float(np.linalg.norm(delta)),
                "rms":float(np.sqrt(np.mean(delta**2))),"max_abs":float(np.max(np.abs(delta))) if len(delta) else 0.0}
        if any(not np.array_equal(bundle[k],v) for k,v in expected.items()):
            bad=[k for k,v in expected.items() if not np.array_equal(bundle[k],v)]
            raise RuntimeError(f"Independent prediction replay failed for {suffix}: {bad}")
        missing=~present
        if not (np.array_equal(bundle["p_real"][missing].view(np.uint64),p["p_a02"][missing].view(np.uint64))
                and np.array_equal(bundle["p_shuffled"][missing].view(np.uint64),p["p_a02"][missing].view(np.uint64))
                and np.array_equal(bundle["label_real"][missing],p["label_champion"][missing])
                and np.array_equal(bundle["label_shuffled"][missing],p["label_champion"][missing])):
            raise RuntimeError(f"Missing route is not bit-exact A02 for {suffix}")
        outputs[suffix]={"bundle_sha256":pilot.sha256_file(pilot.AUDIT_ROOT/item["path"]),
            "rows":item["rows"],"row_ids_sha256":pilot.sha256_array(bundle["record_ids"]),
            "missing_fallback_exact":True,"query_labels_read":False,"delta_logit":deltas}
    body={"schema_version":1,"candidate_id":"CLS_POS_EF_CONTEXT_V1","status":"PASS_TARGET_FREE_PRELABEL",
        "freeze_signature":freeze["freeze_signature"],"source_commit":freeze["source_commit"],
        "predictions_ready_sha256":pilot.sha256_file(pilot.READY_PATH),
        "prediction_output_sha256s":{k:v["sha256"] for k,v in ready["predictions"].items()},
        "fit_receipts":fit_checks,"predictions":outputs,"query_labels_read":False,
        "query_label_hash_present":False,"test_or_sample_access":False,"optimizer_refits":0,
        "preprocessing_fits":0,"numerical_reconstruction":"independent sparse logistic objective, gradient, context shuffle, and predictions"}
    body["verification_signature"]=pilot.canonical_sha256(body)
    if write: pilot.write_once(pilot.PRELABEL_PATH,body)
    return body


def verify_synthetic_prelabel(prediction_path: Path, prelabel_path: Path) -> dict[str,Any]:
    """Independent target-free check in the synthetic lifecycle."""
    body=pilot.read_json(prelabel_path); signature=body.pop("verification_signature",None)
    if signature!=pilot.canonical_sha256(body): raise RuntimeError("Synthetic prelabel signature failed")
    if (body.get("status")!="PASS_TARGET_FREE_PRELABEL" or body.get("query_labels_read") is not False
        or body.get("query_label_hash_present") is not False
        or body.get("prediction_sha256")!=pilot.sha256_file(prediction_path)):
        raise RuntimeError("Synthetic prelabel source/boundary binding failed")
    with np.load(prediction_path,allow_pickle=False) as archive:
        expected={"p_a02","p_pos_context","p_champion","p_candidate","present","delta"}
        if set(archive.files)!=expected: raise RuntimeError("Synthetic bundle has an unexpected member")
        p_a=np.asarray(archive["p_a02"],dtype=np.float64); p_pos=np.asarray(archive["p_pos_context"],dtype=np.float64)
        champion=np.asarray(archive["p_champion"],dtype=np.float64); candidate=np.asarray(archive["p_candidate"],dtype=np.float64)
        present=np.asarray(archive["present"],dtype=bool)
    replay=p_a.copy(); replay[present]=0.5*p_a[present]+0.5*p_pos[present]
    if not np.array_equal(replay,candidate): raise RuntimeError("Synthetic candidate blend did not replay")
    if not np.isfinite(champion).all() or not np.isfinite(candidate).all():
        raise RuntimeError("Synthetic prelabel predictions are not finite")
    if not np.array_equal(candidate[~present].view(np.uint64),p_a[~present].view(np.uint64)):
        raise RuntimeError("Synthetic missing fallback is not bit-exact")
    return {"status":"PASS_SYNTHETIC_PRELABEL","query_labels_read":False,
            "prediction_sha256":pilot.sha256_file(prediction_path),"missing_exact":True}


def _metrics(y: Sequence[int], p: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    probability = np.asarray(p, dtype=np.float64).reshape(-1)
    predicted = np.asarray(labels, dtype=np.int8).reshape(-1)
    if not (len(target) == len(probability) == len(predicted)) or len(target) == 0:
        raise ValueError("Independent metric vectors do not align")
    if not np.isin(target, (0,1)).all() or not np.isin(predicted,(0,1)).all():
        raise ValueError("Independent metric targets/predictions must be binary")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Independent metric probabilities invalid")
    tn=int(np.count_nonzero((target==0)&(predicted==0)))
    fp=int(np.count_nonzero((target==0)&(predicted==1)))
    fn=int(np.count_nonzero((target==1)&(predicted==0)))
    tp=int(np.count_nonzero((target==1)&(predicted==1)))
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    precision0=tn/(tn+fn) if tn+fn else 0.0
    recall0=tn/(tn+fp) if tn+fp else 0.0
    f1_1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    f1_0=2*precision0*recall0/(precision0+recall0) if precision0+recall0 else 0.0
    clipped=np.clip(probability,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
    logloss=float(-np.mean(target*np.log(clipped)+(1-target)*np.log1p(-clipped)))
    has_both=np.unique(target).size==2
    return {"n":int(len(target)),"positive":int(target.sum()),"negative":int(len(target)-target.sum()),
        "macro_f1":float((f1_0+f1_1)/2),"positive_precision":float(precision),
        "positive_recall":float(recall),
        "roc_auc":float(roc_auc_score(target,probability)) if has_both else None,
        "average_precision":float(average_precision_score(target,probability)) if has_both else None,
        "log_loss":logloss,"brier":float(np.mean((probability-target)**2)),
        "confusion_matrix_labels_0_1":[[tn,fp],[fn,tp]]}


def independent_metric_comparison(
    y: Sequence[int], p_champion: Sequence[float], p_candidate: Sequence[float],
    yhat_champion: Sequence[int], yhat_candidate: Sequence[int],
) -> dict[str, Any]:
    champion=_metrics(y,p_champion,yhat_champion)
    candidate=_metrics(y,p_candidate,yhat_candidate)
    ca=candidate; ch=champion
    return {"support":{"n":ca["n"],"positive":ca["positive"],"negative":ca["negative"]},
        "champion":ch,"candidate":ca,
        "gain_macro_f1":ca["macro_f1"]-ch["macro_f1"],
        "gain_positive_recall":ca["positive_recall"]-ch["positive_recall"],
        "gain_positive_precision":ca["positive_precision"]-ch["positive_precision"],
        "gain_roc_auc":None if ca["roc_auc"] is None or ch["roc_auc"] is None else ca["roc_auc"]-ch["roc_auc"],
        "gain_average_precision":None if ca["average_precision"] is None or ch["average_precision"] is None else ca["average_precision"]-ch["average_precision"],
        "gain_log_loss":ch["log_loss"]-ca["log_loss"],"gain_brier":ch["brier"]-ca["brier"]}


def compare_metric_outputs(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    differences=[]
    def walk(a: Any,b: Any,path: str="") -> None:
        if isinstance(a,Mapping) and isinstance(b,Mapping):
            if set(a)!=set(b):
                differences.append({"path":path,"expected_keys":sorted(a),"observed_keys":sorted(b)})
                return
            for key in sorted(a): walk(a[key],b[key],f"{path}/{key}")
        elif isinstance(a,(int,np.integer)) and isinstance(b,(int,np.integer)):
            if int(a)!=int(b): differences.append({"path":path,"expected":int(a),"observed":int(b)})
        elif a is None or b is None:
            if a is not b: differences.append({"path":path,"expected":a,"observed":b})
        elif isinstance(a,(float,int,np.floating,np.integer)) and isinstance(b,(float,int,np.floating,np.integer)):
            delta=abs(float(a)-float(b))
            if not np.isfinite(delta) or delta>5e-12:
                differences.append({"path":path,"expected":float(a),"observed":float(b),"abs_diff":delta})
        else:
            if a!=b: differences.append({"path":path,"expected":str(a),"observed":str(b)})
    walk(expected,observed)
    return {"status":"PASS" if not differences else "FAIL","max_abs_difference":max((d.get("abs_diff",0.0) for d in differences),default=0.0),
            "difference_count":len(differences),"differences":differences[:20]}


def build_metric_verification(views: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not checks or any(check.get("status")!="PASS" for check in checks):
        raise RuntimeError("Independent metric checks are incomplete or failed")
    body={"status":"PASS_INDEPENDENT_METRIC_RECALC","independent_check_count":len(checks),
        "checks":[dict(c) for c in checks],"views_sha256":pilot.canonical_sha256(views),
        "query_labels_reopened":False,"method":"Independent confusion/F1/precision/recall/logloss/Brier arithmetic plus sklearn ROC-AUC/AP"}
    body["verification_signature"]=pilot.canonical_sha256(body)
    return body


def _verify_gate_reconstruction(score: Mapping[str, Any]) -> dict[str, Any]:
    views=score["views"]
    primary=views["UNIQUE_PRIMARY"]["UNIQUE_PRIMARY"]["real_vs_champion"]
    e67_h12=views["E67"]["H12_E67"]["real_vs_champion"]["gain_macro_f1"]
    e75_w6=views["E75"]["W6"]["real_vs_champion"]["gain_macro_f1"]
    real_shuffled_primary=views["UNIQUE_PRIMARY"]["UNIQUE_PRIMARY"]["real_vs_shuffled"]["gain_macro_f1"]
    real_shuffled_e75=views["E75"]["W6"]["real_vs_shuffled"]["gain_macro_f1"]
    legacy_ok=True; exploratory_ok=True
    for view_name, view in views.items():
        if view_name=="UNIQUE_PRIMARY": candidates=[view["UNIQUE_PRIMARY"]]
        else: candidates=[v for k,v in view.items() if isinstance(v,Mapping) and "supported_subgroups" in v]
        for item in candidates:
            for group in item.get("supported_subgroups",[]):
                metrics=group.get("metrics")
                if not group.get("support",{}).get("supported") or metrics is None: continue
                f1=metrics["gain_macro_f1"]; rec=metrics["gain_positive_recall"]
                legacy_ok &= f1>=-0.01 and rec>=-0.02
                exploratory_ok &= f1>=-0.02 and rec>=-0.03
    scientific=(primary["gain_macro_f1"]>=0.0005 and e67_h12>=0 and e75_w6>=0
        and real_shuffled_primary>=0.0002 and real_shuffled_e75>=0 and legacy_ok)
    exploratory_gains=primary["gain_macro_f1"]>0 or (e75_w6>=0.0005 and primary["gain_macro_f1"]>=-0.0005)
    prelabel=pilot.verify_committed_prelabel()
    missing_exact=all(v.get("missing_fallback_exact") is True for v in prelabel["predictions"].values())
    metric_qa=(score.get("metric_verification",{}).get("status")=="PASS_INDEPENDENT_METRIC_RECALC"
        and score.get("metric_verification",{}).get("independent_check_count",0)>0)
    qa_pass=prelabel.get("status")=="PASS_TARGET_FREE_PRELABEL" and metric_qa
    scientific=(scientific and missing_exact and qa_pass)
    exploratory=exploratory_gains and exploratory_ok and missing_exact and qa_pass
    recorded=score["gates"]
    checks={"scientific_qualified":bool(scientific),"exploratory_eligible":bool(exploratory),
        "exploratory_gains_ok":bool(exploratory_gains),"legacy_subgroup_guard_ok":bool(legacy_ok),
        "exploratory_subgroup_guard_ok":bool(exploratory_ok),"missing_fallback_exact":bool(missing_exact),
        "independent_qa_pass":bool(qa_pass)}
    if any(recorded.get(k)!=v for k,v in checks.items()):
        raise RuntimeError("Recorded qualification gates differ from independent gate reconstruction")
    return {"status":"PASS_GATE_RECONSTRUCTION","reconstructed":checks}


def verify_score() -> dict[str, Any]:
    score_path=pilot.SCORE_PATH; metric_path=pilot.METRIC_VERIFY_PATH
    if not score_path.is_file() or not metric_path.is_file():
        raise RuntimeError("Score or metric verification artifact is missing")
    score=pilot.read_json(score_path); metric=pilot.read_json(metric_path)
    body=dict(score["metric_verification"]); signature=body.pop("verification_signature",None)
    if signature!=pilot.canonical_sha256(body) or body.get("status")!="PASS_INDEPENDENT_METRIC_RECALC":
        raise RuntimeError("Embedded independent metric verification signature failed")
    if body["views_sha256"]!=pilot.canonical_sha256(score["views"]):
        raise RuntimeError("Metric verification is not bound to current score views")
    if metric.get("score_sha256")!=pilot.sha256_file(score_path) or metric.get("status")!="PASS_INDEPENDENT_METRIC_RECALC":
        raise RuntimeError("Independent metric verification report is not bound to SCORE.json")
    report_body=dict(metric); report_signature=report_body.pop("report_signature",None)
    if report_signature!=pilot.canonical_sha256(report_body):
        raise RuntimeError("Independent metric verification report signature failed")
    freeze=pilot.verify_freeze(Path(score["official_repository_root"]) if score.get("official_repository_root") else Path(pilot.AUDIT_ROOT.parent))
    ledger=pilot.events(pilot.RUN_ROOT/"fit_ledger.jsonl")
    opened=[e for e in ledger if e.get("event")=="QUERY_LABELS_OPENED"]
    if len(opened)!=1 or opened[0].get("prelabel_sha256")!=score["prelabel_sha256"]:
        raise RuntimeError("Query-label accounting differs from the single committed prelabel pass")
    if score["prelabel_sha256"]!=pilot.sha256_file(pilot.PRELABEL_PATH):
        raise RuntimeError("Score is not bound to the current committed prelabel receipt")
    gate=_verify_gate_reconstruction(score)
    return {"status":"PASS_INDEPENDENT_METRIC_AND_GATE_VERIFICATION",
        "score_sha256":pilot.sha256_file(score_path),"metric_verification_sha256":pilot.sha256_file(metric_path),
        "independent_metric_check_count":body["independent_check_count"],"query_label_reads":1,
        "test_or_sample_access":False,"freeze_signature":freeze["freeze_signature"],"gate":gate}


def write_metric_report_from_saved_score(score: Mapping[str, Any]) -> dict[str, Any]:
    body=score.get("metric_verification")
    if not isinstance(body,Mapping): raise RuntimeError("Saved SCORE.json lacks embedded metric verification")
    check=dict(body); signature=check.pop("verification_signature",None)
    if signature!=pilot.canonical_sha256(check): raise RuntimeError("Saved metric verification signature failed")
    report={"status":"PASS_INDEPENDENT_METRIC_RECALC","score_sha256":pilot.sha256_file(pilot.SCORE_PATH),**body}
    report["report_signature"]=pilot.canonical_sha256(report)
    pilot.write_once(pilot.METRIC_VERIFY_PATH,report)
    return report


def main(argv: Sequence[str] | None=None) -> int:
    import argparse
    ap=argparse.ArgumentParser(description=__doc__); sub=ap.add_subparsers(dest="cmd",required=True)
    pre=sub.add_parser("prelabel"); pre.add_argument("--official-root",type=Path,required=True)
    sub.add_parser("score")
    args=ap.parse_args(argv)
    result=verify_prelabel_outputs(args.official_root) if args.cmd=="prelabel" else verify_score()
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)); return 0


if __name__=="__main__":
    raise SystemExit(main())


if __name__=="__main__":
    print(json.dumps(verify_score(),sort_keys=True,indent=2,allow_nan=False))
