"""Independent, zero-fit verification of CLASSIFICATION_NEXT_V1 bundles and gates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score

from src.models.research_next_contract import file_sha256, read_bundle

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/raw/classification/Train.csv"
REPORT = ROOT / "reports/optimization_v3/classification_next_v1"
RUNS = ROOT / "experiments/runs/optimization_v3_classification_next_v1"
FREEZE = ROOT / "experiments/optimization_v3/classification_next_v1/freeze.json"
PATTERN = re.compile(r"[A-Za-z0-9]+")
WINDOWS = {"W1": [(55,56,59)], "W2": [(59,60,63)], "W3": [(63,64,67)],
           "W4": [(67,68,71)], "W5": [(71,72,75)], "W6": [(75,76,79)]}
C19_VIEWS = {**WINDOWS, "primary_fullsix": sum((WINDOWS[name] for name in WINDOWS), []),
    "legacy_primary_W1_W4": WINDOWS["W1"] + WINDOWS["W4"], "late_W5_W6": WINDOWS["W5"] + WINDOWS["W6"],
    "H12_55": [(55,56,67)], "H12_67": [(67,68,79)], "H12_pooled": [(55,56,67),(67,68,79)]}
TTR_VIEWS = {"W1": WINDOWS["W1"], "W4": WINDOWS["W4"],
    "primary_two_origin": WINDOWS["W1"] + WINDOWS["W4"], "H12_55": [(55,56,67)],
    "H12_67": [(67,68,79)], "H12_pooled": [(55,56,67),(67,68,79)]}
TTR_MATCHED_METRICS = ("macro_f1", "positive_recall", "roc_auc", "average_precision", "log_loss",
                       "brier", "predicted_positive_rate", "mean_probability")


def sha(path: Path) -> str:
    return file_sha256(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def independent_metric(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0 or not np.isfinite(p).all():
        raise AssertionError("Metric vectors are misaligned or invalid")
    pred = p >= .5
    tp = int(np.sum((y == 1) & pred)); fn = int(np.sum((y == 1) & ~pred))
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    return {"n": int(len(y)), "class_0_n": int(np.sum(y == 0)), "class_1_n": int(np.sum(y == 1)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "positive_recall": float(tp / (tp + fn)) if tp + fn else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[0,1])), "brier": float(brier_score_loss(y, p)),
        "true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
        "predicted_positive_rate": float(pred.mean()), "mean_probability": float(p.mean()), "threshold": .5}


def token_count_unique(value) -> tuple[int, int]:
    if value is None or pd.isna(value):
        tokens = []
    else:
        tokens = PATTERN.findall(str(value).lower())
    return len(tokens), len(set(tokens))


def quartile_bins(frame: pd.DataFrame, cutoff: int) -> np.ndarray:
    values = pd.to_numeric(frame["text_load"], errors="coerce").to_numpy(dtype=np.float64)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    fit_values = values[(cycles <= cutoff) & np.isfinite(values)]
    bins = np.full(len(frame), -1, dtype=np.int64)
    if len(fit_values):
        edges = np.unique(np.quantile(fit_values, [.25,.50,.75]))
        valid = np.isfinite(values)
        bins[valid] = np.searchsorted(edges, values[valid], side="right")
    return bins


def compare_tree(actual, saved, path="$"):
    mismatches = []
    if isinstance(actual, dict) and isinstance(saved, dict):
        if set(actual) != set(saved):
            return [{"path": path, "expected_keys": sorted(actual), "saved_keys": sorted(saved)}]
        for key in actual:
            mismatches.extend(compare_tree(actual[key], saved[key], f"{path}.{key}"))
    elif isinstance(actual, list) and isinstance(saved, list):
        if len(actual) != len(saved): return [{"path": path, "expected_len": len(actual), "saved_len": len(saved)}]
        for index, (left,right) in enumerate(zip(actual,saved)):
            mismatches.extend(compare_tree(left,right,f"{path}[{index}]"))
    elif actual is None or saved is None:
        if actual is not saved: mismatches.append({"path":path,"expected":actual,"saved":saved})
    elif isinstance(actual, (int,bool,str)) or isinstance(saved,(int,bool,str)):
        if actual != saved: mismatches.append({"path":path,"expected":actual,"saved":saved})
    elif not np.isclose(float(actual), float(saved), rtol=0, atol=1e-12):
        mismatches.append({"path":path,"expected":actual,"saved":saved})
    return mismatches


def _load_outputs(task: str):
    frame = pd.read_csv(TRAIN)
    target = pd.to_numeric(frame.artifact_signal, errors="raise").to_numpy(dtype=np.int8)
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    freeze = load_json(FREEZE)
    outputs = {}
    for cutoff in (55,59,63,67,71,75):
        if task == "C19" and cutoff in (55,67):
            source = ROOT / f"experiments/runs/optimization_v3_batch1/C19/C19_fit{cutoff}.npz"
            sidecar = load_json(source.with_suffix(".json"))
            if sidecar.get("npz_sha256") != sha(source): raise AssertionError(f"C19 historical cache hash mismatch {cutoff}")
            with np.load(source,allow_pickle=False) as z: arrays={name:z[name].copy() for name in z.files}
            rows = arrays["score_rows"].astype(np.int64)
            expected = np.flatnonzero((cycles>cutoff)&(cycles<=cutoff+12))
            np.testing.assert_array_equal(rows,expected)
            np.testing.assert_array_equal(arrays["score_record_ids"].astype(str),frame.record_id.astype(str).to_numpy()[rows])
            pred = arrays["prediction"].astype(np.float64)
            base = arrays["baseline_prediction"].astype(np.float64)
            missing = arrays["control__stream_missing"].astype(bool)
            raw = arrays["raw_candidate"].astype(np.float64)
            cb = arrays["control__cb_probability"].astype(np.float64)
            reconstructed=np.where(missing,cb,.75*base+.25*raw)
            np.testing.assert_array_equal(reconstructed,pred)
        else:
            arm=task if task!="TTR" else None
            if task=="C19":
                if cutoff not in (59,63,71,75): continue
                folder=RUNS/"C19_CONTINUATION"/f"cutoff_{cutoff}"
                receipt=load_json(folder/"receipt.json")
                bundle=read_bundle(folder,receipt["identity"],task="classification")
                arrays=bundle.arrays
                if receipt["identity"].get("freeze_signature")!=freeze["freeze_signature"] or receipt["identity"].get("candidate_kind")!="C19":
                    raise AssertionError(f"C19 continuation bundle identity mismatch at {cutoff}")
                rows=arrays["score_rows"].astype(np.int64)
                expected=np.flatnonzero((cycles>cutoff)&(cycles<=cutoff+4))
                np.testing.assert_array_equal(rows,expected)
                np.testing.assert_array_equal(arrays["score_record_ids"].astype(str),frame.record_id.astype(str).to_numpy()[rows])
                pred=arrays["prediction"].astype(np.float64);base=arrays["baseline_prediction"].astype(np.float64)
                missing=arrays["control_stream_missing"].astype(bool);raw=arrays["raw_candidate"].astype(np.float64)
                cb=arrays["control_cb_probability"].astype(np.float64)
                reconstructed=np.where(missing,cb,.75*base+.25*raw)
                np.testing.assert_array_equal(reconstructed,pred)
            else:
                continue
        outputs[cutoff]={"rows":rows,"cycle_code":cycles[rows],"prediction":pred,
            "baseline_prediction":base,"origin":np.full(len(rows),cutoff,dtype=np.int16)}
    return frame,target,outputs


def _evaluate(frame,target,outputs,definitions):
    metric_rows={};group_rows=[]
    bins={cutoff:quartile_bins(frame,cutoff) for cutoff in outputs}
    cycles=pd.to_numeric(frame.cycle_code,errors="raise").to_numpy(dtype=np.int64)
    for view,blocks in definitions.items():
        segments=[]
        for cutoff,lo,hi in blocks:
            row=outputs[cutoff];take=(row["cycle_code"]>=lo)&(row["cycle_code"]<=hi)
            segments.append((cutoff,row["rows"][take],row["prediction"][take],row["baseline_prediction"][take]))
        rows=np.concatenate([item[1] for item in segments]); assert len(np.unique(rows))==len(rows)
        pred=np.concatenate([item[2] for item in segments]);base=np.concatenate([item[3] for item in segments]); y=target[rows]
        c=independent_metric(y,pred);b=independent_metric(y,base)
        metric_rows[view]={"n":len(rows),"candidate":c,"sparse25":b,"gain_macro_f1":c["macro_f1"]-b["macro_f1"]}
        qlabels=np.concatenate([bins[origin][row] for origin,row,_,_ in segments])
        origins=np.concatenate([np.full(len(row),origin,dtype=np.int16) for origin,row,_,_ in segments])
        for label in sorted(set(qlabels.tolist())):
            take=qlabels==label;gy=y[take];n=int(take.sum());n0=int(np.sum(gy==0));n1=int(np.sum(gy==1))
            cm=independent_metric(gy,pred[take]);bm=independent_metric(gy,base[take])
            f1_harm=bm["macro_f1"]-cm["macro_f1"]
            recall_harm=(bm["positive_recall"]-cm["positive_recall"] if bm["positive_recall"] is not None else None)
            supported=n>=200 and n0>=30 and n1>=30
            group_rows.append({"view":view,"field":"fit_defined_text_load_quartile",
                "group":"missing" if label<0 else f"Q{min(int(label)+1,4)}",
                "n":n,"class_0_n":n0,"class_1_n":n1,"supported":supported,
                "candidate_macro_f1":cm["macro_f1"],"sparse25_macro_f1":bm["macro_f1"],"f1_harm":f1_harm,
                "candidate_positive_recall":cm["positive_recall"],"sparse25_positive_recall":bm["positive_recall"],
                "positive_recall_harm":recall_harm,"fit_origins":sorted(set(origins[take].astype(int).tolist())),
                "fails_0_01_f1":bool(supported and f1_harm>0.01),
                "fails_0_03_recall":bool(supported and recall_harm is not None and recall_harm>0.03),
                "fails_0_02_recall":bool(supported and recall_harm is not None and recall_harm>0.02)})
    return metric_rows,group_rows


def _c19_gate(metrics, groups):
    gains={name:float(metrics[name]["gain_macro_f1"]) for name in WINDOWS}
    supported=[row for row in groups if row["supported"]]
    checks={
        "primary_gain_min_0_0005":metrics["primary_fullsix"]["gain_macro_f1"]>=0.0005,
        "H12_pooled_gain_min_0_002":metrics["H12_pooled"]["gain_macro_f1"]>=0.002,
        "H12_origin67_gain_min_0_001":metrics["H12_67"]["gain_macro_f1"]>=0.001,
        "late_W5W6_harm_max_0_0005":metrics["late_W5_W6"]["gain_macro_f1"]>=-0.0005,
        "W6_harm_max_0_001":gains["W6"]>=-0.001,
        "at_least_4_of_6_windows_nonnegative":sum(value>=0 for value in gains.values())>=4,
        "supported_text_load_groups_F1_harm_max_0_01":not any(row["f1_harm"]>0.01 for row in supported),
        "supported_text_load_groups_recall_harm_max_0_03":not any(row["positive_recall_harm"] is not None and row["positive_recall_harm"]>0.03 for row in supported),
    }
    return {"status":"PASS_ALL_FROZEN_GATES" if all(checks.values()) else "NO_GO_GATE_FAILED",
            "checks":checks,"window_gains":gains}


def _ttr_matched(metrics_by_arm):
    matched={}
    for view in TTR_VIEWS:
        count=metrics_by_arm["COUNT"][view]["candidate"]
        both=metrics_by_arm["BOTH"][view]["candidate"]
        matched[view]={metric:(None if count[metric] is None or both[metric] is None else float(both[metric]-count[metric]))
                       for metric in TTR_MATCHED_METRICS}
    return matched


def _ttr_gate(metrics,matched,groups):
    supported=[row for row in groups if row["supported"]]
    checks={
        "primary_gain_vs_sparse25_min_0_0005":metrics["primary_two_origin"]["gain_macro_f1"]>=0.0005,
        "primary_gain_vs_COUNT_min_0_0002":matched["primary_two_origin"]["macro_f1"]>=0.0002,
        "H12_each_and_pooled_harm_max_0_001":all(metrics[name]["gain_macro_f1"]>=-0.001 for name in ("H12_55","H12_67","H12_pooled")),
        "W1_W4_harm_max_0_005":all(metrics[name]["gain_macro_f1"]>=-0.005 for name in ("W1","W4")),
        "supported_groups_F1_harm_max_0_01":not any(row["f1_harm"]>0.01 for row in supported),
        "supported_groups_recall_harm_max_0_02":not any(row["positive_recall_harm"] is not None and row["positive_recall_harm"]>0.02 for row in supported),
    }
    return {"status":"PASS_DIAGNOSTIC_GATE" if all(checks.values()) else "FAIL_DIAGNOSTIC_GATE","checks":checks}


def _matched_group_rows(groups):
    keyed={(row["arm"],row["view"],row["group"]):row for row in groups}
    rows=[]
    for view,group in sorted((view,group) for arm,view,group in keyed if arm=="COUNT"):
        count=keyed[("COUNT",view,group)];both=keyed[("BOTH",view,group)]
        for field in ("n","class_0_n","class_1_n","supported","fit_origins"):
            if count[field]!=both[field]:raise AssertionError(f"Matched group support differs: {view}/{group}/{field}")
        recall_delta=(None if count["candidate_positive_recall"] is None or both["candidate_positive_recall"] is None
                      else float(both["candidate_positive_recall"]-count["candidate_positive_recall"]))
        rows.append({"view":view,"field":"fit_defined_text_load_quartile","group":group,
            "n":count["n"],"class_0_n":count["class_0_n"],"class_1_n":count["class_1_n"],
            "supported":count["supported"],"fit_origins":count["fit_origins"],
            "count_macro_f1":count["candidate_macro_f1"],"both_macro_f1":both["candidate_macro_f1"],
            "delta_macro_f1_both_minus_count":float(both["candidate_macro_f1"]-count["candidate_macro_f1"]),
            "count_positive_recall":count["candidate_positive_recall"],"both_positive_recall":both["candidate_positive_recall"],
            "delta_positive_recall_both_minus_count":recall_delta})
    return rows


def _compare_csv_rows(path, expected_rows, key_fields):
    saved_rows=list(csv.DictReader(path.open("r",encoding="utf-8",newline="")))
    key=lambda row:tuple(str(row[field]) for field in key_fields)
    saved={key(row):row for row in saved_rows}
    expected={key(row):row for row in expected_rows}
    mismatches=[]
    if len(saved)!=len(saved_rows) or set(saved)!=set(expected):
        mismatches.append({"path":str(path),"expected_keys":len(expected),"saved_keys":len(saved),
                           "missing":sorted(set(expected)-set(saved))[:10],"extra":sorted(set(saved)-set(expected))[:10]})
    for row_key,expected_row in expected.items():
        if row_key not in saved:continue
        saved_row=saved[row_key]
        for field,actual in expected_row.items():
            value=saved_row.get(field,"")
            try:
                if isinstance(actual,bool):
                    equal=value.lower()==str(actual).lower()
                elif isinstance(actual,list):
                    equal=json.loads(value)==actual
                elif actual is None:
                    equal=value==""
                elif isinstance(actual,(int,float,np.integer,np.floating)):
                    equal=np.isclose(float(value),float(actual),rtol=0,atol=1e-12)
                else:
                    equal=value==str(actual)
            except (TypeError,ValueError,json.JSONDecodeError):
                equal=False
            if not equal:mismatches.append({"key":row_key,"field":field,"expected":actual,"saved":value})
    return mismatches


def verify_c19() -> dict:
    result_path=REPORT/"C19_CONTINUATION"/"results.json"
    result=load_json(result_path)
    frame,target,outputs=_load_outputs("C19")
    metrics,groups=_evaluate(frame,target,outputs,C19_VIEWS)
    metric_mismatches=compare_tree(metrics,result["views"],"views")
    gate=_c19_gate(metrics,groups)
    saved_gate={key:result["gate"].get(key) for key in gate}
    gate_mismatches=compare_tree(gate,saved_gate,"gate")
    saved_groups=list(csv.DictReader((REPORT/"C19_CONTINUATION"/"groups.csv").open("r",encoding="utf-8",newline="")))
    group_key=lambda r:(r["view"],r["group"])
    saved={group_key(row):row for row in saved_groups}
    group_mismatches=[]
    for row in groups:
        if group_key(row) not in saved: group_mismatches.append({"missing_group":group_key(row)});continue
        old=saved[group_key(row)]
        for key in ("n","class_0_n","class_1_n","supported","fit_origins"):
            expected=str(row[key])
            if key=="fit_origins": expected="["+", ".join(map(str,row[key]))+"]"
            if old[key].lower()!=expected.lower():group_mismatches.append({"group":group_key(row),"field":key,"expected":expected,"saved":old[key]})
        for key in ("f1_harm","positive_recall_harm"):
            actual=None if row[key] is None else float(row[key]);stored=None if old[key]=="" else float(old[key])
            if actual is not None and (stored is None or not np.isclose(actual,stored,rtol=0,atol=1e-12)):
                group_mismatches.append({"group":group_key(row),"field":key,"expected":actual,"saved":stored})
    passed=not metric_mismatches and not group_mismatches and not gate_mismatches
    qa={"status":"PASS" if passed else "FAIL","lane":"C19_CONTINUATION","metric_mismatch_count":len(metric_mismatches),
        "gate_mismatch_count":len(gate_mismatches),"gate_recomputed_independently":True,
        "independently_recomputed_gate_status":gate["status"],"group_mismatch_count":len(group_mismatches),"metrics_recomputed_independently":True,
        "historical_and_new_prediction_routes_reconstructed":True,"score_row_id_cycle_boundaries_verified":True,
        "Test_labels_read":False,"new_estimator_fits":0,"metric_mismatches":metric_mismatches[:40],
        "gate_mismatches":gate_mismatches[:40],"group_mismatches":group_mismatches[:40],"results_sha256":sha(result_path),
        "official_train_sha256":sha(TRAIN)}
    dest=REPORT/"C19_CONTINUATION"/"independent_verification.json"
    write_json_once(dest,qa)
    return qa


def verify_ttr() -> dict:
    result_path=REPORT/"TREE_TTR_PILOT"/"results.json";result=load_json(result_path)
    frame=pd.read_csv(TRAIN)
    target=pd.to_numeric(frame.artifact_signal,errors="raise").to_numpy(dtype=np.int8)
    cycles=pd.to_numeric(frame.cycle_code,errors="raise").to_numpy(dtype=np.int64)
    outputs={"COUNT":{},"BOTH":{}};state_hashes={}
    for arm in ("COUNT","BOTH"):
        for cutoff in (55,67):
            folder=RUNS/"TREE_TTR_PILOT"/arm/f"cutoff_{cutoff}"
            receipt=load_json(folder/"receipt.json");bundle=read_bundle(folder,receipt["identity"],task="classification")
            a=bundle.arrays;rows=a["score_rows"].astype(np.int64)
            np.testing.assert_array_equal(rows,np.flatnonzero((cycles>cutoff)&(cycles<=cutoff+12)))
            np.testing.assert_array_equal(a["score_record_ids"].astype(str),frame.record_id.astype(str).to_numpy()[rows])
            counts=[];unique=[]
            for text in frame.iloc[rows].signal_stream.tolist():
                count,uniq=token_count_unique(text);counts.append(count);unique.append(uniq)
            np.testing.assert_array_equal(a["token_count"],np.asarray(counts,dtype=np.int32))
            np.testing.assert_array_equal(a["token_unique"],np.asarray(unique,dtype=np.int32))
            c2b=a["control_c2b_probability"].astype(np.float64);cb=a["control_cb_probability"].astype(np.float64)
            missing=a["control_stream_missing"].astype(bool);raw=a["raw_candidate"].astype(np.float64)
            pred=np.where(missing,cb,.75*c2b+.25*raw)
            np.testing.assert_array_equal(pred,a["prediction"])
            model_state=receipt["metadata"]["payload_meta"]["model_state"]
            if model_state["arm"]!=arm or model_state["undefined_ttr_policy"]!="LightGBM native NaN handling; no imputation or extra indicator":
                raise AssertionError(f"TTR model state policy mismatch: {arm}/{cutoff}")
            state_hashes[f"{arm}_{cutoff}"]=model_state["base_sparse_matrix_sha256"]
            outputs[arm][cutoff]={"rows":rows,"cycle_code":cycles[rows],"prediction":a["prediction"],
                "baseline_prediction":a["baseline_prediction"],"origin":np.full(len(rows),cutoff,dtype=np.int16)}
    if any(state_hashes[f"COUNT_{o}"]!=state_hashes[f"BOTH_{o}"] for o in (55,67)):
        raise AssertionError("TTR changed original structured/TF-IDF/missing sparse blocks between matched arms")
    recomputed={};groups=[]
    for arm in ("COUNT","BOTH"):
        recomputed[arm],group_rows=_evaluate(frame,target,outputs[arm],TTR_VIEWS)
        groups += [{**row,"arm":arm} for row in group_rows]
    metric_mismatches=compare_tree(recomputed,result["metrics_by_arm"],"metrics_by_arm")
    matched=_ttr_matched(recomputed)
    metric_mismatches += compare_tree(matched,result["matched_gain_BOTH_minus_COUNT"],"matched_gain")
    candidate_groups=[row for row in groups if row["arm"]=="BOTH"]
    gate=_ttr_gate(recomputed["BOTH"],matched,candidate_groups)
    saved_gate={key:result["gate"].get(key) for key in gate}
    gate_mismatches=compare_tree(gate,saved_gate,"gate")
    matched_groups=_matched_group_rows(groups)
    matched_group_mismatches=_compare_csv_rows(REPORT/"TREE_TTR_PILOT"/"matched_groups.csv",matched_groups,
        ("view","field","group"))
    all_group_mismatches=_compare_csv_rows(REPORT/"TREE_TTR_PILOT"/"groups.csv",groups,("arm","view","field","group"))
    passed=not metric_mismatches and not gate_mismatches and not matched_group_mismatches and not all_group_mismatches
    qa={"status":"PASS" if passed else "FAIL","lane":"TREE_TTR_PILOT","metric_mismatch_count":len(metric_mismatches),
        "gate_mismatch_count":len(gate_mismatches),"independently_recomputed_gate_status":gate["status"],
        "matched_group_mismatch_count":len(matched_group_mismatches),"all_group_mismatch_count":len(all_group_mismatches),
        "token_counts_and_unique_counts_recomputed_from_official_Train":True,
        "count_and_both_original_sparse_blocks_hash_equal":True,
        "route_reconstructed_independently":True,"Test_labels_read":False,"new_estimator_fits":0,
        "metric_mismatches":metric_mismatches[:40],"gate_mismatches":gate_mismatches[:40],
        "matched_group_mismatches":matched_group_mismatches[:40],"all_group_mismatches":all_group_mismatches[:40],
        "base_matrix_sha256_by_arm_origin":state_hashes,
        "results_sha256":sha(result_path),"official_train_sha256":sha(TRAIN)}
    write_json_once(REPORT/"TREE_TTR_PILOT"/"independent_verification.json",qa)
    return qa


def write_json_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8")!=encoded:raise RuntimeError(f"Immutable verifier output differs: {path}")
    else:
        with path.open("x",encoding="utf-8",newline="\n") as stream:stream.write(encoded)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument("--c19",action="store_true");group.add_argument("--ttr",action="store_true")
    args=parser.parse_args();result=verify_c19() if args.c19 else verify_ttr()
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False));return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
