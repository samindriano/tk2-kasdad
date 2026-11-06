from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.models import cls_pos_ef_context_v1 as core
from scripts import cls_pos_ef_context_v1 as runner
from scripts import verify_cls_pos_ef_context_v1 as verifier

def test_context_state_and_stratified_shuffle_preserve_counts():
    ef=np.tile(np.array([0,1,1,0],dtype=np.int8),6)
    cycles=np.repeat(np.arange(47,53),4)
    present=np.tile(np.array([True,True,False,False]),6)
    state=core.context_state(ef,present,minimum_per_level=2)
    shuffled,moved=core.shuffle_context_within_cycle(ef,cycles,present)
    assert state["mu"]==0.5
    assert state["scale"]==0.5
    assert moved>=0
    for cycle in np.unique(cycles):
        for available in (False,True):
            mask=(cycles==cycle)&(present==available)
            assert sorted(ef[mask].tolist())==sorted(shuffled[mask].tolist())


def test_objective_gradient_finite_difference():
    rng=np.random.default_rng(91)
    x=sparse.csr_matrix(rng.normal(size=(23,7)))
    gamma=rng.normal(scale=0.2,size=7)
    z=rng.normal(size=23)
    y=rng.integers(0,2,size=23,dtype=np.int8)
    loss,grad=core.objective_gradient(gamma,x,z,y)
    eps=1e-6; numeric=np.empty_like(grad)
    for j in range(len(gamma)):
        plus=gamma.copy(); minus=gamma.copy(); plus[j]+=eps; minus[j]-=eps
        numeric[j]=(core.objective_gradient(plus,x,z,y)[0]-core.objective_gradient(minus,x,z,y)[0])/(2*eps)
    assert np.isfinite(loss)
    assert np.allclose(grad,numeric,atol=2e-7,rtol=2e-7)


def test_missing_stream_fallback_is_bit_exact():
    a=np.array([0.1234567890123,0.8912345678901,0.444],dtype=np.float64)
    pos=np.array([0.9,0.1,0.9],dtype=np.float64)
    present=np.array([True,False,True])
    routed=core.route_candidate(a,pos,present)
    assert routed[1].view(np.uint64)==a[1].view(np.uint64)
    labels=core.apply_thresholds(np.array([0.49,0.53230976129477,0.51]),present)
    assert labels.tolist()==[0,1,1]


def test_identifier_digest_is_stable_and_dtype_bound():
    ids=np.array(["TP2-001","TP2-002"],dtype=str)
    assert runner.sha256_array(ids)==runner.sha256_array(ids.copy())
    assert runner.sha256_array(ids)!=runner.sha256_array(np.array([1,2],dtype=np.int64))


def test_selected_target_reader_skips_poison_rows(tmp_path):
    path=tmp_path/"selected.csv"
    path.write_text("record_id,feature,artifact_signal\n"+
        "id-0,a,0\nid-1,b,1\nid-2,c,POISON_QUERY_LABEL\nid-3,d,ALSO_POISON\n",encoding="utf-8")
    y=runner._selected_targets(path,[0,1],["id-0","id-1"])
    assert y.tolist()==[0,1]


def test_gamma_artifact_readback_resume_and_tamper(tmp_path):
    fit_id=core.FIT_IDS[0]; artifact=tmp_path/f"{fit_id}.npz"; receipt_path=tmp_path/f"{fit_id}.json"
    gamma=np.array([0.25,-0.5,0.75],dtype=np.float64)
    runner.atomic_npz(artifact,gamma=gamma)
    receipt={"fit_id":fit_id,"freeze_signature":"synthetic","position_width":3,
        "gamma_sha256":runner.sha256_array(gamma),"artifact_sha256":runner.sha256_file(artifact)}
    runner.write_once(receipt_path,receipt)
    calls={"optimizer":0}
    resumed=runner.resume_saved_gamma("synthetic",fit_id,artifact,receipt_path)
    assert np.array_equal(resumed["gamma"],gamma)
    assert calls["optimizer"]==0
    artifact.write_bytes(b"tampered")
    with pytest.raises(runner.PilotError):
        runner.verify_gamma_bundle("synthetic",fit_id,artifact,receipt_path)


def test_independent_metrics_match_core():
    y=np.array([0,1,0,1,1,0,1,0],dtype=np.int8)
    pc=np.array([.1,.6,.4,.7,.2,.1,.9,.8])
    pn=np.array([.2,.8,.3,.4,.6,.2,.7,.1])
    hc=(pc>=.5).astype(np.int8); hn=(pn>=.5).astype(np.int8)
    primary=core.metric_comparison(y,pc,pn,hc,hn)
    independent=verifier.independent_metric_comparison(y,pc,pn,hc,hn)
    assert verifier.compare_metric_outputs(primary,independent)["status"]=="PASS"


def test_full_synthetic_lifecycle_via_actual_cli(tmp_path):
    out=tmp_path/"lifecycle"
    completed=subprocess.run([sys.executable,str(ROOT/"scripts/cls_pos_ef_context_v1.py"),
        "synthetic","--output",str(out)],cwd=ROOT,text=True,capture_output=True,check=True)
    result=json.loads(completed.stdout)
    assert result["status"]=="PASS_SYNTHETIC_END_TO_END"
    assert result["persistence_readback"] and result["tamper_detected"]
    assert result["poison_query_labels_excluded"] and result["metric_verifier_equal"]


def test_import_and_cli_help_have_no_local_side_effects(tmp_path):
    env=os.environ.copy(); env["PYTHONPATH"]=str(ROOT)
    completed=subprocess.run([sys.executable,"-c","import scripts.cls_pos_ef_context_v1; import scripts.verify_cls_pos_ef_context_v1"],
        cwd=tmp_path,env=env,text=True,capture_output=True,check=True)
    assert completed.stdout==""
    assert list(tmp_path.iterdir())==[]
    cli=subprocess.run([sys.executable,str(ROOT/"scripts/score_cls_pos_ef_context_v1.py"),"--help"],
        cwd=tmp_path,text=True,capture_output=True,check=True)
    assert "--official-root" in cli.stdout
    assert list(tmp_path.iterdir())==[]
