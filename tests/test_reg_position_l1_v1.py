import joblib
import json
import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import subprocess
import sys
from pathlib import Path

from src.models.reg_position_l1_v1 import unigram_bag, head_design, compose_residual_prediction


def test_formula_cap_bounds_and_exact_missing():
    b=np.array([0.,100.,50.,40.])
    p,d=compose_residual_prediction(b,[100,-100,4,100],[False,False,False,True])
    np.testing.assert_array_equal(d,[5,-5,1,0])
    np.testing.assert_array_equal(p,[5,95,51,40])
    np.testing.assert_array_equal(p[-1:],b[-1:])
    q,_=compose_residual_prediction([99.,1.],[100.,-100.],[False,False])
    np.testing.assert_array_equal(q,[100.,0.])


@pytest.mark.parametrize("b,h,m", [([1.],[np.nan],[False]), ([101.],[0.],[False]),
                                  ([1.],[1.,2.],[False]),([1.],[1.],[0])])
def test_invalid_values_and_mask_rejected(b,h,m):
    with pytest.raises(ValueError):
        compose_residual_prediction(b,h,m)


def test_unigram_transform_is_fit_free(monkeypatch):
    vec=TfidfVectorizer(ngram_range=(1,2)).fit(["aa bb aa", "bb cc"])
    monkeypatch.setattr(vec,"fit",lambda *a,**k:pytest.fail("fit prohibited"))
    monkeypatch.setattr(vec,"fit_transform",lambda *a,**k:pytest.fail("fit_transform prohibited"))
    b=unigram_bag(vec,["aa bb aa", "", "newtoken"])
    assert b.shape==(3,3)
    np.testing.assert_allclose(np.asarray(b.multiply(b).sum(axis=1)).ravel(),[1,0,0])
    x=head_design(b,sparse.csr_matrix((3,9)),np.array([20.,50.,80.]))
    np.testing.assert_allclose(x.toarray()[:,-1],[.2,.5,.8])


def test_feature_join_rejects_bad_rows_and_nan():
    with pytest.raises(ValueError):
        head_design(sparse.csr_matrix((2,1)),sparse.csr_matrix((1,3)),[1,2])
    with pytest.raises(ValueError):
        head_design(sparse.csr_matrix([[np.nan]]),None,[1])


def test_synthetic_l1_estimator_and_unicode_readback(tmp_path):
    from lightgbm import LGBMRegressor
    rng=np.random.default_rng(42)
    b=sparse.csr_matrix(rng.random((80,3)))
    p=sparse.csr_matrix(rng.random((80,9)))
    anchor=np.linspace(20,80,80)
    x=head_design(b,p,anchor)
    residual=np.asarray(p[:,0].toarray()).ravel()*4-2
    model=LGBMRegressor(objective="regression_l1",n_estimators=5,num_leaves=3,
                       min_child_samples=5,n_jobs=2,verbosity=-1,random_state=42)
    model.fit(x,residual)
    path=tmp_path/"model.joblib";joblib.dump(model,path)
    prediction,delta=compose_residual_prediction(anchor,model.predict(x),np.zeros(80,dtype=bool))
    replay,_=compose_residual_prediction(anchor,joblib.load(path).predict(x),np.zeros(80,dtype=bool))
    np.testing.assert_array_equal(prediction,replay)
    np.savez(tmp_path/"arrays.npz",ids=np.array(["長い-id-α", "samuel"]),prediction=prediction,delta=delta)
    with np.load(tmp_path/"arrays.npz",allow_pickle=False) as z:
        assert z["ids"].tolist()==["長い-id-α", "samuel"]


def test_actual_entrypoint_synthetic_lifecycle_verification_and_tamper_rejection(tmp_path):
    runner = Path(__file__).resolve().parents[1] / "scripts" / "reg_position_l1_v1.py"
    result = subprocess.run(
        [sys.executable, str(runner), "synthetic-e2e", "--temp-root", str(tmp_path / "pilot-e2e")],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "PASS_SYNTHETIC_ENTRYPOINT_E2E"
    assert report["fit_starts"] == report["fit_saved"] == report["fit_invocations"] == 6
    assert report["fit_failed"] == 0
    assert report["resumed_from_durable_model_without_refit"] is True
    assert report["source_root_write_guard"] == "PASS"
    assert report["tampered_copy_rejected"] is True
    assert all(report["guard_checks"].values())
    independent = report["synthetic_independent_verification"]
    assert independent["status"] == "PASS_SYNTHETIC_INDEPENDENT_METRICS"
    assert independent["unsupported_groups_remain_unknown"] is True
    assert independent["fit_invocations_reported"] == 6
