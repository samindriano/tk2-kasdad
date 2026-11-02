"""Synthetic-only sparse MLP API/determinism check; never opens official data."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'experiments/optimization_v3/classification_mlp_v1/protocol.json'
OUT=ROOT/'reports/optimization_v3/classification_mlp_v1/SYNTHETIC_FEASIBILITY.json'


def run():
    protocol=json.loads(PROTOCOL.read_text())
    params=protocol['model']['parameters'].copy()
    params['hidden_layer_sizes']=tuple(params['hidden_layer_sizes'])
    rng=np.random.RandomState(42)
    x=sparse.random(512,256,density=.03,format='csr',random_state=rng,data_rvs=rng.standard_normal)
    y=(rng.uniform(size=512)>.55).astype(int)
    outputs=[];warning_records=[];losses=[]
    for repeat in range(2):
        model=MLPClassifier(**params)
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=2):
            warnings.simplefilter('always')
            model.fit(x,y)
        for warning in caught:
            assert issubclass(warning.category,ConvergenceWarning),str(warning.message)
            warning_records.append({'repeat':repeat,'category':warning.category.__name__,'message':str(warning.message)})
        p=model.predict_proba(x)[:,1]
        assert model.n_iter_==30 and len(model.loss_curve_)==30
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        assert np.isfinite(model.loss_curve_).all() and model.loss_curve_[-1]<model.loss_curve_[0]
        outputs.append(p);losses.append([float(model.loss_curve_[0]),float(model.loss_curve_[-1])])
    np.testing.assert_array_equal(outputs[0],outputs[1])
    return dict(status='PASS_SYNTHETIC_API_ONLY_NOT_OFFICIAL_PIPELINE_QA',synthetic_estimator_fits=2,official_estimator_fits=0,
                shape=list(x.shape),nnz=x.nnz,CSR_input=True,exact_seed_reproducibility=True,epochs=30,initial_final_losses=losses,
                warnings=warning_records,protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                sklearn_version=importlib.metadata.version('scikit-learn'),official_files_opened=0,
                pending='Luna must still prove actual C2_B fit-only design and exact real payloadbuilder persistence/resume/all-view verification before officialfits.')


if __name__=='__main__':
    result=run();OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
