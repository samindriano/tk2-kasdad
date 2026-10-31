"""Fresh winning-model refit from raw CSVs, with no ignored cache dependency."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from scripts.prepare_initial_submissions import _load_inputs
from src.models.regression_research_lgb_text import build_lgb_text
from src.models.regression_research_text import build_text_model
from src.models.optimization_v2_regression import SparseHybridRegressor
from src.models.round1_evaluation import array_digest

def replay(output,receipt):
    output=output.resolve();output.relative_to((ROOT/'experiments/runs/optimization_v2_replay').resolve())
    assert not output.exists() and not receipt.exists()
    manifest=json.loads((ROOT/'reports/optimization_v2/kaggle/v2_reg_sparse_lgb25.json').read_text())
    for p,h in manifest['source_code_sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,p
    train,test,sample,hashes=_load_inputs('regression');assert hashes==manifest['input_sha256']
    assert not {'artifact_signal','threat_rank'}.intersection(test.columns)
    models=[build_lgb_text('base'),build_text_model('hybrid_ridge'),SparseHybridRegressor('lgbm_sparse_l1')]
    parts=[];states=[];cycles=test.cycle_code.to_numpy();start=time.perf_counter()
    for model,c,expected in zip(models,manifest['recipe']['components'],manifest['component_audits']):
        fields=list(model.input_feature_names)
        with threadpool_limits(limits=2):
            model.fit(train[fields],train.threat_rank.to_numpy());state=model.audit_state();p=model.predict(test[fields])
        assert state==model.audit_state()==expected['state'],'Fresh replay differs from accepted full-Train fit state'
        rank=np.empty_like(p)
        for cycle in np.unique(cycles):
            mask=cycles==cycle;rank[mask]=100*rankdata(p[mask],method='average')/mask.sum()
        w=c['rank_weight'];parts.append(c['weight']*np.clip((1-w)*p+w*rank,0,100));states.append(state)
        print('FRESH FIT',c['model'],flush=True)
    pred=np.clip(np.sum(parts,axis=0),0,100)
    assert array_digest(pred)==manifest['prediction_sha256'],'Fresh predictions must be bit-identical'
    assert sample.record_id.equals(test.record_id)and len(pred)==29421 and np.isfinite(pred).all()
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'record_id':test.record_id,'threat_rank':pred}).to_csv(output,index=False,float_format='%.17g')
    result={'status':'PASS','at_utc':datetime.now(timezone.utc).isoformat(),'fresh_full_train_fits':3,
            'train_rows':len(train),'test_rows':len(test),'ignored_cache_or_prior_CSV_read':False,
            'model_binaries_read_or_written':False,'test_labels_used':False,'kaggle_API_called':False,
            'input_sha256':hashes,'all_three_fullTrain_states_equal_accepted_manifest':True,
            'prediction_sha256':array_digest(pred),'prediction_bit_identical_to_accepted_manifest':True,
            'output_csv_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'accepted_csv_sha256':manifest['csv_sha256'],'seconds':time.perf_counter()-start,
            'replay_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x',encoding='utf-8',newline='\n')as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS fresh raw-data replay, bit-identical predictions; seconds',round(result['seconds'],1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();replay(a.output,a.receipt)
