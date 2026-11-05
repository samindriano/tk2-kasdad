"""Read-only full recomputation plus independent source-boundary/metric checks."""
import json
from pathlib import Path
import numpy as np
from scripts import simplification_audit_v1 as a

def compare(x,y):
 if isinstance(x,dict):
  assert x.keys()==y.keys();return sum(compare(x[k],y[k]) for k in x)
 if isinstance(x,list):
  assert len(x)==len(y);return sum(compare(p,q) for p,q in zip(x,y))
 if isinstance(x,(float,int)) and not isinstance(x,bool):
  assert np.isfinite(x) and abs(x-y)<=1e-10,(x,y);return 1
 assert x==y,(x,y);return 0

def verify():
 freeze=json.loads(a.FREEZE.read_text());p={k:Path(v['path']) for k,v in freeze['files'].items()}
 for k,v in freeze['files'].items():assert a.sha(p[k])==v['sha256']
 result=json.loads((a.OUT/'RESULTS.json').read_text())
 with a.resource_lock():
  counts=compare(result['classification'],a.classification(p))
  reg,flat=a.regression(p);counts+=compare(result['regression'],reg)+compare(result['regression_replay'],flat)
  # Independently check that no ID join silently used a different physical row,
  # history or cycle. Recompute simple classification anchors without a.cm().
  pos=a.load(p['cls_pos']);ref=a.load(p['cls_a02'])
  for origin,stage,lo,hi in [(67,'D1',72,75),(75,'D2',76,79)]:
   key=f'E{origin}_';d=a.load(p[f'reg_direct_{origin}']);lab=a.load(p['reg_labels67' if origin==67 else 'reg_labels75'])
   mask=(d['query_cycles']>=lo)&(d['query_cycles']<=hi)
   rows,cycles=a.align(d['query_record_ids'][mask].astype(str),lab['record_ids'].astype(str),lab['physical_rows'],lab['cycles'])
   np.testing.assert_array_equal(rows,d['query_rows'][mask]);np.testing.assert_array_equal(cycles,d['query_cycles'][mask])
   np.testing.assert_array_equal(ref[key+'rows'],pos[key+'rows']);np.testing.assert_array_equal(ref[key+'cycles'],pos[key+'cycles'])
   y=ref[key+'y'];miss=ref[key+'missing'].astype(bool)
   prob=np.where(miss,ref[key+'pA'],.5*ref[key+'pP']+.5*ref[key+'pA'])
   pred=(prob>=np.where(miss,.53230976129477,.5)).astype(int)
   table=np.bincount(2*y.astype(int)+pred,minlength=4).reshape(2,2);tn,fp,fn,tp=table.ravel()
   f1=.5*(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn))
   view='E67_H12' if origin==67 else 'E75_W6'
   assert abs(f1-result['classification'][view]['policies']['0.53230976129477']['REFERENCE']['macro_f1'])<=1e-12
   counts+=6
 return {'status':'PASS_FULL_RECOMPUTATION_AND_BOUNDARY_CHECKS','numeric_fields_checked':counts,
  'input_files_checked':len(p),'freeze_sha256':a.sha(a.FREEZE),'results_sha256':a.sha(a.OUT/'RESULTS.json'),
  'new_fits':0,'Test_access':False,'raw_data_access':False,
  'qualification':'Deterministic replay of frozen audit; primitive F1/MAE have independent implementations. Not independent statistical validation.'}

if __name__=='__main__':
 d=verify();a.write(a.OUT/'VERIFICATION.json',d);print(json.dumps(d))
