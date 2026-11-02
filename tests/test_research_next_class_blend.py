import copy
import json

import numpy as np
from sklearn.metrics import f1_score,recall_score

from scripts.optimization_v3_research_next_class_blend import PROTOCOL,gate_result,metric


def test_metric_ties_and_zero_class_denominators():
    for y,p in [([0,1,1,0],[.5,.5,.49,.1]),([0,0],[.1,.8]),([1,1],[.2,.9])]:
        m=metric(y,p);pred=np.asarray(p)>=.5
        assert abs(m['macro_f1']-f1_score(y,pred,labels=[0,1],average='macro',zero_division=0))<1e-15
        assert abs(m['positive_recall']-recall_score(y,pred,zero_division=0))<1e-15


def passing_views():
    return {name:{'metrics':{arm:{'macro_f1':value} for arm,value in
            [('candidate',.85),('sparse25',.8),('N01_class_only',.82),('N02_sketch',.83),('matched_combo',.81)]}}
            for name in ['primary','W1','W4','H12_55','H12_67','H12_pooled']}


def test_h12_only_group_failure_blocks_otherwise_passing_blend():
    g=json.loads(PROTOCOL.read_text())['gates'];v=passing_views()
    assert gate_result(v,[],g)['passed']
    bad=[dict(arm='candidate',view='H12_67',supported=True,guard_pass=False)]
    assert not gate_result(v,bad,g)['passed']
    bad[0]['supported']=False
    assert gate_result(v,bad,g)['passed']


def test_best_component_and_matched_control_are_protected():
    g=json.loads(PROTOCOL.read_text())['gates'];v=passing_views()
    for arm in ['N01_class_only','N02_sketch','matched_combo']:
        bad=copy.deepcopy(v);bad['primary']['metrics'][arm]['macro_f1']=.85
        assert not gate_result(bad,[],g)['passed']
