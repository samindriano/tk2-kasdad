import numpy as np
from src.models.optimization_v2_stack import StructuredTextHead

def test_structure_text_head_labels_and_probability_contract():
    p=np.column_stack([np.linspace(.01,.99,90),np.linspace(.2,.9,90)])
    y=(np.arange(90)>38).astype(int)
    for kind,width in [('expanding_base',1),('expanding_stack_cb',2)]:
        model=StructuredTextHead(kind).fit(p,y);before=model.audit_state()
        pred=model.predict(p[:10]);assert before==model.audit_state()
        assert before['recipe_id']==kind and len(before['coef'][0])==width
        assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
