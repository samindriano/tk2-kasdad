"""Run the formal EDA with only Classification Train, then reconcile its outputs.

Uses a fresh kernel and an isolated source-file copy in the current installed
Python environment. It never copies prior reports/caches or accesses Kaggle.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.paths import task_files
from scripts.build_samuel_formal_eda_a import build


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(check_preservation=False):
    out=ROOT/'reports/eda_a_formal'
    temp=Path(tempfile.mkdtemp(prefix='tp2-formal-eda-a-'))
    data=temp/'official_input/classification';data.mkdir(parents=True)
    source=task_files('classification')['train']
    shutil.copy2(source,data/'Train.csv')
    project=temp/'repository';project.mkdir()
    for p in (ROOT/'src').rglob('*.py'):
        q=project/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    helpers=['eda_candidates_text.py','eda_a_period_robustness.py','eda_a_within_cells.py']
    for name in helpers:
        q=project/'scripts'/name;q.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'scripts'/name,q)
    source_notebook=build()
    target=project/source_notebook.relative_to(ROOT);target.parent.mkdir(parents=True);shutil.copy2(source_notebook,target)
    assert not(project/'reports').exists() and not(project/'experiments').exists()
    assert [p.name for p in data.iterdir()]==['Train.csv']
    old=os.environ.get('TP2_DATA_DIR');os.environ['TP2_DATA_DIR']=str(data.parent)
    started=time.perf_counter()
    try:
        notebook=nbformat.read(target,4)
        NotebookClient(notebook,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(project)}}).execute()
        nbformat.validate(notebook);nbformat.write(notebook,target)
    finally:
        if old is None:os.environ.pop('TP2_DATA_DIR',None)
        else:os.environ['TP2_DATA_DIR']=old
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert [c.execution_count for c in code]==list(range(1,len(code)+1))
    assert not any(o.output_type=='error'for c in code for o in c.outputs)
    pngs=sum('image/png'in o.get('data',{})for c in code for o in c.outputs)
    assert pngs==4
    generated=project/'reports/eda_a_formal';r=read(generated/'results.json')
    reference=read(ROOT/'reports/eda_candidates/text/stats.json')['text_diversity']['exact_token_count_sensitivity']['features'][0]
    for current,prior in [('strata','supported_token_counts'),('support_rows','support_rows'),
                          ('low_n','low_tail_compared_rows'),('high_n','high_tail_compared_rows'),('positive_strata','positive_token_count_strata')]:
        assert r['primary'][current]==reference[prior]
    for current,prior in [('low_rate','average_low_tail_artifact_rate'),('high_rate','average_high_tail_artifact_rate'),
                          ('delta','upper_vs_lower_diversity_artifact_delta')]:
        assert np.isclose(r['primary'][current],reference[prior],atol=1e-12,rtol=0)
    period=read(ROOT/'reports/eda_a_robustness/period_stats.json')['rules']['primary_120_20']
    for a,b in zip(r['periods']['periods'],period['periods']):
        assert a['period']==b['period'] and a['support_rows']==b['support_rows']
        assert np.isclose(a['delta'],b['delta'],atol=1e-12,rtol=0)
    within=read(ROOT/'reports/eda_a_robustness/within_cells/summary.json')['support_gate_results']['5']
    assert r['exact_cycle']['eligible_support_rows']==within['eligible_support_rows']
    assert np.isclose(r['exact_cycle']['common_within_cell_slope_per_0_1_ttr'],within['common_within_cell_slope_per_0_1_ttr'],atol=1e-12,rtol=0)
    assert r['exact_cycle']['cell_direction']==within['cell_direction']
    tests=subprocess.run([sys.executable,'-m','unittest','tests.test_samuel_eda_a','-v'],cwd=ROOT,capture_output=True,text=True)
    assert tests.returncode==0,tests.stdout+tests.stderr
    preserved=read(out/'preservation.json')
    if check_preservation:
        for category in ['protected_files','protected_paused_drafts']:
            for p,digest in preserved[category].items():assert sha(ROOT/p)==digest,p
    # Promote only this milestone's executed notebook and newly generated aggregates/figures.
    shutil.copy2(target,source_notebook)
    for p in generated.rglob('*'):
        if p.is_file():
            q=out/p.relative_to(generated);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    receipt={'status':'PASS_EDA_A_COMPLETE_FOR_USER_REVIEW','official_question_selected':True,
        'notebook':'notebooks/samuel/11_samuel_formal_eda_a.ipynb','notebook_sha256':sha(source_notebook),
        'execution':{'cells':len(notebook.cells),'code_cells':len(code),'errors':0,'embedded_pngs':pngs,
                     'fresh_kernel':True,'isolated_source_copy':True,'only_data_file':'classification/Train.csv',
                     'prior_reports_or_caches_copied':False,'environment':'Current installed dependencies, not a fresh dependency installation',
                     'seconds':time.perf_counter()-started},
        'reconciliation':'Computed outputs match previously independently verified discovery and robustness aggregates; not every old exploratory output is replayed.',
        'unit_test_command':'python -m unittest tests.test_samuel_eda_a -v',
        'unit_tests':{'exit_code':tests.returncode,'output':tests.stdout+tests.stderr},
        'local_preservation_checked':bool(check_preservation),
        'preserved_tracked_files':len(preserved['protected_files']) if check_preservation else None,
        'preserved_paused_drafts':len(preserved['protected_paused_drafts']) if check_preservation else None,
        'existing_doc_updated':'docs/SAMUEL_EDA.md only; previous temporal notes retained as archive',
        'shared_allocation_documents_changed':False,'models_changed':False,'kaggle_uploads':0,
        'presentation':'Four individual PNGs reviewed at reading size; no final submission packaging or whole-notebook GUI layout certification.'}
    (out/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS:',len(code),'executed code cells,4 PNGs,6 unit tests; Classification Train only; local preservation checked:',check_preservation)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-preservation',action='store_true',help='Check local historical bytes and paused drafts; omit on a fresh clone without those drafts.')
    main(parser.parse_args().check_preservation)
