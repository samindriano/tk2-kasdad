"""Fresh-kernel validation from committed evidence and official Train only.

No ignored experiment cache, Test file, submission CSV, or credentials are
available in the isolated copy. Uses currently installed dependencies.
"""
from pathlib import Path
import argparse
import base64
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

ROOT=Path(__file__).resolve().parents[1]
NAMES={'regression':'09_samuel_final_regression.ipynb','classification':'10_samuel_final_classification.ipynb'}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--check-preservation',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('Use --execute to run the two fixed notebook recipes.')
    sys.path.insert(0,str(ROOT))
    from src.utils.paths import task_files
    bundle=Path(tempfile.mkdtemp(prefix='tp2-modeling-final-'))
    for folder in ['src','reports','docs']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.json','.md','.png'} and '__pycache__'not in p.parts:
                if p==ROOT/'reports/samuel_final/VALIDATION.json':continue
                q=bundle/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    # Frozen protocol JSONs are committed inputs, not ignored prediction caches.
    for task in ['classification_round1','classification_round2']:
        for p in (ROOT/'experiments'/task).glob('*.json'):
            q=bundle/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for name in ['eda_candidates_text.py','eda_a_period_robustness.py','eda_a_within_cells.py']:
        p=ROOT/'scripts'/name;q=bundle/'scripts'/name;q.parent.mkdir(exist_ok=True);shutil.copy2(p,q)
    for p in ROOT.glob('requirements*.txt'):shutil.copy2(p,bundle/p.name)
    for p in (ROOT/'notebooks/samuel').glob('*.ipynb'):
        q=bundle/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    raw=bundle/'official_data';input_hashes={}
    for task in NAMES:
        source=task_files(task)['train'];assert source is not None
        dest=raw/task/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        input_hashes[f'{task}/train']=digest(source)
    assert not(bundle/'experiments/runs').exists() and not(bundle/'submissions').exists()
    assert len(list(raw.rglob('*.csv')))==2
    old=os.environ.get('TP2_DATA_DIR');os.environ['TP2_DATA_DIR']=str(raw)
    receipt={'status':'RUNNING','evidence_base_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'isolated_source_copy':True,'cache_dependencies_present':False,'input_sha256':input_hashes,
        'environment':'Currently installed dependencies; fresh kernels, not a fresh dependency installation',
        'full_train_inference_executed':False,'kaggle_uploads':0,'notebooks':{}}
    started=time.perf_counter()
    try:
        for task,name in NAMES.items():
            path=bundle/'notebooks/samuel'/name;nb=nbformat.read(path,4)
            start=time.perf_counter()
            NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(bundle)}}).execute()
            nbformat.validate(nb);nbformat.write(nb,path)
            codes=[c for c in nb.cells if c.cell_type=='code']
            assert [c.execution_count for c in codes]==list(range(1,len(codes)+1))
            assert not any(o.output_type=='error'for c in codes for o in c.outputs)
            figures=[]
            for ci,c in enumerate(nb.cells):
                for oi,o in enumerate(c.get('outputs',[])):
                    if 'image/png'in o.get('data',{}):
                        p=bundle/f'artifacts/samuel_final/figures/{task}_{ci}_{oi}.png';p.parent.mkdir(parents=True,exist_ok=True)
                        p.write_bytes(base64.b64decode(o.data['image/png']));figures.append(str(p))
            assert len(figures)>=2
            core=json.loads((bundle/f'artifacts/samuel_final/{task}_core_receipt.json').read_text(encoding='utf-8'))
            assert core['matches_historical_predictions'] and core['fit_preprocessing_only_train'] and not core['test_used_in_core']
            receipt['notebooks'][task]={'cells':len(nb.cells),'executed_code_cells':len(codes),'errors':0,
                'embedded_figures':len(figures),'local_figure_paths':figures,'core_recomputed':core,
                'seconds':time.perf_counter()-start,'notebook_sha256':digest(path)}
            print(task,'PASS',len(codes),'code cells;',len(figures),'figures;',core['metrics'],flush=True)
        if args.check_preservation:
            historical=json.loads((ROOT/'reports/samuel_final/PRESERVATION.json').read_text(encoding='utf-8'))
            for p,sha in historical['protected_tracked_files'].items():assert digest(ROOT/p)==sha,p
            receipt['preserved_historical_files']=len(historical['protected_tracked_files'])
            receipt['approved_eda11_unchanged']=True
        for task,name in NAMES.items():shutil.copy2(bundle/'notebooks/samuel'/name,ROOT/'notebooks/samuel'/name)
        receipt['status']='PASS_CORE_RESTART_RUN_ALL';receipt['seconds']=time.perf_counter()-started
        receipt['local_review_copy']=str(bundle);receipt['visual_review']='PENDING_MAIN_PNG_REVIEW'
        out=ROOT/'reports/samuel_final/VALIDATION.json'
        out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('Saved',out,flush=True)
    finally:
        if old is None:os.environ.pop('TP2_DATA_DIR',None)
        else:os.environ['TP2_DATA_DIR']=old


if __name__=='__main__':main()
