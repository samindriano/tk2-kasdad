"""Run discovery in an isolated file copy without pre-existing reports or caches.

Uses the current installed Python environment and the documented external official
Train root. This is file/dependency isolation, not a newly installed environment.
"""
from pathlib import Path
import json
import hashlib
import os
import shutil
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.paths import data_root
from scripts.build_eda_candidate_discovery import build_notebook


def main():
    temp=Path(tempfile.mkdtemp(prefix='tp2-eda-discovery-clean-'))
    for p in (ROOT/'src').rglob('*.py'):
        q=temp/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    paths=list((ROOT/'scripts').glob('eda_candidates_*.py'))+[ROOT/'scripts/build_eda_candidate_discovery.py']
    for p in paths:
        q=temp/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    build_notebook()
    source=ROOT/'notebooks/samuel/eda_candidate_discovery.ipynb'
    dest=temp/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    assert not(temp/'reports').exists() and not(temp/'experiments').exists() and not(temp/'data').exists()
    raw=data_root();old=os.environ.get('TP2_DATA_DIR');os.environ['TP2_DATA_DIR']=str(raw)
    started=time.perf_counter()
    try:
        notebook=nbformat.read(dest,4)
        NotebookClient(notebook,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(temp)}}).execute()
        nbformat.write(notebook,dest);nbformat.validate(notebook)
        codes=[c for c in notebook.cells if c.cell_type=='code']
        errors=[o for c in codes for o in c.outputs if o.output_type=='error']
        pngs=sum('image/png'in o.get('data',{})for c in codes for o in c.outputs)
        assert not errors and [c.execution_count for c in codes]==list(range(1,len(codes)+1)) and pngs>=9
        compared=[]
        for p in (ROOT/'reports/eda_candidates').rglob('*'):
            if p.is_file() and p.suffix in ['.json','.csv'] and p.parent.name in ['text','missingness','categories','targets']:
                q=temp/p.relative_to(ROOT)
                assert q.exists() and p.read_bytes()==q.read_bytes(),str(p)
                compared.append(str(p.relative_to(ROOT)))
        shutil.copy2(dest,source)
        receipt={'status':'PASS','execution':'Fresh NotebookClient kernel / Restart & Run All equivalent',
            'cells':len(notebook.cells),'executed_code_cells':len(codes),'errors':0,'embedded_pngs':pngs,
            'isolated_copy_had_no_reports_caches_or_local_data':True,
            'only_external_input':'Official Train root via documented TP2_DATA_DIR',
            'installed_environment':'Current environment reused; not a fresh dependency installation',
            'compared_aggregate_json_csv':compared,'elapsed_seconds':time.perf_counter()-started,
            'analysis_script_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'local_preview_root':str(temp),'no_models_or_uploads':True,
            'resolved_portability_issue':'Category runner root argument now denotes project root; dataset consistently uses TP2_DATA_DIR, including when the copy has no data/raw folder.'}
        (ROOT/'reports/eda_candidates/execution.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        print('PASS:',len(codes),'executed code cells;',pngs,'embedded PNGs;',len(compared),'identical aggregate JSON/CSV files.')
        return receipt
    finally:
        if old is None:os.environ.pop('TP2_DATA_DIR',None)
        else:os.environ['TP2_DATA_DIR']=old


if __name__=='__main__':main()
