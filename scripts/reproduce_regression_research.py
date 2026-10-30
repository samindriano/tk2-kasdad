"""Reconstruct/cache-check research without replacing historical measured reports."""
from pathlib import Path
import argparse,subprocess,sys,json,hashlib
from datetime import datetime,timezone
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'experiments/runs/regression_research_reproduction'
STAGES={'stage1':('research_regression_ranking.py',['stage1_ranking.json']),
        'stage23':('research_regression_models.py',['stage23_models.json']),
        'stage4':('research_regression_refinement.py',['stage4_refinement.json']),
        'stage5':('research_regression_lgb_text.py',['stage5_lgb_text.json'])}


@contextmanager
def preserve_reports(paths, destination):
    """Restore only the explicitly snapshotted reports, even if execution fails."""
    before={Path(p):Path(p).read_bytes()for p in paths}
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    try:
        yield
    finally:
        for path,original in before.items():
            if path.exists():
                (destination/path.name).write_bytes(path.read_bytes())
            path.write_bytes(original)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['models','stage1','stage23','stage4','stage5'],default='models')
    a=p.parse_args();names=['stage23','stage4','stage5']if a.stage=='models'else[a.stage]
    OUTPUT.mkdir(parents=True,exist_ok=True);records=[]
    for name in names:
        script,files=STAGES[name];paths=[ROOT/'reports/regression_research'/f for f in files]
        if not all(path.is_file()for path in paths):raise FileNotFoundError('Historical committed reports required before reproduction.')
        before={path.name:hashlib.sha256(path.read_bytes()).hexdigest()for path in paths}
        with preserve_reports(paths,OUTPUT/name):
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,capture_output=True,text=True)
            (OUTPUT/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        after={path.name:hashlib.sha256(path.read_bytes()).hexdigest()for path in paths}
        assert before==after,'Historical report restoration failed'
        record={'stage':name,'exit_code':result.returncode,'historical_reports_unchanged':True,'historical_hashes':before}
        records.append(record);print(json.dumps(record),flush=True)
        if result.returncode:raise RuntimeError('Reproduction failed; historical reports restored. Inspect ignored '+name+'.log')
    receipt={'run_at_utc':datetime.now(timezone.utc).isoformat(),'stages':records,
             'note':'New measured runtime reports/logs stay ignored. This runner never uploads or predicts official Test.'}
    (OUTPUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
