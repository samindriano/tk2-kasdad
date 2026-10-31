"""Isolated campaign submission gate, live quota and idempotent receipts."""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import autonomous_kaggle as historical_gate
from src.utils.paths import task_files
OUT=ROOT/'reports/dual_campaign/kaggle';CONFIG=ROOT/'experiments/dual_campaign'
read=historical_gate.read;sha=historical_gate.sha;simple=historical_gate.simple;slug=historical_gate.slug
def now():return datetime.now(timezone.utc).isoformat()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.tmp')
    with temp.open('x',encoding='utf-8',newline='\n')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(p)
def preflight(api,task):
    # Reuse verified read-only API queries, redirect every new receipt to this campaign.
    original=historical_gate.OUT
    try:
        historical_gate.OUT=OUT
        record=historical_gate.preflight(api,task)
    finally:historical_gate.OUT=original
    record.pop('milestone_limit_successful_new_submissions',None)
    record['campaign_budget']=read(CONFIG/'milestone.json')['research_upload_budget'][task]
    record['campaign']='dual_campaign_20261031_v1'
    deadline=datetime.fromisoformat(record['deadline_api']).replace(tzinfo=timezone.utc)
    assert datetime.now(timezone.utc)<deadline,'Competition deadline has passed'
    record['deadline_interpretation']='Kaggle API UTC deadline; consistent with course23:55WIB'
    history=api.competition_submissions(slug(task),page_size=200)
    initial=next(s for s in history if str(s.ref)==historical_gate.BASE[task][1])
    record['other_team_member_submission_count']=sum(s.submitted_by!=initial.submitted_by for s in history)
    record['all_team_history_count_checked']=len(history)
    save(OUT/f'{task}_preflight.json',record);return record

def validate(task,manifest_path):
    path=Path(manifest_path).resolve();path.relative_to(OUT.resolve());m=read(path)
    assert m['status']=='PREPARED_AND_QA_PASS'and m['task']==task and m['competition']==slug(task)
    decision_path=(ROOT/m['decision_path']).resolve();decision_path.relative_to((CONFIG/'decisions').resolve())
    assert sha(decision_path)==m['decision_sha256'];decision=read(decision_path)
    assert decision['candidate_id']==m['candidate_id']and decision['approved_for_external_evaluation']is True
    assert decision['recipe']==m['recipe']and decision['submission_message']==m['submission_message']
    assert m['source_code_sha256']==decision['source_code_sha256']and m['source_code_sha256']
    for p,h in m['source_code_sha256'].items():
        content=(ROOT/p).read_bytes().replace(b'\r\n',b'\n');assert hashlib.sha256(content).hexdigest()==h,p
        committed=subprocess.check_output(['git','-C',str(ROOT),'show',m['source_commit']+':'+p])
        assert hashlib.sha256(committed.replace(b'\r\n',b'\n')).hexdigest()==h,p
    for p,h in decision['internal_result_sha256'].items():assert sha(ROOT/p)==h,p
    canonical=historical_gate.EXPECTED_INPUTS[task];assert m['input_sha256']==canonical
    inputs=task_files(task);assert all(inputs.values())
    for role,h in canonical.items():assert sha(inputs[role])==h
    target='artifact_signal'if task=='classification'else'threat_rank'
    ids={}
    for role in ['test','sample']:
        with inputs[role].open(encoding='utf-8-sig',newline='')as f:
            reader=csv.DictReader(f)
            if role=='test':assert not {'artifact_signal','threat_rank'}.intersection(reader.fieldnames)
            else:assert reader.fieldnames==['record_id',target]
            ids[role]=[r['record_id']for r in reader]
    assert ids['test']==ids['sample']and len(ids['test'])==len(set(ids['test']))==29421
    output=(ROOT/m['csv_path']).resolve();output.relative_to((ROOT/'submissions/dual_campaign').resolve());assert sha(output)==m['csv_sha256']
    with output.open(encoding='utf-8',newline='')as f:
        reader=csv.reader(f);assert next(reader)==['record_id',target];rows=list(reader)
    assert len(rows)==29421 and all(len(r)==2 for r in rows)and [r[0]for r in rows]==ids['test']
    assert hashlib.sha256(('\n'.join(ids['test'])+'\n').encode()).hexdigest()==m['ordered_test_ids_sha256']
    for _,raw in rows:
        value=float(raw);assert math.isfinite(value)
        if task=='classification':assert raw in ['0','1']
        else:assert 0<=value<=100
    assert m['fit_preprocessing_only_train']is True and m['test_labels_used']is False
    assert m['candidate_differs_from_accepted_baseline']is True
    return m,output

@contextmanager
def lock(task):
    path=OUT/f'.{task}.lock';OUT.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:f.write(str(os.getpid()))
    try:yield
    finally:path.unlink()

def poll(api,task,candidate):
    assert re.fullmatch('[a-z0-9_]{1,80}',candidate);path=OUT/f'{task}__{candidate}_ledger.json';d=read(path)
    if not d.get('ref'):
        matches=[s for s in api.competition_submissions(slug(task),page_size=200)if s.file_name==d['file_name']and s.description==d['message']]
        assert len(matches)==1,'No unique acceptance found; no automatic retry';d['ref']=str(matches[0].ref)
    result=simple(api.get_submission(int(d['ref'])),True)
    assert result['file_name']==d['file_name']and result['description']==d['message']and result['team_name']=='kasdud lovers'
    d.update({'status':result['status'],'receipt':result,'checked_at_utc':now()})
    if result['status']=='COMPLETE':
        assert result['public_score']is not None
        champ=historical_gate.BASE[task][2];score=result['public_score']
        d['public_comparison']={'official_champion_score':champ,'candidate_score':score,
            'improvement':score-champ if task=='classification'else champ-score,'unbiased_generalization':False,'automatic_promotion':False}
    save(path,d);print(json.dumps(d,ensure_ascii=True))

def upload(api,task,path):
    m,output=validate(task,path);candidate=m['candidate_id'];assert re.fullmatch('[a-z0-9_]{1,80}',candidate)
    ledger=OUT/f'{task}__{candidate}_ledger.json'
    with lock(task):
        assert not ledger.exists(),'Already reserved/accepted; reconcile rather than duplicate upload'
        maximum=read(CONFIG/'milestone.json')['research_upload_budget'][task]
        assert len(list(OUT.glob(f'{task}__*_ledger.json')))<maximum
        live=preflight(api,task)
        assert live['num_allowed_now']>1,'Leave at least one shared-team slot; no upload at last available slot'
        assert not any(r['file_name']==output.name or r['description']==m['submission_message']for r in live['historical_submissions_without_other_scores'])
        d={'task':task,'candidate_id':candidate,'status':'RESERVED_BEFORE_UPLOAD','at_utc':now(),'ref':None,
            'file_name':output.name,'message':m['submission_message'],'csv_sha256':sha(output),'manifest_sha256':sha(path),
            'num_team_total_before':live['num_total'],'num_allowed_before':live['num_allowed_now'],'campaign_budget':maximum}
        save(ledger,d)
        try:
            result=api.competition_submit(str(output),m['submission_message'],slug(task),quiet=True)
            assert getattr(result,'ref',None),'No reference returned; reconcile before retry'
            d['ref']=str(result.ref);d['status']='ACCEPTED_AWAITING_SCORE';save(ledger,d);poll(api,task,candidate)
        except Exception as error:
            d=read(ledger);d['last_error_type']=type(error).__name__;save(ledger,d);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','upload','poll']);p.add_argument('task',choices=['classification','regression']);p.add_argument('--manifest',type=Path);p.add_argument('--candidate');args=p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    if args.action=='preflight':
        r=preflight(api,args.task);print(json.dumps({k:r[k]for k in ['task','num_allowed_now','num_total','team_name','deadline_api','campaign_budget']}))
    elif args.action=='upload':assert args.manifest;upload(api,args.task,args.manifest)
    else:assert args.candidate;poll(api,args.task,args.candidate)
if __name__=='__main__':main()
