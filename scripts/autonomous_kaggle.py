"""Kaggle SDK preflight/max-two-new-per-task receipts; standard authentication.

Run with the installed Kaggle CLI environment's Python. No credential values
are read, logged, or passed as command-line arguments by this script.
"""
from pathlib import Path
from datetime import datetime,timezone
from contextlib import contextmanager
import argparse,csv,hashlib,json,math,os,re,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/autonomous_eda/kaggle'
BASE={'classification':('artifact_signal','57808225',.82641),'regression':('threat_rank','57810150',11.71363)}
EXPECTED_INPUTS={
    'classification':{'train':'875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c',
        'test':'4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c',
        'sample':'889f97d69659a64945421d841dcf96e5813020606f29b2992d4b9560cecdcba7'},
    'regression':{'train':'756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082',
        'test':'4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c',
        'sample':'63a51f4b7dfbb0eec9800099407abd8be14d25ec18a533cd39dc854ffc190a32'}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    with temp.open('x',encoding='utf-8',newline='\n')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(p)
def slug(task):return f'tp2-{task}-cyber-vulnerability-kasdd-gasal-2026-2027'
def simple(s,score=False):
    status=getattr(s.status,'name',str(s.status).split('.')[-1])
    d={'ref':str(s.ref),'date':str(s.date),'file_name':s.file_name,'description':s.description,'status':status,'team_name':s.team_name}
    if score:d['public_score']=float(s.public_score)if s.public_score not in [None,'']else None
    if getattr(s,'error_description',None):d['error_description']=s.error_description
    return d
def preflight(api,task):
    comp=slug(task);history=api.competition_submissions(comp,page_size=200)
    limits=api.competition_get_submission_limits(comp);pages=api.competition_list_pages(comp,page_name='rules')
    meta=[c for c in api.competitions_list(group='entered',search=comp).competitions if c.ref.rstrip('/').endswith('/'+comp)]
    assert len(meta)==1;meta=meta[0]
    old=next(s for s in history if str(s.ref)==BASE[task][1])
    assert re.sub('[^a-z]','',old.team_name.lower())=='kasdudlovers','Current account baseline is not the required group team'
    assert meta.user_has_entered and not meta.submissions_disabled and not meta.is_kernels_submissions_only
    assert limits.num_allowed_now>0,'No Kaggle submission slot currently available'
    rules='\n'.join(p.content for p in pages);assert rules and 'tim' in rules.lower()
    out={'at_utc':now(),'task':task,'competition':comp,'source':'Official Kaggle authenticated SDK API',
        'url':f'https://www.kaggle.com/competitions/{comp}','num_allowed_now':limits.num_allowed_now,
        'num_today':limits.num_today,'num_total':limits.num_total,'max_daily_submissions':meta.max_daily_submissions,
        'deadline_api':str(meta.deadline),'metric':meta.evaluation_metric,'team_name':old.team_name,
        'baseline_reference':simple(old,True),'baseline_team_verified_from_authenticated_submission_history':True,
        'rules_sha256':hashlib.sha256(rules.encode()).hexdigest(),
        'rules_summary':['Only provided assistant/course data; no additional datasets.','Do not copy other groups; cite external references and understand implementation.','Submit through the group team bearing the group name.'],
        'rules_url':f'https://www.kaggle.com/competitions/{comp}/rules',
        'historical_submissions_without_other_scores':[simple(s)for s in history],
        'milestone_limit_successful_new_submissions':2,'leaderboard_used_for_tuning':False}
    save(OUT/f'{task}_preflight.json',out);return out

def validate(task,manifest_path):
    p=Path(manifest_path).resolve();p.relative_to(OUT.resolve());m=read(p)
    assert m['task']==task and m['competition']==slug(task)and m['status']=='PREPARED_AND_QA_PASS'
    assert m['main_selection']['eligible_for_external_evaluation']is True
    selection_path=ROOT/'experiments/autonomous_eda/selection.json'
    assert sha(selection_path)==m['selection_file_sha256']
    selection=read(selection_path)
    selected=[c for c in selection['candidates'][task]if c['candidate_id']==m['candidate_id']]
    assert len(selected)==1 and selected[0]['eligible_for_external_evaluation']is True
    assert m['main_selection']==selected[0]
    assert m['recipe']==selected[0]['recipe']and m['submission_message']==selected[0]['submission_message']
    assert m['source_code_sha256']==selection['source_code_sha256']and m['source_code_sha256']
    for path,h in m['source_code_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h
    csv_path=(ROOT/m['csv_path']).resolve();csv_path.relative_to((ROOT/'submissions/autonomous_eda').resolve())
    assert sha(csv_path)==m['csv_sha256'];target=BASE[task][0]
    assert m['input_sha256']==EXPECTED_INPUTS[task]
    assert set(m['input_relative_paths'])=={'train','test','sample'}
    for role,h in EXPECTED_INPUTS[task].items():
        path=(ROOT/m['input_relative_paths'][role]).resolve()if not Path(m['input_relative_paths'][role]).is_absolute()else Path(m['input_relative_paths'][role])
        assert sha(path)==h
        if role=='test':
            with path.open(encoding='utf-8-sig',newline='')as f:header=next(csv.reader(f))
            assert not {'artifact_signal','threat_rank'}.intersection(header),'Test must not contain target columns'
    with csv_path.open(encoding='utf-8',newline='')as f:
        reader=csv.reader(f);assert next(reader)==['record_id',target];rows=list(reader)
    assert len(rows)==29421 and all(len(r)==2 for r in rows)
    ids=[r[0]for r in rows];assert all(ids)and len(set(ids))==len(ids)
    assert hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()==m['ordered_test_ids_sha256']
    for _,v in rows:
        x=float(v);assert math.isfinite(x)
        if task=='classification':assert v in ['0','1']
        else:assert 0<=x<=100
    for role in ['test','sample']:
        input_path=Path(m['input_relative_paths'][role]);input_path=input_path if input_path.is_absolute()else ROOT/input_path
        with input_path.open(encoding='utf-8-sig',newline='')as f:
            sample=list(csv.DictReader(f));assert [r['record_id']for r in sample]==ids
    assert m['fit_preprocessing_only_train']is True and m['test_labels_used']is False
    assert m['candidate_differs_from_accepted_baseline']is True
    return m,csv_path

@contextmanager
def lock(task):
    path=OUT/f'.{task}_upload.lock';OUT.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:f.write(str(os.getpid()))
    try:yield
    finally:path.unlink()

def upload(api,task,path):
    m,csv_path=validate(task,path);candidate=m['candidate_id']
    assert re.fullmatch('[a-z0-9_]{1,70}',candidate)
    ledger=OUT/f'{task}__{candidate}_ledger.json'
    with lock(task):
        if ledger.exists():raise RuntimeError('Submission already reserved/accepted for this task; reconcile/poll rather than upload again')
        phase=read(ROOT/'experiments/autonomous_eda/milestone.json')
        assert len(list(OUT.glob(f'{task}__*_ledger.json')))<phase['max_successful_new_submissions'][task]<=2
        pre=preflight(api,task)
        assert not any(s['file_name']==csv_path.name or s['description']==m['submission_message']for s in pre['historical_submissions_without_other_scores'])
        state={'task':task,'milestone':'autonomous_eda_20261031_v1','status':'RESERVED_BEFORE_UPLOAD','at_utc':now(),
            'candidate_id':m['candidate_id'],'manifest_sha256':sha(path),'csv_sha256':sha(csv_path),'file_name':csv_path.name,
            'message':m['submission_message'],'num_total_before_upload':pre['num_total'],'successful_limit_per_task':2,'ref':None}
        save(ledger,state)
        try:
            response=api.competition_submit(str(csv_path),m['submission_message'],slug(task),quiet=True)
            ref=getattr(response,'ref',None)
            if not ref:raise RuntimeError('Kaggle did not return a submission reference')
            state.update({'status':'ACCEPTED_AWAITING_SCORE','ref':str(ref),'accepted_at_utc':now()});save(ledger,state)
            poll(api,task,candidate)
        except Exception as error:
            # Never erase the reservation or automatically repeat an ambiguous upload.
            current=read(ledger);current['last_error_type']=type(error).__name__;current['last_error_at_utc']=now();save(ledger,current);raise

def poll(api,task,candidate):
    assert re.fullmatch('[a-z0-9_]{1,70}',candidate)
    path=OUT/f'{task}__{candidate}_ledger.json';state=read(path)
    if not state.get('ref'):
        found=[s for s in api.competition_submissions(slug(task),page_size=200)if s.file_name==state['file_name']and s.description==state['message']]
        assert len(found)==1,'No unique accepted submission found; no retry issued'
        state['ref']=str(found[0].ref)
    submission=api.get_submission(int(state['ref']));receipt=simple(submission,True)
    assert receipt['file_name']==state['file_name']and receipt['description']==state['message']
    assert re.sub('[^a-z]','',receipt['team_name'].lower())=='kasdudlovers'
    state.update({'status':receipt['status'],'last_checked_utc':now(),'receipt':receipt})
    if receipt['status']=='COMPLETE':
        assert receipt['public_score']is not None
        baseline=BASE[task][2];score=receipt['public_score']
        state['external_comparison']={'baseline_public_score':baseline,'candidate_public_score':score,
            'improvement':score-baseline if task=='classification'else baseline-score,
            'unbiased_generalization_estimate':False,'automatically_promote_final_model':False}
    save(path,state);print(json.dumps(state,ensure_ascii=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','upload','poll']);p.add_argument('task',choices=list(BASE));p.add_argument('--manifest',type=Path);p.add_argument('--candidate');a=p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    if a.action=='preflight':
        s=preflight(api,a.task);print(json.dumps({k:s[k]for k in ['task','num_allowed_now','num_total','deadline_api','team_name','metric']},ensure_ascii=True))
    elif a.action=='upload':assert a.manifest;upload(api,a.task,a.manifest)
    else:assert a.candidate;poll(api,a.task,a.candidate)
if __name__=='__main__':main()
