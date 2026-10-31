"""Isolated V2 live leaderboard/preflight and guarded submission entry point."""
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import campaign_kaggle as gate

gate.OUT=ROOT/'reports/optimization_v2/kaggle'
gate.CONFIG=ROOT/'experiments/optimization_v2'
gate.historical_gate.BASE={'classification':('artifact_signal','57833406',.82701),
                           'regression':('threat_rank','57832438',11.64134)}

def preflight(api, task):
    r=gate.preflight(api,task)
    r['campaign']='optimization_v2'
    rules=api.competition_list_pages(gate.slug(task),page_name='rules')
    r['rules_content']=[p.content for p in rules]
    # Public leaderboard rows only; never competitor code or private evidence.
    leaderboard=api.competition_leaderboard_view(gate.slug(task),page_size=200)
    rows=[]
    for i,row in enumerate(leaderboard or [],1):
        rows.append({'rank_in_returned_page':i,'team_name':str(row.team_name),
                     'score':float(row.score),'team_id':str(row.team_id)})
    ours=[d for d in rows if ''.join(c for c in d['team_name'].lower() if c.isalpha())=='kasdudlovers']
    assert len(ours)==1,'Team not uniquely present in retrieved leaderboard; paginate rather than infer'
    assert rows,'No leaderboard records returned'
    leader=rows[0]
    score=ours[0]['score']
    r['public_leaderboard']={'retrieved_rows':len(rows),'pagination_scope':'First200 API leaderboard rows',
         'team':ours[0],'leader':{'score':leader['score'],'rank_in_returned_page':1},
         'gap_to_leader':leader['score']-score if task=='classification' else score-leader['score'],
         'not_used_to_infer_labels':True}
    gate.save(gate.OUT/f'{task}_preflight.json',r)
    print(json.dumps({k:r[k] for k in ['task','num_allowed_now','num_total','team_name','deadline_api','public_leaderboard']}))
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','upload','poll']);p.add_argument('task',choices=['classification','regression']);p.add_argument('--manifest',type=Path);p.add_argument('--candidate');a=p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    if a.action=='preflight':preflight(api,a.task)
    elif a.action=='upload':
        assert a.manifest
        # The existing uploader invokes the isolated, authenticated preflight
        # internally immediately before its exclusive reservation.
        gate.upload(api,a.task,a.manifest)
    else:assert a.candidate;gate.poll(api,a.task,a.candidate)

if __name__=='__main__':main()
