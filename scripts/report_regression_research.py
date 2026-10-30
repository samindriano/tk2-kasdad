"""Build compact scientific comparisons and figures from verified saved research."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.regression_research_common import REPORT,CACHE,save_json
from src.models.regression_round2_evaluation import load_protocol
from src.models.regression_round2_reporting import _table
from src.utils.paths import PROJECT_ROOT,task_files


def load_stages():
    return {s:json.loads((REPORT/(s+'.json')).read_text(encoding='utf-8'))for s in ['stage1_ranking','stage23_models','stage4_refinement','stage5_lgb_text']if(REPORT/(s+'.json')).exists()}


def comparisons(stages):
    out=[]
    for stage,r in stages.items():
        for label,a in r['aggregate']['primary'].items():
            b=r['aggregate']['secondary'][label]
            if stage=='stage1_ranking':m,policy=label.split('__');family='control'
            else:m,policy=label.split('__');family=stage
            jobs=[j for j in r['jobs']if(j.get('recipe',j.get('model'))==m and j['policy']==policy)]
            windows={j['window']:j['raw']['mae']for j in jobs}
            out.append({'Stage':stage,'Recipe':m,'Policy':policy,'Primary MAE':a['raw']['mae'],'Equal-cycle MAE':a['equal_cycle_mae'],
                        'H12 MAE':b['raw']['mae'],'H12_E':windows['H12_E'],'H12_L':windows['H12_L'],
                        'RMSE':a['raw']['rmse'],'Bias':a['raw']['signed_error'],'EF0':a['subgroups']['EF0']['mae'],'EF1':a['subgroups']['EF1']['mae'],
                        'Low tail':a['subgroups']['tail_low_le5']['mae'],'High tail':a['subgroups']['tail_high_ge95']['mae'],
                        'Missing':a['subgroups']['stream_missing']['mae'],'Present':a['subgroups']['stream_present']['mae'],
                        'Spearman':a['pooled_spearman']})
    return pd.DataFrame(out)


def selected_keys(stages):
    values=[('stage1_ranking','R1_D','raw','R1-D'),('stage23_models','hybrid_ridge','inner','Hybrid Ridge'),
            ('stage23_models','svd_hgb','inner','SVD32 + HGB'),('stage4_refinement','svd128','inner','SVD128 + HGB')]
    if 'stage5_lgb_text'in stages:values.append(('stage5_lgb_text','lgb_text','inner','SVD32 + LightGBM'))
    return values


def vector(stage,recipe,policy,family,frame):
    p=load_protocol();parts=[]
    for w in p['primary_windows'if family=='primary'else'secondary_windows']:
        if stage=='stage1_ranking':path=CACHE/f"ranking__{w['id']}__{recipe}__{policy}.npz"
        elif stage=='stage23_models':path=CACHE/f"eval__fit{w['fit'][1]}__{recipe}__{policy}.npz"
        elif stage=='stage4_refinement':path=CACHE/f"eval_refine__fit{w['fit'][1]}__{recipe}__{policy}.npz"
        else:path=CACHE/f"eval_lgbtext__fit{w['fit'][1]}__{policy}.npz"
        with np.load(path,allow_pickle=False)as z:rows,pred=z['validation_rows'].copy(),z['prediction'].copy()
        mask=frame.iloc[rows].cycle_code.between(*w['validation']).to_numpy();parts.append((rows[mask],pred[mask]))
    rows=np.concatenate([x[0]for x in parts]);pred=np.concatenate([x[1]for x in parts]);order=np.argsort(rows)
    assert len(np.unique(rows))==len(rows);return rows[order],pred[order]


def build():
    stages=load_stages();table=comparisons(stages);selected=selected_keys(stages)
    lines=['# Regression research — complete candidate comparison','', 'Raw/unclipped metrics remain separately labelled; all mapping outputs are clipped. H12 reuses primary validation records. Inner policies use only past inner labels; fixed-policy outer views are descriptive and are not unbiased winner selection.','',_table(table)]
    for stage,r in stages.items():
        for family in ['primary','secondary']:
            windows=list(dict.fromkeys(j['window']for j in r['jobs']if j['family']==family))
            rows=[]
            for label in r['aggregate'][family]:
                m,pol=label.split('__');jobs=[j for j in r['jobs']if j['family']==family and j.get('recipe',j.get('model'))==m and j['policy']==pol]
                rows.append({'Recipe/policy':label,**{w:next(j['raw']['mae']for j in jobs if j['window']==w)for w in windows}})
            lines+=['## '+stage+' '+family+' windows','',_table(pd.DataFrame(rows))]
    (REPORT/'COMPARISON_TABLES.md').write_text('\n'.join(lines),encoding='utf-8')
    folder=REPORT/'figures';folder.mkdir(exist_ok=True);figures=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white'})
    colors=['#837022','#245e8a','#b35d30','#768236','#a24a79'];styles=['-','--','-.',':','-'];markers=['o','s','^','D','X']
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),dpi=145,bbox_inches='tight');plt.close(fig);figures.append(str((folder/(name+'.png')).relative_to(PROJECT_ROOT)))
    # Every actual model, including negative controls, with a zero-origin bar scale.
    core=table[((table.Stage!='stage1_ranking')&table.Policy.eq('inner'))|((table.Stage=='stage1_ranking')&table.Policy.eq('raw'))]
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True);y=np.arange(len(core))
    names=[m+(' control'if s=='stage1_ranking'else'')for m,s in zip(core.Recipe,core.Stage)]
    for ax,col in zip(axes,['Primary MAE','H12 MAE']):
        ax.barh(y,core[col],color='#245e8a',alpha=.85);ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlabel('MAE (lower is better)');ax.set_title(col);ax.set_xlim(0,max(core[col])*1.12)
        for i,v in enumerate(core[col]):ax.text(v+.12,i,f'{v:.3f}',va='center',fontsize=8)
    fig.suptitle('Model comparisons — inner-selected mappings, reused development windows',fontweight='bold');fig.tight_layout();save(fig,'research_model_comparison')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for family,ax in zip(['primary','secondary'],axes):
        for m,color,marker in zip(['R1_C','R1_D','B_D_RECENT8','A_NO_CODES'],colors,markers):
            values=[stages['stage1_ranking']['aggregate'][family][m+'__'+name]['raw']['mae']for name in ['clip','rank25','rank50','rank75','rank100']]
            ax.plot([0,.25,.5,.75,1],values,color=color,marker=marker,label=m)
        ax.set_title(family);ax.set_xlabel('Fixed weight on prediction-only cycle percentile');ax.set_ylabel('MAE');ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Pure rank mapping does not repair ordering mistakes',fontweight='bold');fig.tight_layout();save(fig,'research_rank_mapping')
    wins=[w['id']for w in load_protocol()['primary_windows']+load_protocol()['secondary_windows']]
    data=[]
    for s,m,pol,label in selected:
        jobs=stages[s]['jobs'];data.append([next(j['raw']['mae']for j in jobs if j.get('recipe',j.get('model'))==m and j['policy']==pol and j['window']==w)for w in wins])
    fig,ax=plt.subplots(figsize=(11,4.2));im=ax.imshow(data,cmap='YlOrBr',aspect='auto');ax.set_xticks(range(len(wins)),wins);ax.set_yticks(range(len(selected)),[v[3]for v in selected])
    for i,row in enumerate(data):
        for j,v in enumerate(row):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white'if v>16 else'#222')
    ax.set_title('Temporal MAE — preserve early H12 counterexamples');fig.colorbar(im,ax=ax,label='Raw or selected-policy MAE');fig.tight_layout();save(fig,'research_window_heatmap')
    frame=pd.read_csv(task_files('regression')['train']);bins={};paired={}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for (s,m,pol,label),color,style,marker in zip(selected,colors,styles,markers):
        rows,pred=vector(s,m,pol,'primary',frame);y=frame.iloc[rows].threat_rank.to_numpy();stats=[]
        for lo in range(0,100,10):
            mask=(y>=lo)&((y<lo+10)if lo<90 else(y<=100));err=pred[mask]-y[mask];stats.append({'midpoint':lo+5,'n':int(mask.sum()),'mae':float(np.abs(err).mean()),'bias':float(err.mean())})
        bins[label]=stats
        for ax,key in zip(axes,['mae','bias']):ax.plot([v['midpoint']for v in stats],[v[key]for v in stats],color=color,linestyle=style,marker=marker,label=label)
    axes[0].set_title('Absolute error');axes[1].set_title('Signed error');axes[0].set_ylabel('MAE');axes[1].set_ylabel('Prediction − actual');axes[1].axhline(0,color='#555',linewidth=.8)
    for ax in axes:ax.set_xlabel('Fixed 10-point actual target bin (midpoint)');ax.grid(alpha=.18)
    axes[0].legend(fontsize=8);fig.suptitle('Primary conditional residuals — actual bins are evaluation only',fontweight='bold');fig.tight_layout();save(fig,'research_residuals')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,family in zip(axes,['primary','secondary']):
        x=np.arange(len(selected));width=.36
        for off,group,color in [(-width/2,'stream_present','#245e8a'),(width/2,'stream_missing','#b35d30')]:
            ax.bar(x+off,[stages[s]['aggregate'][family][m+'__'+pol]['subgroups'][group]['mae']for s,m,pol,_ in selected],width,color=color,label=group.replace('stream_',''))
        ax.set_xticks(x,[v[3].replace(' + ','\n')for v in selected],rotation=15);ax.set_ylabel('MAE');ax.set_title(family);ax.legend(fontsize=8)
    fig.suptitle('Stream groups: present n=36,010 / missing n=2,578 per family',fontweight='bold');fig.tight_layout();save(fig,'research_stream_subgroups')
    stress=stages['stage1_ranking']['batch_stress'];fig,ax=plt.subplots(figsize=(8,4.3))
    for m,color,marker in zip(['R1_C','R1_D','B_D_RECENT8','A_NO_CODES'],colors,markers):
        values=[]
        for f in [.25,.5,.75]:
            a=[v for v in stress if v['model']==m and v['fraction']==f and v['rank_weight']==1.]
            values.append(sum(v['mean_abs_mapping_change']*v['n']for v in a)/sum(v['n']for v in a))
        ax.plot([25,50,75],values,color=color,marker=marker,label=m)
    ax.set_xlabel('Rows retained per supplied cycle (%)');ax.set_ylabel('Mean |subset rank score − full-batch rank score|');ax.set_title('Incomplete-reference stress — source completeness remains unknown');ax.set_xticks([25,50,75]);ax.legend(fontsize=8);ax.grid(alpha=.18);fig.tight_layout();save(fig,'research_batch_sensitivity')
    ledger=json.loads((REPORT/'kaggle_log.json').read_text(encoding='utf-8'))if(REPORT/'kaggle_log.json').exists()else{'events':[]}
    scores={'Initial R1-D':13.53473};refs={}
    for e in ledger['events']:
        if e.get('event')=='candidate_status'and e.get('status')=='COMPLETE'and e.get('public_score')is not None:
            scores[e['candidate_id']]=float(e['public_score']);refs[e['candidate_id']]=e['submission_ref']
    fig,ax=plt.subplots(figsize=(10,4.6));x=np.arange(len(scores));ax.bar(x,list(scores.values()),color='#245e8a');ax.axhline(12.23275,color='#b35d30',linestyle='--',label='Official baseline 12.23275 (user supplied)')
    ax.set_xticks(x,[k.replace('_inner_v1','').replace('_','\n')for k in scores],rotation=10);ax.set_ylabel('Kaggle public MAE');ax.set_ylim(0,max(scores.values())*1.12)
    for i,v in enumerate(scores.values()):ax.text(i,v+.07,f'{v:.5f}',ha='center')
    ax.set_title('Limited external feedback — no private score or final-model claim');ax.legend(fontsize=8);fig.tight_layout();save(fig,'research_kaggle_scores')
    save_json(REPORT/'report_data.json',{'comparison_records':table.to_dict('records'),'target_bins':bins,'kaggle_scores':scores,'kaggle_refs':refs,'figures':figures})
    print('Built',len(figures),'figures and',len(table),'comparison rows')
    return figures


if __name__=='__main__':build()
