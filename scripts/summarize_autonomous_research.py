"""Explicitly separate primary/H12 views in research summary artifacts."""
from pathlib import Path
import hashlib,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/autonomous_eda'
def main():
    r=json.loads((OUT/'regression/results.json').read_text());inner=json.loads((OUT/'regression/inner_refinement.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,family,title in zip(axes,['primary','h12'],['Primary: history diperbarui tiap 4 cycle','H12: history tetap selama 12 cycle']):
        rows=r['aggregate'][family]['per_cycle'];values=[x['delta_mae']for x in rows]
        ax.bar([x['cycle']for x in rows],values,color=['#107969'if v<0 else'#bb642e'for v in values])
        ax.axhline(0,color='gray',lw=1);ax.set_title(title,fontsize=11);ax.set_xlabel('Cycle');ax.set_ylabel('Δ MAE SVD128 − SVD32');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Per-cycle differences: view terpisah, records saling overlap',fontsize=12)
    fig.savefig(OUT/'regression/figures/primary_h12_cycle_differences.png',dpi=160);plt.close(fig)
    rows=[]
    for family in ['primary','h12']:
        b=r['aggregate'][family]
        for model,metric in [('SVD32 frozen',b['control']),('SVD128 paired',b['candidate']),('SVD128 inner',inner['selected_vs_stage5']['aggregate'][family]['candidate'])]:
            rows.append({'family':family,'model':model,'MAE':metric['raw_mae'],'equal_cycle_MAE':metric['equal_cycle_mae'],'bias':metric['signed_error'],'delta_MAE_vs_SVD32':metric['raw_mae']-b['control']['raw_mae']})
    pd.DataFrame(rows).to_csv(OUT/'regression/comparison.csv',index=False)
    windows=[];subgroups=[]
    for variant,report in [('paired',r),('inner',inner['selected_vs_stage5'])]:
        for window,v in report['windows'].items():windows.append({'variant':variant,'window':window,**{k:v[k]for k in ['control_mae','candidate_mae','delta_mae']}})
        for family,v in report['aggregate'].items():
            for group,s in v['subgroups'].items():subgroups.append({'variant':variant,'family':family,'group':group,'n':s['candidate']['n'],
                'control_MAE':s['control']['mae'],'candidate_MAE':s['candidate']['mae'],'delta_MAE':s['candidate']['mae']-s['control']['mae']})
    pd.DataFrame(windows).to_csv(OUT/'regression/windows_summary.csv',index=False)
    pd.DataFrame(subgroups).to_csv(OUT/'regression/subgroups_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
