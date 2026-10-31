"""Small decision artifacts from already verified discovery/ablation results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/impact_eda'
def main():
    a=json.loads((OUT/'availability/stats.json').read_text(encoding='utf-8'))
    s=a['text_load_sensitivity']['comparisons']
    schemes=[('Mentah',a['overall_missing_minus_present_pp'],1.),
        ('Exact cycle',s['cycle_only']['standardized_missing_minus_present_pp'],s['cycle_only']['coverage_all_rows']),
        ('Volume (5 kuantil)',s['volume_only'][0]['standardized_missing_minus_present_pp'],s['volume_only'][0]['coverage_all_rows']),
        ('Periode × volume (3×5)',s['period_x_volume'][0]['standardized_missing_minus_present_pp'],s['period_x_volume'][0]['coverage_all_rows']),
        ('Periode × volume (3×10)',s['period_x_volume'][1]['standardized_missing_minus_present_pp'],s['period_x_volume'][1]['coverage_all_rows'])]
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    for i,(label,value,coverage)in enumerate(schemes):
        ax.scatter(value,i,color='#28718e'if value>=0 else'#b26222',s=65,zorder=3)
        ax.annotate(f'{value:+.2f} pp; coverage {coverage:.2%}',(value,i),xytext=(8,0),textcoords='offset points',va='center',fontsize=10)
    ax.set_yticks(range(len(schemes)),[x[0]for x in schemes]);ax.invert_yaxis();ax.axvline(0,color='gray',lw=1)
    ax.set_xlim(-4.2,8.5);ax.set_xlabel('Prevalence artifact: missing − present (poin persentase)')
    ax.set_title('Komposisi volume mengubah pembacaan gap availability\nStandardisasi deskriptif; bukan efek kausal',fontsize=12)
    ax.grid(axis='x',alpha=.2);fig.savefig(OUT/'availability/standardized_contrasts.png',dpi=160);plt.close(fig)
    r=json.loads((OUT/'ablation/results.json').read_text());sub=[];period=[]
    for family,models in r['aggregate'].items():
        for group,b in models['control']['subgroups'].items():
            c=models['coverage']['subgroups'][group]
            sub.append({'family':family,'subgroup':group,'n':b['n'],'control_MAE':b['mae'],'coverage_MAE':c['mae'],
                'delta_MAE_candidate_minus_control':c['mae']-b['mae'],'control_bias':b['signed_error'],'candidate_bias':c['signed_error']})
    for label,windows in [('discovery_56_67',['W1','W2','W3']),('late_sensitivity_68_79',['W4','W5','W6'])]:
        values={}
        for model in ['control','coverage']:
            jobs=[j for j in r['jobs']if j['model']==model and j['window']in windows]
            weights=[j['raw']['n']for j in jobs];mae=float(np.average([j['raw']['mae']for j in jobs],weights=weights))
            cycle_mae=[c['mae']for j in jobs for c in j['per_cycle']]
            period.append({'view':label,'model':model,'n':sum(weights),'MAE':mae,'equal_cycle_MAE':float(np.mean(cycle_mae))});values[model]=mae
        print(label,'gain',values['control']-values['coverage'])
    pd.DataFrame(sub).to_csv(OUT/'ablation/subgroups.csv',index=False);pd.DataFrame(period).to_csv(OUT/'ablation/periods.csv',index=False)
if __name__=='__main__':main()
