"""Compact research tables and two figures from verified ablation outputs."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/classification_ttr'
def main():
    d=json.loads((OUT/'results.json').read_text());a=d['aggregate'];rows=[];classes=[];subgroups=[];cycles=[]
    for j in d['jobs']:
        baseline=next(b for b in d['jobs']if b['window']==j['window']and b['model']=='C4_HGB')
        rows.append({'family':j['family'],'window':j['window'],'model':j['model'],'rows':j['metrics']['n'],
            'macro_f1':j['metrics']['macro_f1'],'delta_C4':j['metrics']['macro_f1']-baseline['metrics']['macro_f1']})
    for family,models in a.items():
        for model,r in models.items():
            for v in r['metrics']['per_class']:classes.append({'family':family,'model':model,**v})
            for group,v in r['subgroups'].items():subgroups.append({'family':family,'model':model,'subgroup':group,
                'rows':v['n'],'macro_f1':v['macro_f1'],'delta_C4':v['macro_f1']-models['C4_HGB']['subgroups'][group]['macro_f1']})
            for v in r['per_cycle']:cycles.append({'family':family,'model':model,'cycle':v['cycle'],'rows':v['n'],'macro_f1':v['macro_f1']})
    for filename,values in [('windows',rows),('per_class',classes),('subgroups',subgroups),('per_cycle',cycles)]:
        pd.DataFrame(values).to_csv(OUT/f'{filename}.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    names=['S0','S_LEN','S_TTR','S_BOTH','H0','H_LEN','H_TTR','H_BOTH','C4_HGB','R_LEN','R_TTR','R_BOTH']
    for ax,family,title in zip(axes,['primary','secondary'],['Primary temporal','Sensitivitas H12']):
        values=[a[family][n]['metrics']['macro_f1']for n in names]
        ax.scatter(values,names,s=35,color=['#627b97'if n.startswith('S')else'#168c82'if n.startswith('H')else'#b17d2b'for n in names])
        ax.axvline(a[family]['C4_HGB']['metrics']['macro_f1'],color='black',ls='--',lw=1,label='C4_HGB lama')
        ax.set_xlim(.65,.85);ax.set_xlabel('Macro F1 (threshold 0,50)');ax.set_title(title);ax.invert_yaxis();ax.legend(loc='lower left',fontsize=8);ax.grid(axis='x',alpha=.2)
    fig.savefig(OUT/'ablation_comparison.png',dpi=160);plt.close(fig)
    table=pd.DataFrame(rows);fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    order=['W1','W2','W3','W4','W5','W6','H12_E','H12_L']
    for model,marker in [('R_LEN','o'),('R_TTR','s'),('R_BOTH','^')]:
        vals=table[table.model.eq(model)].set_index('window').loc[order,'delta_C4']*100
        ax.plot(order,vals,marker=marker,label=model)
    ax.axhline(0,color='black',lw=1);ax.axvline(5.5,color='gray',ls=':');ax.set_ylabel('Perubahan Macro F1 terhadap C4 (poin persentase)')
    ax.set_title('Konsistensi temporal: fitur tambahan pada present-stream expert');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'window_deltas.png',dpi=160);plt.close(fig)
    print('Saved four diagnostics tables and two figures.')
if __name__=='__main__':main()
