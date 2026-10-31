"""Two compact period figures; the supplement supplies one exact-cycle figure."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'reports/eda_a_robustness'
    results=json.loads((folder/'period_stats.json').read_text(encoding='utf-8'))
    out=folder/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    rules=results['rules'];x=np.arange(3);periods=['47–57','58–68','69–79']
    configs=[('primary_120_20','n strata≥120; tail≥20','#276b86','o'),
             ('sensitivity_60_10','n strata≥60; tail≥10','#b45d36','s')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for i,(key,label,color,marker)in enumerate(configs):
        rows=rules[key]['periods'];y=np.array([r['delta']for r in rows])*100
        axes[0].plot(x,y,marker=marker,color=color,label=label,ls='-'if i==0 else'--')
        bars=axes[1].bar(x+(i-.5)*.34,[r['coverage_present']*100 for r in rows],.32,color=color,label=label)
        for bar,r in zip(bars,rows):
            axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{100*r['coverage_present']:.1f}%",ha='center',fontsize=9)
        if i==0:
            for xx,yy in zip(x,y):axes[0].annotate(f'{yy:+.2f} pp',(xx,yy),xytext=(0,-18),textcoords='offset points',ha='center',color=color)
    axes[0].axhline(0,color='black',lw=1)
    axes[0].set(title='Gap positif di ketiga periode',ylabel='Artifact tinggi − rendah (poin persentase)',ylim=(-1,20))
    axes[0].legend(fontsize=9,loc='upper left')
    axes[1].set(title='Coverage denominator: stream tersedia',ylabel='Baris dalam strata yang lolos (%)',ylim=(0,100))
    axes[1].legend(fontsize=9,loc='lower right')
    for ax in axes:ax.set_xticks(x,periods);ax.set_xlabel('Periode cycle');ax.grid(axis='y',alpha=.18)
    fig.suptitle('Exact token count × periode; average-rank quartile tails; bobot n strata',fontsize=12)
    fig.tight_layout();fig.savefig(out/'period_effect_and_coverage.png',dpi=160,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for i,(key,label,color,marker)in enumerate(configs):
        rule=rules[key];rows=rule['common_length_periods'];common=rule['common_length_support']
        y=np.array([r['delta']for r in rows])*100
        axes[0].plot(x,y,marker=marker,color=color,ls='-'if i==0 else'--',
                     label=f"{common['common_lengths']} panjang bersama; cov {common['coverage_present']:.1%}")
    axes[0].axhline(0,color='black',lw=1)
    axes[0].set(title='Panjang dan bobot yang sama antarperiode',ylabel='Artifact tinggi − rendah (poin persentase)',ylim=(-1,27))
    axes[0].legend(loc='upper left',fontsize=9)
    rows=rules['primary_120_20']['periods'];shares=np.array([r['positive_support_row_share']for r in rows])*100
    axes[1].bar(x,shares,color='#276b86',label='Strata dengan gap positif')
    axes[1].bar(x,100-shares,bottom=shares,color='#b45d36',label='Strata dengan gap negatif')
    for xx,share,row in zip(x,shares,rows):
        axes[1].text(xx,share/2,f'{share:.1f}%\nbaris support',ha='center',va='center',color='white')
        axes[1].text(xx,102,f"{row['positive_strata']}+ / {row['negative_strata']}− strata",ha='center',fontsize=9)
    axes[1].set(title='Mayoritas support berada pada strata positif',ylabel='Share baris support primary (%)',ylim=(0,115))
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.30),fontsize=9,ncol=2)
    for ax in axes:ax.set_xticks(x,periods);ax.set_xlabel('Periode cycle');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out/'common_lengths_and_direction.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print('Saved 2 period figures; no statistics or historical files rewritten.')


if __name__=='__main__':main()
