"""Four focused, editable figures for the official EDA A notebook."""
import numpy as np
import matplotlib.pyplot as plt

BLUE='#276b86'
RUST='#b45d36'


def overview_figure(frame):
    valid=frame.loc[~frame.stream_missing & frame.token_count.gt(0)]
    fig,axes=plt.subplots(1,2,figsize=(11,3.7))
    axes[0].hist(valid.token_count,bins=np.arange(5,151,5),color=BLUE,edgecolor='white')
    axes[0].set(title='Panjang signal_stream yang tersedia',xlabel='Jumlah token per laporan',ylabel='Jumlah laporan')
    axes[1].hist(valid.ttr,bins=np.linspace(0,1,21),color=RUST,edgecolor='white')
    axes[1].set(title='Distribusi proporsi token unik',xlabel='TTR = token unik / token total',ylabel='Jumlah laporan',xlim=(0,1))
    fig.suptitle(f'{len(valid):,} stream tokenizable; stream kosong tidak diberi TTR=0',fontsize=11)
    fig.tight_layout();return fig


def primary_figure(table,summary):
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    rates=[100*summary['low_rate'],100*summary['high_rate']]
    bars=axes[0].bar(['Keragaman rendah','Keragaman tinggi'],rates,color=[BLUE,RUST],width=.6)
    for bar,rate,n in zip(bars,rates,[summary['low_n'],summary['high_n']]):
        axes[0].text(bar.get_x()+bar.get_width()/2,rate+1,f'{rate:.2f}%\nn kontras={n:,}',ha='center',fontsize=10)
    axes[0].set(title='Proporsi artifact pada panjang yang sama',ylabel='artifact_signal=1, tertimbang per panjang (%)',ylim=(0,55))
    positive=table.delta.gt(0)
    for mask,color,label,marker in [(positive,BLUE,'Gap positif','o'),(~positive,RUST,'Gap nol/negatif','x')]:
        axes[1].scatter(table.loc[mask,'token_count'],100*table.loc[mask,'delta'],color=color,label=label,marker=marker,s=28)
    axes[1].axhline(0,color='black',lw=1)
    axes[1].axhline(100*summary['delta'],color=RUST,ls='--',lw=1.2,label=f"Gap tertimbang {100*summary['delta']:.2f} pp")
    axes[1].set(title=f"{summary['positive_strata']}/{summary['strata']} panjang berarah positif",xlabel='Jumlah token persis',ylabel='Gap tinggi − rendah (poin persentase)')
    axes[1].legend(fontsize=8,loc='upper left');axes[1].grid(alpha=.18)
    fig.suptitle('Bobot panjang sama pada dua grup: n_panjang / total n_support; setiap titik = satu panjang',fontsize=10)
    fig.tight_layout();return fig


def period_figure(summary):
    rows=summary['periods'];periods=[r['period']for r in rows];x=np.arange(3)
    fig,axes=plt.subplots(1,2,figsize=(11,3.9))
    bars=axes[0].bar(x,[100*r['delta']for r in rows],color=BLUE,width=.6)
    for bar,r in zip(bars,rows):axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,f"{100*r['delta']:+.2f} pp",ha='center')
    axes[0].set(title='Arah sama, besar asosiasi berbeda',ylabel='Gap artifact tinggi − rendah (pp)',ylim=(0,20))
    bars=axes[1].bar(x,[100*r['coverage_present']for r in rows],color=RUST,width=.6)
    for bar,r in zip(bars,rows):axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.4,f"{100*r['coverage_present']:.1f}%\nn={r['support_rows']:,}",ha='center',fontsize=10)
    axes[1].set(title='Coverage pada stream tersedia tiap periode',ylabel='Baris di strata yang memenuhi support (%)',ylim=(0,100))
    for ax in axes:ax.set_xticks(x,periods);ax.set_xlabel('Periode cycle');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();return fig


def cycle_figure(cycles,summary):
    fig,ax=plt.subplots(figsize=(10.5,3.9))
    x=cycles.cycle_code;y=100*cycles.slope_per_0_1_ttr
    ax.plot(x,y,'-o',color=BLUE,markersize=4,lw=1.3)
    negative=y.lt(0)
    ax.scatter(x[negative],y[negative],color=RUST,marker='x',s=70,zorder=4,label='Cycle berarah negatif')
    ax.axhline(0,color='black',lw=1)
    ax.axhline(100*summary['common_within_cell_slope_per_0_1_ttr'],color=RUST,ls='--',label='Ringkasan within-cell seluruh cycle')
    count=summary['cycle_concentration']['positive_cycle_slopes']
    ax.set(title=f'Exact length × exact cycle: {count}/33 cycle berarah positif',xlabel='Cycle disclosure',ylabel='Koefisien deskriptif (pp per +0,10 TTR)')
    ax.legend(fontsize=9,loc='upper left');ax.grid(alpha=.15)
    fig.tight_layout();return fig
