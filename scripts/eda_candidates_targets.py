"""Explore concordance of the two official Train outcomes; never fit a predictor.

Joining outcomes is for descriptive EDA only. Neither outcome is made a feature
for the other task. No Test file, raw text output or private cache is used.
"""
from pathlib import Path
import hashlib
import json
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.utils.paths import task_files


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root=None, output_dir=None):
    root = Path(root or ROOT)
    out = Path(output_dir or root/'reports/eda_candidates/targets')
    out.mkdir(parents=True, exist_ok=True)
    paths = {task: task_files(task)['train'] for task in ['regression', 'classification']}
    hashes = {k: sha(p) for k,p in paths.items()}
    assert hashes == {'regression':'756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082',
                      'classification':'875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c'}
    reg, cls = pd.read_csv(paths['regression']), pd.read_csv(paths['classification'])
    common = sorted(set(reg) & set(cls))
    assert len(common) == 18 and reg[common].equals(cls[common])
    assert reg.record_id.is_unique and cls.record_id.is_unique
    assert reg.record_id.equals(cls.record_id)
    frame = reg.drop(columns='signal_stream').merge(cls[['record_id','artifact_signal']],
                on='record_id', how='inner', validate='one_to_one', sort=False)
    assert len(frame) == 50608
    y0 = frame.loc[frame.artifact_signal.eq(0), 'threat_rank']
    y1 = frame.loc[frame.artifact_signal.eq(1), 'threat_rank']
    summary = frame.groupby('artifact_signal').threat_rank.agg(
        n='size', mean='mean', median='median', sd='std',
        p10=lambda s:s.quantile(.1), p25=lambda s:s.quantile(.25),
        p75=lambda s:s.quantile(.75), p90=lambda s:s.quantile(.9))
    summary.to_csv(out/'by_artifact.csv')
    # Five-point bands are fixed descriptions of the supplied 0–100 scale,
    # not thresholds optimized for target separation.
    frame['rank_band'] = pd.cut(frame.threat_rank, np.arange(0,101,5), include_lowest=True)
    bands = frame.groupby('rank_band', observed=True).artifact_signal.agg(n='size', positives='sum', rate='mean')
    bands['midpoint'] = np.arange(2.5,100,5)
    bands.to_csv(out/'rank_bands.csv')
    thresholds = []
    for high in [2*100/3, 75., 80., 90.]:
        g = frame[frame.threat_rank >= high]
        count = int(g.artifact_signal.eq(0).sum())
        thresholds.append({'threshold':high,'n_high':len(g),'no_observed_artifact':count,
                           'share_within_high':count/len(g), 'share_of_all_records':count/len(frame)})
    for low in [10.,20.,25.,100/3]:
        g = frame[frame.threat_rank <= low]
        count = int(g.artifact_signal.eq(1).sum())
        thresholds.append({'threshold':low,'n_low':len(g),'observed_artifact':count,
                           'share_within_low':count/len(g), 'share_of_all_records':count/len(frame)})
    pd.DataFrame(thresholds).to_csv(out/'threshold_sensitivity.csv', index=False)
    cycle = []
    for c,g in frame.groupby('cycle_code'):
        a = g.loc[g.artifact_signal.eq(1),'threat_rank']
        b = g.loc[g.artifact_signal.eq(0),'threat_rank']
        high = g[g.threat_rank >= 75]
        cycle.append({'cycle':int(c),'n':len(g),'n_class0':len(b),'n_class1':len(a),
                      'mean_difference':float(a.mean()-b.mean()),
                      'median_difference':float(a.median()-b.median()),
                      'high_n':len(high),'high_no_artifact':int(high.artifact_signal.eq(0).sum()),
                      'high_no_artifact_share':float(high.artifact_signal.eq(0).mean())})
    cycles = pd.DataFrame(cycle); cycles.to_csv(out/'cycle_sensitivity.csv', index=False)
    # Leave-one-cycle-out checks deterministically use every cycle, no favorable subset.
    leave_one = []
    for c in sorted(frame.cycle_code.unique()):
        g = frame[frame.cycle_code.ne(c)]
        d = g.groupby('artifact_signal').threat_rank.mean()
        hi = g[g.threat_rank >= 75]
        leave_one.append({'excluded_cycle':int(c),'mean_difference':float(d.loc[1]-d.loc[0]),
                          'high_no_artifact_share':float(hi.artifact_signal.eq(0).mean())})
    pd.DataFrame(leave_one).to_csv(out/'leave_one_cycle_out.csv', index=False)
    # Category check: support for both classes >=30, chosen using observed label
    # counts (not target means); this is outcome-conditioned support, not label-blind.
    # This is descriptive; pattern_code cannot identify a causal adjustment set.
    frame['pattern_group'] = frame.pattern_code.fillna('<MISSING>')
    counts = frame.groupby(['pattern_group','artifact_signal']).size().unstack(fill_value=0)
    supported = counts[(counts[0]>=30)&(counts[1]>=30)]
    sub = frame[frame.pattern_group.isin(supported.index)]
    pattern_rows = []
    for group,g in sub.groupby('pattern_group'):
        means = g.groupby('artifact_signal').threat_rank.mean()
        pattern_rows.append({'pattern_group':group,'n':len(g),
                             'n0':int(g.artifact_signal.eq(0).sum()),'n1':int(g.artifact_signal.eq(1).sum()),
                             'mean_difference':float(means.loc[1]-means.loc[0])})
    patterns = pd.DataFrame(pattern_rows).sort_values('n', ascending=False)
    patterns.to_csv(out/'pattern_sensitivity.csv', index=False)
    weights = patterns.n / patterns.n.sum()
    standardized = float(np.dot(weights, patterns.mean_difference))
    # Counterexamples use an explicitly declared descriptive top/bottom quartile,
    # not high-risk clinical/security decisions or a causal label.
    high = frame.threat_rank.ge(75)
    low = frame.threat_rank.le(25)
    stats = {'candidate_id':'F_target_concordance', 'input_sha256':hashes,
             'n':len(frame),'join':{'type':'one_to_one','shared_features_equal':True,'shared_feature_count':18,
                                  'test_used':False,'other_target_used_for_model':False},
             'groups':summary.reset_index().to_dict('records'),
             'mean_difference':float(y1.mean()-y0.mean()),
             'point_biserial':float(frame.artifact_signal.corr(frame.threat_rank)),
             'probability_superiority':float(mannwhitneyu(y1,y0,method='asymptotic').statistic/(len(y1)*len(y0))),
             'high_quartile':{'n':int(high.sum()),'no_observed_artifact':int((high&frame.artifact_signal.eq(0)).sum()),
                              'no_observed_artifact_share':float(frame.loc[high,'artifact_signal'].eq(0).mean())},
             'low_quartile':{'n':int(low.sum()),'observed_artifact':int((low&frame.artifact_signal.eq(1)).sum()),
                             'observed_artifact_share':float(frame.loc[low,'artifact_signal'].eq(1).mean())},
             'cycle_mean_difference_range':[float(cycles.mean_difference.min()),float(cycles.mean_difference.max())],
             'cycle_positive_differences':int(cycles.mean_difference.gt(0).sum()),
             'cycle_high_no_artifact_share_range':[float(cycles.high_no_artifact_share.min()),float(cycles.high_no_artifact_share.max())],
             'leave_one_cycle_out_mean_difference_range':[min(z['mean_difference']for z in leave_one),max(z['mean_difference']for z in leave_one)],
             'leave_one_cycle_out_high_no_artifact_share_range':[min(z['high_no_artifact_share']for z in leave_one),max(z['high_no_artifact_share']for z in leave_one)],
             'pattern_supported':{'minimum_per_class':30,'groups':len(patterns),'rows':len(sub),'coverage':len(sub)/len(frame),
                                  'positive_difference_groups':int(patterns.mean_difference.gt(0).sum()),
                                  'group_size_weighted_mean_gap':standardized,
                                  'support_uses_observed_class_counts':True},
             'threshold_sensitivity':thresholds,
             'limitations':['Two observed outcomes; association does not identify a mechanism or cause.',
                'artifact_signal=0 means not observed, not proven absent.',
                'threat_rank has observed within-cycle percentile-rank geometry; the official source formula is unknown. It is not an observed real-world probability.',
                'Outcome-defined quadrants are descriptive; neither target is a predictor for the other task.',
                'Exploratory reused Train; sensitivity checks are not independent statistical validation.',
                'Anonymized patterns cannot rule out unmeasured composition or observation quality.']}
    (out/'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for label,values,color in [('0: tidak teramati',y0,'#276b86'),('1: teramati',y1,'#b45d36')]:
        sorted_values=np.sort(values)
        axes[0].plot(sorted_values,np.arange(1,len(values)+1)/len(values),label=label,color=color)
    axes[0].set(title='Kedua distribusi target tetap tumpang tindih',xlabel='threat_rank (0–100)',ylabel='Proporsi kumulatif dalam kelas',xlim=(0,100),ylim=(0,1))
    axes[0].legend(title='artifact_signal',fontsize=9)
    axes[1].plot(bands.midpoint,100*bands.rate,'-o',color='#276b86',markersize=4)
    axes[1].set(title='Indikasi artefak lebih sering pada rank tinggi',xlabel='Titik tengah bin threat_rank (lebar 5)',ylabel='artifact_signal=1 (%) dalam bin',xlim=(0,100),ylim=(0,100))
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('EDA F · 50.608 Train; label kedua task disejajarkan berdasarkan ID',fontsize=12)
    fig.tight_layout();fig.savefig(out/'target_concordance.png',dpi=160,bbox_inches='tight');plt.close(fig)
    table=pd.crosstab(pd.cut(frame.threat_rank,[0,25,50,75,np.inf],right=False),frame.artifact_signal)
    table.index=['0–<25','25–<50','50–<75','75–100'];table.to_csv(out/'quartile_crosstab.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    bottom=np.zeros(4)
    for label,color in [(0,'#276b86'),(1,'#b45d36')]:
        share=100*table[label]/table.sum(axis=1)
        axes[0].bar(table.index,share,bottom=bottom,color=color,label=f'artifact_signal={label}')
        for i,(n,s,b) in enumerate(zip(table[label],share,bottom)):
            if s>5: axes[0].text(i,b+s/2,f'{int(n):,}\n{s:.1f}%',ha='center',va='center',color='white',fontsize=9)
        bottom+=share
    axes[0].set(title='Asosiasi bukan kesetaraan kedua outcome',ylabel='Persentase dalam bin threat_rank',xlabel='Bin deskriptif threat_rank',ylim=(0,100))
    axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=2,fontsize=8)
    axes[1].scatter(cycles.cycle,100*cycles.high_no_artifact_share,color='#276b86',s=28)
    axes[1].axhline(100*stats['high_quartile']['no_observed_artifact_share'],color='#b45d36',ls='--',label='Pooled')
    axes[1].set(title='Tidak bergantung pada satu cycle',xlabel='Cycle (pemeriksaan sensitivitas)',ylabel='Tidak teramati, di antara rank ≥75 (%)',ylim=(0,100));axes[1].legend()
    fig.tight_layout();fig.savefig(out/'target_discordance.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print('F target concordance:',len(frame),'rows;',round(stats['mean_difference'],4),'mean difference;',round(100*stats['high_quartile']['no_observed_artifact_share'],2),'% high-rank without observed artifact')
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);a=p.parse_args();run(output_dir=a.output_dir)
