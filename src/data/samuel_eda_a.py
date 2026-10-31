"""Read-only calculations for Samuel's official token-diversity EDA.

The frozen discovery/robustness arithmetic is reused without calling its report
writers. Functions accept data frames, return derived features/aggregates, and
never load Test, fit estimators, save data, or overwrite research evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.eda_candidates_text import token_stats
from scripts.eda_a_period_robustness import contrasts, summarize
from scripts.eda_a_within_cells import _cell_table, _summarize_gate

QUESTION = ('Pada laporan dengan jumlah token yang sama, apakah keragaman token dalam '
            '"signal_stream" berkaitan dengan proporsi "artifact_signal = 1" yang teramati?')


def derive_features(train: pd.DataFrame) -> pd.DataFrame:
    """Tokenize in memory; missing/untokenizable streams have undefined TTR."""
    required={'signal_stream','artifact_signal','cycle_code'}
    if not isinstance(train,pd.DataFrame) or train.columns.has_duplicates:
        raise ValueError('Train harus DataFrame dengan nama kolom unik.')
    if not required.issubset(train.columns):
        raise ValueError(f'Kolom wajib tidak tersedia: {sorted(required-set(train.columns))}')
    target=pd.to_numeric(train.artifact_signal,errors='raise')
    cycle=pd.to_numeric(train.cycle_code,errors='raise')
    if not target.isin([0,1]).all() or not np.isfinite(cycle).all():
        raise ValueError('Target wajib binary0/1 dan cycle harus finite.')
    text=train.signal_stream
    missing=text.isna()|text.astype('string').str.strip().eq('').fillna(False)
    values=[(0,0,np.nan) if absent else token_stats(value)[:3]
            for value,absent in zip(text,missing)]
    output=pd.DataFrame(values,columns=['token_count','unique_tokens','ttr'])
    output.loc[output.token_count.eq(0),'ttr']=np.nan
    output['stream_missing']=missing.to_numpy(dtype=bool)
    output['artifact_signal']=target.to_numpy(dtype=np.int64)
    output['cycle_code']=cycle.to_numpy()
    return output


def _present(frame: pd.DataFrame) -> pd.DataFrame:
    required={'token_count','ttr','stream_missing','artifact_signal','cycle_code'}
    if not required.issubset(frame.columns):raise ValueError('Gunakan derive_features sebelum analisis.')
    valid=frame.loc[~frame.stream_missing & frame.token_count.gt(0) & frame.ttr.notna()].copy()
    if valid.empty:raise ValueError('Tidak ada stream dengan token yang dapat dianalisis.')
    return valid


def _summary(table: pd.DataFrame, denominator: int, all_rows: int) -> dict:
    s=summarize(table,denominator)
    if table.empty:
        s.update({'denominator_present':denominator,'low_n':0,'high_n':0,'compared_tail_rows':0,
                  'low_rate':None,'high_rate':None,'positive_strata':0,'negative_strata':0,
                  'zero_strata':0,'positive_support_row_share':None})
    s['all_rows']=all_rows
    s['coverage_all_rows']=s['support_rows']/all_rows if all_rows else 0.
    return s


def exact_length_comparison(frame: pd.DataFrame, min_rows=120, min_tail=20):
    valid=_present(frame)
    work=valid.rename(columns={'token_count':'length','ttr':'diversity'})
    table,excluded=contrasts(work,['length'],min_rows,min_tail)
    summary=_summary(table,len(valid),len(frame))
    summary['minimum_rows_per_length']=min_rows;summary['minimum_rows_per_tail']=min_tail
    summary['excluded_strata']=len(excluded)
    if not table.empty:table=table.rename(columns={'length':'token_count'})
    return table,summary


def period_comparison(frame: pd.DataFrame, min_rows=120, min_tail=20):
    valid=_present(frame)
    if not valid.cycle_code.between(47,79).all():raise ValueError('Periode TP2 hanya didefinisikan untuk cycle47–79.')
    work=valid.rename(columns={'token_count':'length','ttr':'diversity'})
    work['period']=pd.cut(work.cycle_code,[46,57,68,79],labels=['47–57','58–68','69–79']).astype(str)
    table,_=contrasts(work,['length','period'],min_rows,min_tail)
    pooled=_summary(table,len(valid),len(frame));periods=[]
    for period in ['47–57','58–68','69–79']:
        population=work.loc[work.period.eq(period)]
        sub=table.loc[table.period.eq(period)] if not table.empty else table
        periods.append({'period':period,**_summary(sub,len(population),len(frame))})
    return table.rename(columns={'length':'token_count'}),{'pooled':pooled,'periods':periods}


def exact_cycle_comparison(frame: pd.DataFrame, min_n=5):
    valid=_present(frame)
    if not valid.cycle_code.between(47,79).all():raise ValueError('Cycle di luar periode resmi Train TP2.')
    cells=_cell_table(valid)
    summary,cycles=_summarize_gate(cells,min_n,len(frame),len(valid),int((~frame.stream_missing).sum()))
    retained=cells.loc[cells.n.ge(min_n)&cells.distinct_ttr.ge(2)].copy()
    return retained,summary,cycles
