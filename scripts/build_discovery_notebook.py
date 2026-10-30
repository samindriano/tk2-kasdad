"""Build and execute Samuel's exploratory question-discovery notebook."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[
      md('''# Samuel — regression EDA discovery (pertanyaan belum dikunci)

## tl;dr

Notebook ini merangkum temuan deskriptif aktual dari file lokal yang tersedia. Perbandingan TP1 bersifat opsional; jika input TP1 tidak ada, seluruh analisis TP2 tetap dijalankan. Status dan ringkasan TL;DR di bawah dibentuk dari hasil run.

Analisis ini tetap **DISCOVERY**. Pilihan pertanyaan EDA belum dikunci, dan tidak ada model, prediksi, atau submission yang dibuat.'''),
      md('''## Context & Methods

### Asumsi yang perlu dikonfirmasi

- Target dikonfirmasi oleh skema train/test/sample. Header membuktikan nama kolom; makna resmi `threat_rank` masih memerlukan data dictionary.
- Materi kuliah TP1 mendefinisikan `cycle_code` sebagai urutan disclosure, dengan nilai lebih besar lebih baru (PDF lokal, p.16). Penerapan makna itu untuk TP2 masih asumsi yang perlu dicek.
- Tiga periode dibentuk dengan membagi **nilai cycle unik yang terurut**, bukan baris atau target: 47–57, 58–68, 69–79.
- Korelasi Pearson/Spearman dan quantile merangkum baris train yang lengkap. Target test tidak digunakan.
- Target rank diuji terhadap percentile rank rata-rata **dalam setiap cycle** yang dibulatkan 4 desimal. Ini hanya diagnostik pola numerik, bukan bukti algoritme pembentukan target.
- Kelompok target rendah/tinggi memakai batas deskriptif tetap `≤5`/`≥95`; batas ini tidak dipilih dengan hasil prediksi atau validasi.
- Jika TP1 tersedia, perbandingan membandingkan concentration/support, bukan lokasi skala mentah yang tak sebanding. TP1 error metrics dari eksperimen lama bukan performa TP2.'''),
      code('''from pathlib import Path
import os,sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Image,Markdown

ROOT=Path.cwd().resolve()
while not (ROOT/'src/utils/paths.py').is_file() and ROOT!=ROOT.parent:
    ROOT=ROOT.parent
if not (ROOT/'src/utils/paths.py').is_file(): raise FileNotFoundError('Run from the TP2 repository.')
sys.path.insert(0,str(ROOT))
from src.data.discovery import run_discovery
tp1_root=Path(os.environ['TP1_DATA_DIR']) if os.environ.get('TP1_DATA_DIR') else ROOT.parent/'TK1'/'tk1-eda-3'/'data'/'raw'
summary=run_discovery(tp1_data_dir=tp1_root)
print('Status:',summary['status'])
print('TP1 comparison:',summary['tp1_target_comparison']['status'])
print('Input hashes:',{k:v['sha256'] for k,v in summary['provenance']['regression_files'].items()})'''),
      code('''tp2=summary['target_tp2']
tp1=summary['tp1_target_comparison']['tp1']
tp1_line=(f"TP1 impact_load tersedia dengan {tp1['unique']:,} nilai unik; top-1 mengambil {tp1['top_value_shares']['top_1']:.1%} baris TP1."
          if tp1 else "Perbandingan TP1 tidak tersedia; analisis dan grafik TP2 tetap lengkap.")
corr=summary['impact_threat']['overall_correlation']
display(Markdown(f"""### Ringkasan discovery aktual

TP2 memiliki **{tp2['n']:,}** target finite dari {tp2['input_rows']:,} baris input, **{tp2['unique']:,}** nilai unik, rentang `{tp2['min']:.4f}–{tp2['max']:.4f}`, dan **{tp2['integer_like_fraction_abs_tol_1e_8']:.2%}** nilai integer-like (toleransi absolut 1e-8). Korelasi seluruh train: Pearson **{corr['pearson']:.3f}**, Spearman **{corr['spearman']:.3f}**.

{tp1_line}

Hubungan yang teramati bersifat deskriptif: tidak membuktikan sebab-akibat atau error model."""))'''),
      md('''## Data

**Train:** 50.608 × 19 · **Test:** 29.421 × 18 · **Sample:** 29.421 × 2. `threat_rank` adalah satu-satunya kolom train-only dan prediksi sample. ID sample berurutan sama dengan test. Target train lengkap, tidak ada ID/row duplikat maupun overlap split. Rincian schema, missingness, dan hash file ada di `reports/audits/regression_audit.json`.'''),
      md('''## Results — profil target

Distribusi TP2 dan, bila tersedia, profil TP1 ditampilkan terpisah. Makna/units kedua target berbeda sehingga shape dan concentration dibandingkan, bukan skala mentah.'''),
      code('''profiles=summary['tp1_target_comparison']
rows=[{'target':'TP2 threat_rank',**{k:profiles['tp2'][k] for k in ('n','missing','unique','min','max','mean','sd','top_value_shares')}}]
if profiles['tp1']:
    rows.insert(0,{'target':'TP1 impact_load',**{k:profiles['tp1'][k] for k in ('n','missing','unique','min','max','mean','sd','top_value_shares')}})
else:
    print('TP1 profile omitted:',profiles['status'])
display(pd.DataFrame(rows).set_index('target'))
print('TP2 target decile occupancy:')
display(pd.DataFrame(summary['impact_threat']['target_decile_bins']))
figures=set(summary['figures'])
if 'discovery_tp1_top_scores.png' in figures:
    display(Image(filename=str(ROOT/'reports'/'eda'/'discovery_tp1_top_scores.png'),alt='TP1 impact_load most frequent values'))
else:
    print('TP1-only figure skipped: optional TP1 comparison unavailable.')
if 'discovery_tp2_target_histogram.png' in figures:
    display(Image(filename=str(ROOT/'reports'/'eda'/'discovery_tp2_target_histogram.png'),alt='TP2 threat_rank histogram'))'''),
      md('''## Results — target rank vs impact/cycle

Korelasi per cycle bervariasi; garis tiap cycle tidak mewakili confidence interval. Median dan quantiles menggambarkan sebaran aktual, bukan uncertainty of the mean. Target marginal yang tetap bisa menutupi perubahan hubungan kondisional.'''),
      code('''display(pd.DataFrame([{'period':p['period'],'cycles':f"{p['cycle_min']}–{p['cycle_max']}",'n':p['n'],
                     'Pearson':p['impact_threat_correlation']['pearson'],'Spearman':p['impact_threat_correlation']['spearman'],
                     'target median':p['target']['quantiles']['50'],'target p10':p['target']['quantiles']['10'],
                     'target p90':p['target']['quantiles']['90']} for p in summary['impact_threat']['by_cycle_period']]).round(4))
per_cycle=pd.DataFrame(summary['impact_threat']['by_cycle'])
display(per_cycle[['cycle_code','n','pearson','spearman','threat_median','threat_p10','threat_p90']].round(4).head(8))
for name,alt in [('discovery_cycle_relationship.png','Per-cycle correlations and target marginal quantiles'),
                 ('discovery_conditional_spread.png','Conditional threat spread by repeated impact and period'),
                 ('discovery_tail_by_impact_cycle.png','Low and high target shares by cycle period and impact band')]:
    if name in figures: display(Image(filename=str(ROOT/'reports'/'eda'/name),alt=alt))'''),
      md('''### Variasi pada impact_load yang sama

`impact_load` berulang persis pada sebagian besar record. Banyak tingkat impact yang sama mengandung rentang target lebar; tabel dan grafik berikut membandingkan enam nilai impact terpadat per periode dengan skala target bersama.'''),
      code('''rep=pd.DataFrame(summary['repeated_impact_values_by_period'])
top=rep[rep.period=='all'].nlargest(12,'n').impact_load
display(rep[rep.impact_load.isin(top)].sort_values(['impact_load','period'])
        [['impact_load','period','n','median','p10','p90','iqr']].round(4).head(36))
display(pd.DataFrame(summary['candidate_signal_examples']))'''),
      md('''### Pola rank per cycle dan ekor target

An exact within-cycle percentile match occurs for the full training target under the tested formula. This may reflect how the score is encoded; it does not prove target lineage. The target test labels are absent, so no target comparison or error analysis on test exists.'''),
      code('''display(summary['within_cycle_percentile_diagnostic'])
tails=pd.DataFrame(summary['conditional_extremes']['by_period_impact_band'])
display(tails[['period','impact_band','n','low_n_le_5','low_share','high_n_ge_95','high_share']].round(4))'''),
      md('''## Kandidat pertanyaan EDA

Pengukuran dan visualisasi di notebook ini menjadi bukti pendukung, sementara tiga sampai lima kandidat pertanyaan beserta bukti dan rekomendasi terbaik dicatat pada [laporan rekomendasi discovery](../../reports/eda/REGRESSION_DISCOVERY.md). Notebook ini berstatus **DISCOVERY** dan tidak mengunci pertanyaan secara mandiri.'''),
      md('''## Takeaways

TP2 `threat_rank` memiliki support yang padat, hubungan positif dengan `impact_load`, variasi kekuatan asosiasi antarcycle, dan sebaran target yang lebar pada nilai impact yang sama. Rank sangat sesuai dengan percentile cycle masing-masing, tetapi ini hanya petunjuk konstruksi numerik, bukan pembuktian lineage. Jika dibandingkan, temuan over/underprediction ekstrem TP1 hanya menjadi motivasi untuk menelaah support dan sebaran kondisional TP2; itu bukan bukti bahwa error TP1 berulang.'''),
    ]
    n=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    path=ROOT/'notebooks'/'samuel'/'02_regression_discovery.ipynb'
    NotebookClient(n,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.validate(n)
    nbformat.write(n,path)
    print('Wrote and executed',path)


if __name__=='__main__': main()
