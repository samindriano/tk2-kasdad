"""Build and execute Samuel's Classification Round1 experiment notebook."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = json.loads((ROOT/'reports/classification_round1/results.json').read_text(encoding='utf-8'))
    primary = results['aggregate']['primary']
    fixed = primary['fixed_0.5']; selected = primary['inner_selected']
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md(f'''# Samuel — Classification Experimental Round 1

KASDAD TP2 · A06 KasdudLovers · Samuel

## tl;dr

**Structured + ordinary TF-IDF Logistic Regression (C2-B)** memimpin primary pada threshold 0,50: Macro F1 **{fixed['C2_B']['metrics']['macro_f1']:.4f}**. Text-only memberi **{fixed['C2_A']['metrics']['macro_f1']:.4f}**, dibanding **{fixed['C1_A']['metrics']['macro_f1']:.4f}** untuk LR structured. Inner-selected threshold tidak otomatis meningkatkan hasil: hybrid primary menjadi **{selected['C2_B']['metrics']['macro_f1']:.4f}**.

Text sangat membantu saat stream tersedia, tetapi hybrid masih kalah dari structured-only pada stream kosong. H12 serta threshold stability tetap perlu diperhatikan. Ini kandidat sementara untuk review, **belum model final**.

## Context & Methods

Target resmi `artifact_signal`: 1 = indikasi artefak eksploit publik teramati; 0 = tidak teramati. Metric utama **Macro F1**, dua kelas selalu berurutan [0,1]. Target ini berbeda dari entry_friction TP1; cutoff dan recipe TP1 tidak disalin. Pertanyaan EDA formal Samuel tetap Regression.

### Key Assumptions

Cycle terurut mewakili disclosure yang makin baru; kalender absolut dan timing rinci fitur lain tetap provisional. Impact/EF resmi diizinkan pengguna. Semua evaluasi adalah development validation. H12 memakai ulang record primary dan bukan tambahan bukti independen. Leaderboard publik tidak dipakai memilih model/threshold.

Protokol dibekukan sebelum fit: 7 recipe × 8 window; enam outer histories berbeda, 42 outer fits +36 inner fits. W1/H12_E dan W4/H12_L memakai fit/threshold yang sama. Tidak ada official-test prediction atau all-Train final fit.'''),
    code('''from pathlib import Path
import sys,json,io,contextlib
import pandas as pd
from IPython.display import display,Image

ROOT=Path.cwd().resolve()
while not (ROOT/'src/utils/paths.py').is_file() and ROOT!=ROOT.parent: ROOT=ROOT.parent
if not (ROOT/'src/utils/paths.py').is_file(): raise FileNotFoundError('Open inside the TP2 repository.')
sys.path.insert(0,str(ROOT))
from src.models.classification_evaluation import verify_freeze,load_protocol,run_classification
from src.models.classification_reporting import build_classification_report
from scripts.verify_classification_round1 import verify_results
freeze=verify_freeze(); protocol=load_protocol()
path=ROOT/'reports/classification_round1/results.json'
ready=False
if path.exists():
    old=json.loads(path.read_text(encoding='utf-8'))
    artifact=ROOT/'experiments/runs'/old['run_id']
    ready=old['status']=='CLASSIFICATION_ROUND1_COMPLETE_REVIEW_NO_FINAL_MODEL' and all(
        (artifact/f"fit{e}__{r['id']}.npz").is_file() and (artifact/f"fit{e}__{r['id']}.json").is_file()
        for e in old['unique_outer_histories'] for r in protocol['recipes'])
if not ready:
    print('Rebuilding missing frozen development cache; no search or Test prediction.')
    with contextlib.redirect_stdout(io.StringIO()): run_classification()
results=json.loads(path.read_text(encoding='utf-8'))
verification=verify_results()
paths=build_classification_report()
review=json.loads((ROOT/'reports/classification_round1/data_review.json').read_text(encoding='utf-8'))
P=results['aggregate']['primary']; H=results['aggregate']['secondary']
models=[r['id'] for r in protocol['recipes']]
FIG=ROOT/'reports/classification_round1/figures'
print(results['status'],results['total_evaluation_jobs'],'evaluation jobs')
print('Freeze:',freeze['signature'])
print('Independent sklearn reconciliation:',verification['status'])'''),
    md('''## Data — classification support

Train 50.608 rows dengan 33.038 class 0 dan 17.570 class 1 (34,72%). Prevalence berubah non-monotonik antarperiode, sehingga cutoff yang cocok pada history sebelumnya belum tentu cocok pada cycle berikutnya. Ringkasan ini modeling support, bukan EDA formal baru; schema dan raw tetap memakai audit lama.'''),
    code('''display(pd.DataFrame(review['by_cycle']['periods'])[['cycles_inclusive','n','positive_rate']])
display(pd.DataFrame(review['by_entry_friction']['overall'])[['entry_friction','n','positive_rate']])
display(pd.DataFrame([{'class':k,**v} for k,v in review['signal_stream']['train']['by_target_class'].items()])
        [['class','rows','missing_or_blank_rows','missing_or_blank_rate','present_rows']])
display(Image(filename=str(FIG/'classification_target_prevalence_by_cycle.png')))'''),
    md('''## Results — fixed versus inner-selected

LR memakai C=1, liblinear, class_weight=None; HGB 200 iterations/15 leaves/learning rate 0,05, tanpa early stopping. TF-IDF word unigram+bigram dibatasi 20.000 fitur, min_df=3, max_df=0,995, sublinear TF. Semua vocabularies, IDF, imputer, scaler dan category mapping fit-only. Text kosong menjadi zero vector plus indicator.

Pooled F1 dihitung dari gabungan prediksi, sedangkan equal-cycle adalah rata-rata F1 setiap cycle. Jangan menyamakan kedua estimand ini.'''),
    code('''rows=[]
for m in models:
    a=P['fixed_0.5'][m]; b=P['inner_selected'][m]
    rows.append({'model':m,'fixed F1':a['metrics']['macro_f1'],'selected F1':b['metrics']['macro_f1'],
      'fixed equal-cycle':a['equal_cycle_macro_f1'],'selected equal-cycle':b['equal_cycle_macro_f1'],
      'AP':a['metrics']['average_precision'],'fixed positive rate':a['metrics']['positive_prediction_rate'],
      'selected positive rate':b['metrics']['positive_prediction_rate']})
display(pd.DataFrame(rows).round(4))'''),
    md('''**Interpretasi:** dua fitur impact/EF mempunyai ranking signal terbatas dan pada threshold 0,50 seluruh prediksi menjadi class 0. Expanded structured features memperbaiki hasil besar; text-only kemudian mengungguli structured-only, dan hybrid menambah manfaat. C0-B merupakan peringatan: pooled score dapat terlihat lebih baik meskipun beberapa window memprediksi seluruh rows ke satu kelas. Periksa equal-cycle serta prediction rate, bukan satu aggregate saja.'''),
    md('''### Semua window dan horizon panjang

W1–W6 memvalidasi cycle 56–59, 60–63, 64–67, 68–71, 72–75, 76–79. Fit mulai 47 dan berakhir tepat sebelum blok validation. H12_E memvalidasi 56–67 dari fit sampai 55; H12_L memvalidasi 68–79 dari fit sampai 67.'''),
    code('''for policy in ['fixed_0.5','inner_selected']:
    print(policy)
    frame=pd.DataFrame([{'model':j['model'],'window':j['window'],'F1':j['policies'][policy]['metrics']['macro_f1']}
                       for j in results['jobs']])
    display(frame.pivot(index='model',columns='window',values='F1')[['W1','W2','W3','W4','W5','W6','H12_E','H12_L']].round(4))
display(Image(filename=str(FIG/'classification_primary_macro_f1_heatmaps.png')))
display(Image(filename=str(FIG/'classification_primary_per_cycle_f1.png')))'''),
    md('''**Interpretasi:** text dan hybrid lebih kuat di cycle akhir daripada structured-only. Horizon 12 cycle menurunkan aggregate score dan memperkecil gain hybrid atas text-only. Pola ini tidak identik dengan Regression Round 1: regression expanded model terutama mempunyai counterexample pada H12 awal, sedangkan classification structured models mempunyai tantangan nyata pada periode akhir. Kedua task tetap menunjukkan perlunya horizon checks.

### Inner temporal threshold selection

Untuk fit berakhir E, inner fit = 47..E−4, inner validation = E−3..E. Threshold grid 0,10..0,90 step 0,01, tie terdekat 0,50 lalu lebih rendah; fallback 0,50 jika support/classes tidak cukup. Threshold dikunci, lalu model di-fit baru ke full outer history. Tidak ada outer label untuk selection/calibration.'''),
    code('''cuts=[]
for end in results['unique_outer_histories']:
    audits=json.loads((ROOT/f'reports/classification_round1/fit_audits/fit{end}.json').read_text(encoding='utf-8'))
    for m,a in audits.items(): cuts.append({'model':m,'fit end':end,'threshold':a['selection']['threshold'],'fallback':a['selection']['fallback']})
display(pd.DataFrame(cuts).pivot(index='model',columns='fit end',values='threshold'))
display(pd.DataFrame([{'model':m,'H12 fixed F1':H['fixed_0.5'][m]['metrics']['macro_f1'],
                     'H12 selected F1':H['inner_selected'][m]['metrics']['macro_f1']} for m in models]).round(4))'''),
    md('''**Interpretasi:** selected threshold memberi trade-off precision/recall dan tidak otomatis memperbaiki Macro F1. Pada hybrid primary, recall class 1 naik tetapi false positives bertambah lebih banyak; fixed 0,50 tetap kandidat pembanding penting. Threshold yang dipilih dari masa lalu bebas outer-label leakage, namun belum tentu stabil terhadap probability/prevalence shift.'''),
    md('''## Error analysis — kelas, EF dan stream

Confusion rows=actual [0,1], columns=predicted [0,1]. AP adalah average precision; undefined class metrics diberi 0 dan single-class actual subsets ditandai pada JSON. AP/balanced accuracy null jika kedua kelas tidak ada.'''),
    code('''rows=[]
for m in ['C1_A','C1_B','C2_A','C2_B']:
    for policy in ['fixed_0.5','inner_selected']:
        a=P[policy][m]['metrics']
        rows.append({'model':m,'policy':policy,'precision1':a['per_class'][1]['precision'],
            'recall1':a['per_class'][1]['recall'],'F1 class0':a['per_class'][0]['f1'],
            'F1 class1':a['per_class'][1]['f1'],'FP':a['false_positive'],'FN':a['false_negative']})
display(pd.DataFrame(rows).round(4))
display(Image(filename=str(FIG/'classification_primary_confusion_matrices.png')))'''),
    md('''### Missing stream adalah trade-off penting

Text-only memprediksi class 0 untuk seluruh stream kosong. Hybrid memperbaiki kasus itu dibanding text-only, tetapi masih kalah dari structured-only pada subset yang sama. Zero text vector tidak menjaga coefficients/intercept model gabungan tetap sama; missing indicator dan block scaling juga dapat berperan. Belum ada ablation untuk memisahkan penjelasan tersebut.

Perbandingan model di dalam subgroup memakai records yang sama. Perbandingan antar-subgroup memiliki prevalence berbeda dan tidak membuktikan penyebab kesulitan. Unknown dan missing code groups bisa overlap.'''),
    code('''rows=[]
for m in ['C1_A','C1_B','C2_A','C2_B']:
    for name in ['EF0','EF1','stream_missing','stream_present']:
        a=P['fixed_0.5'][m]['subgroups'][name]
        rows.append({'model':m,'group':name,'n':a['n'],'positive prevalence':a['positive_rate'],
                     'F1 fixed':a['macro_f1'],'recall1':a['per_class'][1]['recall']})
display(pd.DataFrame(rows).round(4))
display(Image(filename=str(FIG/'classification_primary_subgroup_f1.png')))'''),
    md('''## Probability behavior dan diagnostic history

Reliability memakai 10 bin tetap; tiap point melaporkan predicted probability dan observed positive rate. Ini diagnostic, tanpa calibration fitting. Prediksi H12 dan primary yang sudah tersedia juga dapat dibandingkan pada record sama; history era dan jumlah rows berubah bersamaan, sehingga belum merupakan atribusi kausal recency.'''),
    code('''display(pd.DataFrame([{'model':m,'Brier':P['fixed_0.5'][m]['metrics']['brier_score'],
  'mean probability':P['fixed_0.5'][m]['metrics']['mean_probability'],
  'actual prevalence':P['fixed_0.5'][m]['metrics']['positive_rate']} for m in ['C1_A','C1_B','C2_A','C2_B']]).round(4))
display(Image(filename=str(FIG/'classification_primary_reliability.png')))
paired=pd.DataFrame(verification['paired_history_diagnostics'])
display(paired[paired.model.isin(['C1_B','C2_A','C2_B'])][['model','older_fit_end','refreshed_fit_end',
 'evaluation_cycles','n_same_rows','older_fixed_f1','refreshed_fixed_f1']].round(4))'''),
    md('''## Takeaways dan titik review

Ordinary TF-IDF sudah memberi informasi prediktif tambahan; NB-weighting/stacking TP1 belum diperlukan untuk menunjukkan manfaat text. Hybrid fixed 0,50 menjadi kandidat primary sementara, dengan text-only dan structured HGB sebagai kontrol penting. Belum ada final model atau jaminan skor Kaggle.

Prioritas Round 2 yang **belum dijalankan**:

1. Pisahkan pengaruh missing indicator, shared coefficients dan bobot blok text pada missing-stream trade-off; pertimbangkan kebijakan khusus stream dengan validasi temporal yang sama.
2. Uji threshold transfer/stability dengan inner history yang lebih representatif, tetap tanpa outer-label selection.
3. Pisahkan kontribusi cycle, kategori dan numeric fields untuk memahami late/horizon degradation.
4. Setelah kontrol tersebut, baru pertimbangkan NB weighting atau representasi text lain sebagai ablation terbatas.

QA: seluruh 56 evaluation jobs/112 policy-window metrics, 672 per-cycle metrics dan 36 threshold choices diperiksa independen dari vectors. Representative refit history 67 menguji semua keluarga; cakupannya tercatat pada verification receipt. Regression Round 1, raw dan TP1 dipertahankan.

Laporan: `reports/classification_round1/FINDINGS.md`; tabel detail: `comparison_tables.md`; metode/reproduksi: `docs/CLASSIFICATION_ROUND1.md`. **STOP setelah Round 1 untuk review pengguna.** Tidak ada Classification/Regression Round 2, official-test predictions, final model atau submission.''')]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    NotebookClient(notebook, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.validate(notebook)
    path = ROOT/'notebooks/samuel/05_samuel_classification_round1.ipynb'
    nbformat.write(notebook, path)
    print('Executed', path.name)


if __name__ == '__main__': main()
