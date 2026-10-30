"""Build and execute Samuel's bounded Round 1 report notebook."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md('''# Samuel — Regression Experimental Round 1

KASDAD TP2 · A06 KasdudLovers · Samuel

## Context & Methods

EDA diterima: hubungan impact–threat berbeda antarcycle meskipun marginal target hampir stabil. Round 1 menguji conditional median, bentuk linear/nonlinear, objective dan expanded structured inputs. Target resmi yang dikonfirmasi pengguna adalah **threat_rank**, indeks prioritas respons 0–100; metric **MAE**. Impact dan EF adalah predictor resmi.

Protokol dibekukan sebelum fit: enam expanding windows 4 cycle dan dua sensitivity windows 12 cycle, seed 42, lookup support 30, HGB 200 iterations / 15 leaves / learning_rate = 0.05 / minleaf 30 / L2=1, early_stopping=False, thread limit 2. Tidak ada search atau target reranking. Validation adalah development, bukan untouched holdout. Secondary memakai ulang records primary; jangan menganggapnya bukti independen.

Notebook memverifikasi freeze dan cached validation vectors. Jika cache lokal belum tersedia, 64 recipe×window jobs dibangun ulang tanpa mengganti konfigurasi. Official Test tidak diprediksi. Source terpisah agar notebook demo tetap ringkas.'''),
      code('''from pathlib import Path
import sys,json,io,contextlib
import pandas as pd
from IPython.display import display,Image

ROOT=Path.cwd().resolve()
while not (ROOT/'src/utils/paths.py').is_file() and ROOT!=ROOT.parent: ROOT=ROOT.parent
if not (ROOT/'src/utils/paths.py').is_file(): raise FileNotFoundError('Open inside the TP2 repository.')
sys.path.insert(0,str(ROOT))
from src.models.round1_evaluation import verify_freeze,load_protocol,run_round1
from src.models.round1_behavior import analyze_round1_behavior
from src.models.round1_reporting import build_round1_report
freeze=verify_freeze(); protocol=load_protocol(); path=ROOT/'reports/round1/results.json'
ready=False
if path.exists():
    old=json.loads(path.read_text(encoding='utf-8')); artifact=ROOT/'experiments/runs'/old['run_id']
    ready=old['freeze_signature']==freeze['signature'] and all((artifact/f"{j['window']}__{j['model']}.npz").is_file() for j in old['jobs'])
if not ready:
    print('Building missing frozen validation cache; no tuning.')
    with contextlib.redirect_stdout(io.StringIO()): run_round1()
results=json.loads(path.read_text(encoding='utf-8'))
behavior=analyze_round1_behavior(); report_paths=build_round1_report()
primary,secondary=results['aggregate']['primary'],results['aggregate']['secondary']
print('Freeze:',freeze['signature']); print(results['status'],results['total_jobs'],'jobs')
print('Independent metric audit:',behavior['metric_audit'])'''),
      md('''## tl;dr

R1-D memimpin primary MAE, tetapi H12_E menjadi counterexample. R1-C memberi gain kecil namun konsisten atas lookup dan squared-error HGB. Feature expansion membantu kedua EF secara pooled, sementara tail bias dan sensitivitas training history tetap ada. Kandidat terbaik sementara tidak menjadi final model.

## Data dan recipe

Primary mencakup 38.588 unique validation rows pada 24 cycle 56–79. Semua label train sudah dipakai EDA. Raw input dipertahankan dan SHA diverifikasi; source/raw/code/package identity ada dalam freeze. HGB tidak memakai internal validation untuk early stopping.'''),
      code('''display(pd.DataFrame([{'family':family,**w} for family,key in [('primary','primary_windows'),('secondary','secondary_windows')] for w in protocol[key]]))
display(pd.DataFrame([{'recipe':r['id'],'kind':r['kind'],'input':r['features'],'objective/diagnostic':r['params']} for r in protocol['recipes']]))
display(pd.DataFrame([protocol['hgb_fixed_parameters']]))'''),
      md('''**Preprocessing fit-only:** R0-A global median; R0-B memakai median exact impact jika support fit n ≥ 30, selain itu global; R0-C memakai hierarchy joint impact×EF (n ≥ 30) → impact (n ≥ 30) → EF (n ≥ 30) → global. Linear standardized OLS memakai dua fitur lengkap. Expanded parser mempertahankan numeric component, approximate/suffix/missing/parsefail flags. Numeric median dan rare-category encoder belajar pada fit fold, dengan explicit HGB categorical mask dan unknown−1. Tidak ada target encoding atau vocabulary dari validation/test.

Expanded allowlist berisi 12 raw fields; tidak ada ID, signal_stream, audit_lane, checksum_hint, pulses atau target lain. Kolom tambahan tersedia dalam supplied scoring schema dan mempunyai deskripsi, tetapi timestamp operasional rinci tetap caveat. R1-C dan R1-D memakai family/objective/parameter sama, berbeda input set saja.'''),
      md('''## Results — seluruh model dan primary windows

Primary metric adalah raw/unclipped MAE. Equal-cycle MAE memberi bobot sama ke 24 cycle; row-pooled MAE memberi bobot per observasi. R1-B-ADD adalah satu diagnostic no-interactions terarah yang ditetapkan sebelum training.'''),
      code('''display(pd.DataFrame([{'recipe':m,'n':a['raw']['n'],'MAE':a['raw']['mae'],'equal-cycle MAE':a['equal_cycle_mae'],
 'RMSE':a['raw']['rmse'],'signed error':a['raw']['signed_error']} for m,a in primary.items()]).round(4))
w=pd.DataFrame([{'model':j['model'],'window':j['window'],'MAE':j['metrics']['mae']} for j in results['jobs'] if j['family']=='primary'])
display(w.pivot(index='model',columns='window',values='MAE').round(4))
display(Image(filename=str(ROOT/'reports/round1/figures/round1_primary_window_mae_heatmap.png'),alt='All eight recipes on six primary windows'))'''),
      md('''**Interpretasi:** R0-C mengungguli R0-B pada 6/6 window dan 24/24 cycle. R1-C mengungguli R1-B serta R0-C pada seluruh 6 window / 24 cycle; gain atas lookup kecil. R1-B squared-error hanya sedikit membaik atas linear dan masih di bawah R0-C. Interaksi pada R1-B memberi gain kecil; temuan itu khusus konfigurasi squared-error ini.

R1-D menurunkan pooled primary MAE sekitar 1,80 poin dibanding R1-C, dengan 5/6 window dan 21/24 cycle membaik. W1 memburuk. Ini belum menetapkan keunggulan untuk horizon/test yang lain.'''),
      md('''## Temporal stability — horizon 12 cycle

H12_E dan H12_L memakai ulang label primary dengan history fit berbeda, sehingga dilaporkan terpisah. Plot percycle mencakup seluruh 24 cycle validation utama.'''),
      code('''h=pd.DataFrame([{'model':j['model'],'window':j['window'],'MAE':j['metrics']['mae']} for j in results['jobs'] if j['family']=='secondary'])
display(h.pivot(index='model',columns='window',values='MAE').round(4))
display(pd.DataFrame([{'model':m,'H12 pooled MAE':a['raw']['mae'],'equal-cycle MAE':a['equal_cycle_mae']} for m,a in secondary.items()]).round(4))
display(Image(filename=str(ROOT/'reports/round1/figures/round1_primary_per_cycle_mae.png'),alt='Per-cycle MAE for representative frozen recipes'))'''),
      md('''**Interpretasi:** R1-D kalah pada H12_E 17,7956 vs R1-C 16,4321. Pooled secondary R1-C 15,4779 sedikit lebih baik dari R1-D 15,5772. Feature expansion mempunyai keuntungan set-level pada primary dan horizon risk yang nyata; belum ada attribution ke satu kolom atau jaminan error test 80–91.

### Diagnostic baru tanpa recipe/fit tambahan: refresh pada record yang sama

Prediksi frozen yang sudah ada dibandingkan pada record validation identik. Parameter tetap; history dan jumlah fit rows bertambah bersamaan, sehingga bukan causal attribution recency.'''),
      code('''refresh=[v for v in behavior['refresh_same_validation_rows'] if v['model'] in ['R1_C','R1_D']]
display(pd.DataFrame(refresh)[['model','older_fit_cycles','refreshed_fit_cycles','evaluation_cycles','n_same_validation_rows',
 'older_fit_mae','refreshed_fit_mae','delta_mae_refresh_minus_older']].round(4))
print('Same-fit overlap identical:',all(v['identical_predictions'] for v in behavior['same_fit_overlap_checks']))'''),
      md('''**Interpretasi:** pada 60–63, update fit hingga 59 menurunkan R1-D 17,1342→12,9446, lebih besar daripada R1-C 15,7341→15,1531. Pada 64–67 pola serupa. Ini mendukung investigasi staleness/history, belum membedakan cycle effect, kategori, sample size atau fitur lain. W1 juga memburuk pada subset semua kode known/nonmissing; category novelty saja tidak cukup menjelaskan kegagalan. Diagnostic ini tidak menambah independent validation.'''),
      md('''## Regime, target-region dan tail behavior

Signed error=prediction−actual. Regions utama 0–100/3, 100/3–200/3, 200/3–100; tails ≤ 5 / ≥ 95 adalah diagnostic tambahan, bukan tercile TP1. Denominator dilaporkan. Mean bias global kecil tidak berarti bias subgroup kecil.'''),
      code('''rows=[]
for model in ['R1_C','R1_D']:
    for group in ['EF0','EF1','region_low','region_middle','region_high','tail_low_le5','tail_high_ge95']:
        a=primary[model]['subgroups'][group]
        rows.append({'model':model,'group':group,'n':a['n'],'MAE':a['mae'],'signed error':a['signed_error']})
display(pd.DataFrame(rows).round(4))
display(Image(filename=str(ROOT/'reports/round1/figures/round1_primary_tail_mae_bias.png'),alt='Low/high tail MAE and signed errors'))'''),
      md('''**Interpretasi:** expansion memperbaiki MAE kedua EF secara pooled, dengan gain lebih besar pada EF0. R1-D low-tail tetap overpredicted dan high-tail underpredicted meskipun global bias hampir nol. Ini prediction bias yang kini diukur pada development validation, bukan ekspektasi final/test atau pengulangan otomatis hasil TP1.'''),
      md('''## Prediction distribution dan clipping terpisah'''),
      code('''display(pd.DataFrame([{'model':m,'min':a['raw']['prediction_quantiles']['0'],'median':a['raw']['prediction_quantiles']['0.5'],
 'P99':a['raw']['prediction_quantiles']['0.99'],'max':a['raw']['prediction_quantiles']['1'],
 'below0':a['raw']['below_zero'],'above100':a['raw']['above_100'],'raw MAE':a['raw']['mae'],'clipped MAE':a['clipped']['mae']}
 for m,a in primary.items()]).round(4))
display(Image(filename=str(ROOT/'reports/round1/figures/round1_primary_actual_prediction_distributions.png'),alt='Aligned actual and raw predicted validation distributions'))'''),
      md('''**Interpretasi:** R1-C predictions terkompresi dengan maksimum sekitar 87,99. R1-D memperluas distribusi dan memperbaiki tail MAE, tetapi 72 prediksi keluar 0–100. Clipping terpisah menurunkan MAE hanya sekitar 0,0019; primary tetap raw, tanpa penggantian diam-diam.

## QA, pelajaran TP1 dan keterbatasan

Semua 64 job diulang dari fit baru: prediction arrays bit-identical dan fitted audit-state hashes identik. Recompute independen MAE/bias dari actual validation errors cocok hingga floating-point error. Numeric/category preprocessing fit-only, unknown counts cocok dengan vocabulary fit. Tidak ada label test atau all-train final refit. Rincian di `VALIDATION.md` dan `ROUND1_REPRODUCIBILITY.md`.

Pelajaran conditional median dan tails dari REG-Lite B tetap relevan, sementara support target TP2 tidak mendukung penyalinan exact-score multiclass TP1. Phase16 memotivasi regime checks, tetapi tidak ada gate, model atau label tambahan TP1 yang digunakan. Fitur expanded masih mempunyai uncertainty timing/provenance; gain set-level tidak membuktikan satu fitur penyebabnya.

## Final findings — berhenti untuk review

R1-D adalah leader primary sementara, R1-C pembanding sederhana yang sedikit lebih baik pada pooled H12. Belum ada final model. Prioritas diskusi Round 2: attribution feature-group/cycle dan history sensitivity untuk kegagalan H12_E; kemudian quantile/rank-handling atau text jika disetujui. Semua masih belum dijalankan.

Pilih bersama apakah memperdalam regression atau mulai classification sebagai milestone terpisah. **Tidak ada Round 2, classification training, hyperparameter search, ensemble atau Kaggle submission.** Laporan: `reports/round1/ROUND1_FINDINGS.md`; seluruh tabel: `COMPARISON_TABLES.md`.''')]
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(notebook,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.validate(notebook)
    path=ROOT/'notebooks/samuel/04_samuel_regression_round1.ipynb'
    nbformat.write(notebook,path)
    print('Executed',path.name)


if __name__=='__main__':main()
