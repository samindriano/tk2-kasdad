"""Build/execute the bounded Samuel Classification Round2 review notebook."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT/'reports/classification_round2/results.json').read_text(encoding='utf-8'))
    primary = result['aggregate']['primary']; secondary = result['aggregate']['secondary']
    best_p = max(primary['fixed_0.5'], key=lambda m: primary['fixed_0.5'][m]['metrics']['macro_f1'])
    best_h = max(secondary['fixed_0.5'], key=lambda m: secondary['fixed_0.5'][m]['metrics']['macro_f1'])
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md(f'''# Samuel — Classification Round 2

Hybrid Robustness · Missing Stream · Threshold Stability

## tl;dr

Pada fixed threshold 0,50, **{best_p}** memimpin primary Macro F1 **{primary['fixed_0.5'][best_p]['metrics']['macro_f1']:.4f}**; **{best_h}** memimpin pooled H12 **{secondary['fixed_0.5'][best_h]['metrics']['macro_f1']:.4f}**. C2-B original memberi **{primary['fixed_0.5']['C2_B']['metrics']['macro_f1']:.4f} / {secondary['fixed_0.5']['C2_B']['metrics']['macro_f1']:.4f}**.

Routing memperbaiki missing-stream sambil mempertahankan present predictions pada cutoff tetap. Indicator ablation dan dua text weights belum memperbaiki single hybrid secara konsisten. Threshold alternatives mempunyai trade-off; belum ada policy universal atau final model.

## Context & Methods

Target `artifact_signal`, metric **Macro F1** dengan labels [0,1]. Round 1 adalah checkpoint yang dipertahankan, termasuk source dan saved probabilities. Empat controls, empat ablations/weights, dua availability routes; bukan ensemble search. Seed42, LR C=1/liblinear, seluruh preprocessing/TF-IDF fit-only seperti Round1.

### Key Assumptions

**Adaptive development validation:** Round 2 dirancang dari outcome Round 1 pada cycle yang sama. Freeze tidak membuat validation menjadi untouched. H12 memakai ulang records primary dan bukan evidence independen. Timing rinci fitur masih provisional; tidak ada Test prediction atau leaderboard optimization. Pertanyaan EDA formal Samuel tetap Regression.

Protokol dibekukan sebelum diagnosis saved-vector dan seluruh 72 fit baru. Threshold/gate tidak dipilih dari outer labels. Source lama hanya dibaca; reports/notebook Round1 tidak dijalankan ulang.'''),
    code('''from pathlib import Path
import sys,json,io,contextlib
import pandas as pd
from IPython.display import display,Image

ROOT=Path.cwd().resolve()
while not (ROOT/'src/utils/paths.py').is_file() and ROOT!=ROOT.parent: ROOT=ROOT.parent
if not (ROOT/'src/utils/paths.py').is_file(): raise FileNotFoundError('Open inside TP2 repository.')
sys.path.insert(0,str(ROOT))
from src.models.classification_round2_evaluation import verify_freeze,load_protocol,run_round2
from src.models.classification_round2_reporting import build_round2_report
from scripts.verify_classification_round2 import verify_results
from scripts.analyze_classification_round2 import analyze_round2
freeze=verify_freeze();protocol=load_protocol();path=ROOT/'reports/classification_round2/results.json'
ready=False
if path.exists():
    old=json.loads(path.read_text(encoding='utf-8'));artifact=ROOT/'experiments/runs'/old['run_id']
    ready=old['status']=='CLASSIFICATION_ROUND2_COMPLETE_REVIEW_NO_FINAL_MODEL' and all(
        (artifact/f'fit{end}__{m}.npz').is_file() and (artifact/f'fit{end}__{m}.json').is_file()
        for end in old['unique_outer_histories'] for m in protocol['fit_recipes'])
if not ready:
    print('Rebuilding frozen Round2 cache; baseline dependencies reconstruct only in Round2 namespace if needed.')
    with contextlib.redirect_stdout(io.StringIO()):run_round2(bootstrap=True)
results=json.loads(path.read_text(encoding='utf-8'))
verification=verify_results(); behavior=analyze_round2(); report_paths=build_round2_report()
P=results['aggregate']['primary'];H=results['aggregate']['secondary']
FIG=ROOT/'reports/classification_round2/figures'
models=[r['id'] for r in protocol['recipes']]
print(results['status'],results['total_evaluation_jobs'],'jobs')
print('Freeze:',freeze['signature'])
print('Independent verification:',verification['status'])'''),
    md('''## Data dan validation

Gunakan official Train yang sama: 50.608 rows; primary validation mencakup 38.588 unique rows pada 24 cycle 56–79. Raw tidak diubah. Train/test review dan target prevalence tetap merujuk `reports/classification_round1/data_review.json`.

W1/H12_E dan W4/H12_L berbagi fit. Enam histories menghasilkan 24 control outer refits +24 variant outer fits +24 variant inner fits. Control inner probabilities dipakai ulang. Routes tidak melatih model/gate. Diagnosis pertama memakai saved OOF probabilities; fresh routed outputs harus cocok identik.'''),
    code('''display(pd.DataFrame([{'family':family,**w} for family,key in [('primary','primary_windows'),('secondary','secondary_windows')] for w in protocol[key]]))
display(pd.DataFrame([{'recipe':r['id'],'kind':r['kind'],'change':r['description']} for r in protocol['recipes']]))
print('Fresh estimator fits:',results['fresh_model_fits_this_run'])
print('Saved-vs-fresh routing check:',results['routing_diagnostic_comparison'])'''),
    md('''## Results — semua kandidat pada kontrol fixed 0,50

Pooled F1 dihitung dari seluruh prediksi gabungan, bukan rata-rata subgroup/window F1. Equal-cycle adalah mean F1 setiap cycle. AP/Brier memakai probabilitas dan tidak bergantung cutoff. H12 dilaporkan terpisah.'''),
    code('''rows=[]
for m in models:
    a=P['fixed_0.5'][m];h=H['fixed_0.5'][m]
    rows.append({'recipe':m,'primary F1':a['metrics']['macro_f1'],'equal-cycle F1':a['equal_cycle_macro_f1'],
       'H12 F1':h['metrics']['macro_f1'],'AP primary':a['metrics']['average_precision'],'Brier primary':a['metrics']['brier_score']})
display(pd.DataFrame(rows).round(4))
display(Image(filename=str(FIG/'classification_round2_fixed_pooled_comparison.png')))
display(Image(filename=str(FIG/'classification_round2_fixed_window_f1_heatmap.png')))'''),
    md('''**Interpretasi:** routing adalah hasil terkuat pada cutoff tetap. C4-LR membaik pada seluruh enam primary windows; C4-HGB membaik pada lima, dengan satu penurunan sangat kecil. Kedua route membaik pada kedua H12. Ini belum menghilangkan adaptivity/selection bias atau menetapkan final model.

## C3 — indicator versus joint fitting

C3-B menambahkan indicator ke structured LR; C3-C menghapusnya dari hybrid. Learned structured transform identik dalam setiap history. C3-B hampir tidak mengubah overall performance; menghapus indicator hybrid justru menurunkan missing-stream F1. Indicator saja bukan penjelasan utama gap hybrid terhadap structured models.'''),
    code('''display(pd.DataFrame([{'recipe':m,'primary F1':P['fixed_0.5'][m]['metrics']['macro_f1'],
    'missing F1':P['fixed_0.5'][m]['subgroups']['stream_missing']['macro_f1'],
    'present F1':P['fixed_0.5'][m]['subgroups']['stream_present']['macro_f1'],
    'missing mean p':P['fixed_0.5'][m]['subgroups']['stream_missing']['mean_probability']}
    for m in ['C1_A','C3_B','C2_B','C3_C']]).round(4))
contrasts=[v for v in behavior['within_history_missing_logit_accounting'] if v['fit_end']==67 and v['candidate']=='C2_B' and v['control']=='C3_B']
display(pd.DataFrame(contrasts)[['fit_end','prediction_cycles','n_missing','numeric_coefficient_delta_l2',
 'missing_logit_mean_delta','missing_probability_mean_delta','missing_probability_mean_absolute_change','missing_fixed_label_changes']].round(4))
display(Image(filename=str(FIG/'classification_round2_missing_component_decomposition.png')))'''),
    md('''**Interpretasi accounting:** pada history67, C3-B→C2-B mempunyai text contribution tepat nol pada missing rows, tetapi numeric/categorical coefficients, intercept dan indicator berubah. Kontribusi rata-rata saling mengompensasi; mean logit hampir sama tidak berarti probabilitas setiap record atau labelnya sama. Ini net effect joint fitting dengan indicator dan structured transformation yang dipertahankan, bukan causal effect per token. Perbandingan across-history membutuhkan kehati-hatian karena scaler/vocabulary dan prediction ranges berbeda.

## C4 — availability routing

Gate hanya membaca NaN/blank-after-strip. Present memakai C2-B; missing memakai C1-A atau C1-B. Pada cutoff 0,50, present probabilities/labels tetap identik; missing predictions sama dengan branch yang ditetapkan. Pooled F1 dievaluasi ulang dari gabungan predictions.'''),
    code('''rows=[]
for m in ['C2_B','C4_LR','C4_HGB']:
    a=P['fixed_0.5'][m]; h=H['fixed_0.5'][m]
    rows.append({'recipe':m,'primary F1':a['metrics']['macro_f1'],'H12 F1':h['metrics']['macro_f1'],
      'missing primary F1':a['subgroups']['stream_missing']['macro_f1'],
      'missing H12 F1':h['subgroups']['stream_missing']['macro_f1'],
      'present primary F1':a['subgroups']['stream_present']['macro_f1'],
      'FP primary':a['metrics']['false_positive'],'FN primary':a['metrics']['false_negative']})
display(pd.DataFrame(rows).round(4))
display(Image(filename=str(FIG/'classification_round2_missing_present_f1_and_routes.png')))'''),
    md('''**Trade-off:** C4-HGB mengurangi false positives lebih banyak, dengan sedikit tambahan false negatives dibanding hybrid. C4-LR memperbaiki keduanya dan lebih konsisten antarwindow, tetapi pooled gain lebih kecil. Jika route memakai cutoff inner miliknya sendiri, present labels/F1 dapat berubah walaupun present probabilities masih identik. Klaim present-label invariance berlaku pada perbandingan cutoff yang sama.

## C5 — text weights dan regularisasi

Bobot 0,5× dan 2× ditetapkan sebelum hasil dilihat. Semua C tetap 1. Text scaling juga mengubah L2 penalty efektif pada original TF-IDF basis sebesar 1/w²; ini bukan pure feature importance experiment.'''),
    code('''audit=json.loads((ROOT/'reports/classification_round2/fit_audits/fit67.json').read_text(encoding='utf-8'))
rows=[]
for m in ['C5_LOW','C2_B','C5_HIGH']:
    c=audit[m]['coefficients'];a=P['fixed_0.5'][m];h=H['fixed_0.5'][m]
    rows.append({'recipe':m,'text weight':c['text_block_weight'],'C':c['model_C'],
      'primary F1':a['metrics']['macro_f1'],'missing F1':a['subgroups']['stream_missing']['macro_f1'],
      'present F1':a['subgroups']['stream_present']['macro_f1'],'H12 F1':h['metrics']['macro_f1'],
      'text coef norm fit67':c['text_coefficient_l2_norm'],'effective original-basis norm':c['text_effective_original_basis_l2_norm']})
display(pd.DataFrame(rows).round(4))'''),
    md('''**Interpretasi:** 0,5× memperbaiki missing-stream tetapi merusak present signal dan kalah pada seluruh primary windows. 2× memberi mixed results: sedikit lebih rendah pada primary dan sedikit lebih tinggi pada pooled H12; kenaikan H12 tidak merata pada dua windows. Tidak ada alasan dari putaran ini untuk memperluas weight grid.

## C6 — threshold transfer

Fixed0,50; pooled-inner lama; half-shrink menuju0,50; dan equal-cycle-inner. Semuanya memakai empat cycle inner sebelum outer evaluation. Routes memilih satu cutoff dari gabungan branch inner probabilities, bukan mewarisi cutoff hybrid. Grid/tie/fallback dibekukan; tidak ada outer-label optimization atau calibration fit.'''),
    code('''focus=['C2_B','C4_LR','C4_HGB','C5_LOW','C5_HIGH']
for family,source in [('primary',P),('H12',H)]:
    print(family)
    display(pd.DataFrame([{'recipe':m,**{p:source[p][m]['metrics']['macro_f1'] for p in protocol['threshold_policies']}} for m in focus]).round(4))
example=[a for a in behavior['threshold_transfer'] if a['model']=='C2_B' and a['window']=='W2' and a['policy'] in ['fixed_0.5','inner_pooled','inner_shrink50']]
display(pd.DataFrame(example)[['policy','threshold','inner_prevalence','outer_prevalence','inner_mean_probability',
 'outer_mean_probability','positive_prediction_rate','outer_f1','class1_precision','class1_recall']].round(4))
display(Image(filename=str(FIG/'classification_round2_policy_effects.png')))'''),
    md('''**Interpretasi:** policy universal belum ditemukan. Half-shrink mengurangi sebagian kerusakan transfer tetapi tidak mengalahkan fixed untuk hybrid. C4-LR equal-cycle mencapai pooled H12 tertinggi, terutama dari H12_E; late-window gain jauh lebih kecil. Pada W2 hybrid, prevalence naik, tetapi mean probability juga naik setelah refit; cutoff lama0,34 mengalokasikan terlalu banyak positives. Ranking/AP tetap sama untuk semua policies pada model/window yang sama, sehingga perbedaan F1 policy murni berasal dari allocation, bukan model ranking baru.

## Temporal dan subgroup behavior

Per-cycle F1 dan positive prediction rates tetap perlu dibaca bersama prevalence. Routing tidak menghapus gap horizon; subgroup scores juga dipengaruhi prevalence dan support. Unknown/missing groups dapat overlap.'''),
    code('''rows=[]
for m in ['C2_B','C4_LR','C4_HGB']:
    a=P['fixed_0.5'][m]
    for group in ['EF0','EF1','stream_missing','stream_present','all_codes_known_nonmissing','any_unseen_code']:
        g=a['subgroups'][group]
        rows.append({'recipe':m,'group':group,'n':g['n'],'F1':g['macro_f1'],'precision1':g['per_class'][1]['precision'],
          'recall1':g['per_class'][1]['recall'],'FP':g['false_positive'],'FN':g['false_negative']})
display(pd.DataFrame(rows).round(4))
display(Image(filename=str(FIG/'classification_round2_fixed_cycle_f1_ppr.png')))'''),
    md('''## Takeaways — berhenti untuk review

Missing-stream performance membaik melalui fallback; single-hybrid issue belum selesai melalui indicator/weights saja. C4-HGB fixed menjadi provisional leader, C4-LR penting sebagai pembanding recall/stability, dan C2-B tetap baseline. Jangan langsung mengunci salah satu sebagai final model.

Round3 yang belum dijalankan: validasi robustness route dengan acceptance rules baru; satu past-only probability/rank-allocation transfer policy jika disetujui; diagnosis conditional ranking pada missing rows. TP1 Phase11 memotivasi joint-fitting/missing checks, Phase14 memisahkan ranking dari allocation; recipe keduanya tidak disalin.

Untuk Regression, prioritas berikutnya tetap controlled feature/history attribution R1-D dan H12 sensitivity sebelum final selection. Tidak ada Regression Round2 pada milestone ini.

QA/receipts dan pembahasan lengkap: `reports/classification_round2/FINDINGS.md`, `verification.json`, `behavior_diagnostics.json`, `comparison_tables.md`. Reproduksi: `docs/CLASSIFICATION_ROUND2.md`. **STOP setelah Round2 untuk review pengguna.** Tidak ada Round3, final model, complex ensemble, Test prediction atau Kaggle submission.''')]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    NotebookClient(notebook, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.validate(notebook)
    target = ROOT/'notebooks/samuel/06_samuel_classification_round2.ipynb'
    nbformat.write(notebook, target)
    print('Executed', target.name)


if __name__ == '__main__': main()
