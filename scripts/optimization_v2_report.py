"""Build the final performance readout from verified, tracked study evidence."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/optimization_v2'

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))

def main():
    old=(OUT/'FINDINGS.md').read_text(encoding='utf-8')
    audit=old.split('## Prior-experiment audit and implications\n',1)[1].split('## Completed Classification score-head studies',1)[0].strip()
    c=read('reports/optimization_v2/classification_calibration/results.json')
    s=read('reports/optimization_v2/classification_stack/results.json')
    n=read('reports/optimization_v2/classification_text/pilot_results_v2.json')
    u=read('reports/optimization_v2/classification_unigram/pilot_results.json')
    r=read('reports/optimization_v2/regression_repr/fullsix.json')
    qa=read('reports/optimization_v2/checkpoint_final_QA.json');assert qa['status']=='PASS'
    replay=read('reports/optimization_v2/replay_QA.json');assert replay['status']=='PASS'
    ledger=read('reports/optimization_v2/kaggle/regression__v2_reg_sparse_lgb25_ledger.json')
    assert ledger['status']=='COMPLETE'
    live={task:read(f'reports/optimization_v2/kaggle/{task}_preflight.json')for task in ['classification','regression']}
    cls=live['classification']['public_leaderboard'];reg=live['regression']['public_leaderboard']
    lines=['# Optimization V2 — checkpoint akhir','',
           '**Hasil:** champion Regression baru adalah 75% Ridge25 +25% LightGBM sparse-text, public MAE **11,35970** (ref57835122), membaik0,28164 atau sekitar2,42% dari11,64134. Classification tetap C2_B present + CatBoost missing pada public MacroF1 **0,82701** (ref57833406). Tidak ada challenger Classification yang lolos untuk upload.','',
           f"Leaderboard terautentikasi terakhir: Classification peringkat{cls['team']['competition_rank']} dari{cls['retrieved_rows']} tim yang dikembalikan API, gap{cls['gap_to_leader']:.5f} ke leader{cls['leader']['score']:.5f}; Regression peringkat{reg['team']['competition_rank']} dari{reg['retrieved_rows']}, gap{reg['gap_to_leader']:.5f}. Snapshot: {live['regression']['at_utc']}. Private/final rank belum diketahui.", '',
           '## Batas dan protokol','',
           'Sumber awal6bd5250. Hanya official data kompetisi; tidak ada label Test, dataset tambahan, pretrained model, solusi kelompok lain, atau probing leaderboard. Code/data/runtime/protokol dibekukan sebelum fits. Semua666 file tracked sebelum kampanye tetap byte-identical, termasuk EDA, notebook akademis, pekerjaan anggota lain dan fallback lama. Semua perubahan berada pada namespace OptimizationV2.','',
           'Primary enam expanding fits berakhir55/59/63/67/71/75 dengan empat cycle setelah cutoff; H12_E fit47–55→56–67 dan H12_L fit47–67→68–79. Primary dan H12 masing-masing38.588 observasi unik, dengan outcome yang saling digunakan ulang. Ini development adaptif, bukan untouched holdout. Calibration/stacking memakai skor out-of-fit dan label yang tersedia sebelum cutoff outer.','',
           '## Classification','',
           '| Present expert, missing tetap CatBoost | Primary enam windows | H12 pooled | Keputusan |',
           '|---|---:|---:|---|',
           f"| Champion C2_B | {c['primary']['champion']['all']['macro_f1']:.6f} | {c['H12']['champion']['all']['macro_f1']:.6f} | Retain |"]
    for name in ['calibrated_base','calibrated_nb','stack_base_nb']:
        lines.append(f"| Recent-four {name} | {c['primary'][name]['all']['macro_f1']:.6f} | {c['H12'][name]['all']['macro_f1']:.6f} | Retire |")
    for name in ['expanding_base','expanding_stack_cb']:
        lines.append(f"| {name} | {s['primary'][name]['all']['macro_f1']:.6f} | {s['H12'][name]['all']['macro_f1']:.6f} | Retire |")
    lines += ['',
        'Recent-four heads dilatih pada present rows E−3..E memakai skor model yang fit sampaiE−4. Expanding heads memakai disjoint OOF blocks52..E; structured-CatBoost score memberi pembanding nonlinear. Semua memakai LogisticRegression C1, cutoff0,50; tidak ada threshold/C search. Hanya tiga base fits baru sampai51 diperlukan, sedangkan komponen lain menggunakan cache terverifikasi. Higher AP atau tambahan score tidak otomatis memperbaiki MacroF1 di bawah temporal shift.','',
        '| Pilot representasi baru | Primary Δ vs champion, W1+W4 | H12 Δ | Keputusan |',
        '|---|---:|---:|---|',
        f"| CatBoost native-text | {n['views']['primary_W1_W4']['delta']['all_macro_f1']:+.6f} | {n['views']['H12_pooled']['delta']['all_macro_f1']:+.6f} | Retire setelah2fits |"]
    for name,g in u['continuation_gate'].items():
        assert not g['pilot_gate_pass']
        lines.append(f"| {name} TF-IDF | {g['pooled_primary_delta_macro_f1']:+.6f} | {g['pooled_h12_delta_macro_f1']:+.6f} | Retire setelah2fits |")
    lines += ['',
        'Pilot primary hanya9.064 baris W1+W4; jangan membandingkan angka pooled primary pilot langsung dengan primary enam windows. H12 tetap38.588 baris. CatBoost native-text memakai BoW+onlineNaiveBayes, dictionary fit-only, native categories, 500trees/depth6/lr0,05. W1/H12_E membaik, tetapi W4 −0,012242 dan H12_L −0,017523. Dua unigram variants mempertahankan seluruh LR/preprocessing lama, hanya menghapus bigram dan membandingkan frequency vs binary presence; keduanya kalah. Tidak ada tambahan fold atau upload untuk keluarga gagal. Missing-branch probabilities tetap persis sama di semua studi.','',
        '![Delta Classification](classification_policy_deltas.png)','',
        '## Regression','',
        '| Model | Primary MAE | Equal-cycle MAE | H12 pooled MAE | Public | Rekomendasi |',
        '|---|---:|---:|---:|---:|---|',
        '| Ridge25 awal | 11.800756 | 12.128720 | 14.252094 | 11.64134 | Fallback tetap |']
    for name,title in [('lgbm_sparse_l1','75%Ridge25 +25%sparse LGB'),('sgd_l1','75%Ridge25 +25%SGD L1')]:
        g=r['candidate_gates'][name];eq=np.mean([v[name]['mae']for v in r['per_cycle_blend25'].values()])
        lines.append(f"| {title} | {g['primary_candidate']['mae']:.6f} | {eq:.6f} | {14.252094145239672+g['pooled_H12_delta_mae']:.6f} | {'11.35970' if name=='lgbm_sparse_l1' else 'Tidak disubmit'} | {'Promosi champion riset' if name=='lgbm_sparse_l1' else 'Comparator; redundant dan fixed-budget'} |")
    lines += ['',
        'Kedua model baru memakai sparse TF-IDF langsung +structured/OHE/scaling, tanpaSVD. LightGBM memakai L1 dan parameter base yang sudah ada; SGD memakai epsilon-insensitive epsilon0, targetscale tetap, 2000iterations. Setiap kandidat mendapat fixed rank50 mapping dan fixed25% share, seluruhnya dideklarasikan sebelum fits. Standalone policies gagal pilot; blend yang lolos diteruskan dengan8fits baru dan4pilot fits reused. Tidak ada weight/rank/parameter search atau blend antarkandidat.','',
        'Sparse-LGB blend mengurangi primary MAE0,241366, menang6/6windows dan23/24cycles. H12_E/L berkurang0,116971/0,278668. Primary low/high-tail berubah−0,443755/−0,170411; primary missing sedikit naik0,016030. Worst reported tail/missing harm0,078596; high-tail H12_E/L naik0,025997/0,045109. Bias primary−0,557 vs−0,480; H12_L sedikit lebih negatif. Perbaikan tidak universal di seluruh subgroup, tetapi semua guard lolos.','',
        'SGD blend memperbaiki primary0,054916 dan H120,236619 tetapi hanya4/6windows; seluruh SGD fits mencapai batas2000iterations dengan ConvergenceWarning. Ini approximate fixed-budget L1, bukan optimizer yang diklaim konvergen. Korelasi prediksi kedua blend0,9967 dan error0,9907: runner-up tidak memberikan pembeda yang cukup untuk memakai slot kedua. Tidak ada fine tuning setelah public.','',
        '![Regression windows](regression_repr/fullsix_window_delta.png)','',
        '![Regression tails](regression_repr/fullsix_tail_missing_delta.png)','',
        'Resep aktual champion: 56,25% LGB SVD32 (rank0,50),18,75% HybridRidge (rank0,75),25% sparse-text LGB (rank0,50). Seluruh komponen di-refit pada50.608Train resmi. Output29.421Test IDs tepat/schema/order, finite[0,100]; hash CSV dan source commit3e86b17 dikunci sebelum upload. Public improvement searah hasil internal, tetapi dataset/horizon internal dan public berbeda; ini bukan bukti private MAE.','',
        '## Submission, QA dan reproduksi','',
        f"Kampanye memakai **0 Classification dan1 Regression submission**. Reg ref{ledger['ref']} COMPLETE pada11,35970. Kuota live terakhir Class{live['classification']['num_allowed_now']} /Reg{live['regression']['num_allowed_now']}; totalteam{live['classification']['num_total']}/{live['regression']['num_total']}. Limit API10/day dan deadline2026-11-08 16:55UTC diperiksa ulang, dengan satu shared slot terlindungi. Angka dapat berubah bila anggota tim submit. Upload melalui SDK/API, bukan Chrome.", '',
        'QA: protected666files PASS; recent score heads direkonstruksi independen18fits, expanding12heads sebelumnya diverifikasi; metrics/routes/past-only fingerprints semua study lolos verifier aktif. Suite akhir15tests PASS (termasuk unigram, kuota/idempotency), ditambah1native-text synthetic test yang dijalankan terpisah; warning feature-name LightGBM bersifat nonfatal dan terekam. Root merekonstruksi rank/mixture/target alignment Regression dan full-Train inference secara independen.','',
        'Ada controller/verifier corrections yang disimpan, tanpa mengganti model outputs: native v1 salah memeriksa populasi fit kontrol dan berhenti sebelum fit; v2 memperbaikinya. Reg strict HEAD equality diganti audit konten melalui versioned receipt karena commit Classification yang tidak berkaitan; semua frozen hashes tetap cocok. Verifier unigram awal memakai konvensi JSON digest yang salah; source awal dipulihkan ke hashprefit795922dd… dan corrected verifier_v2 kembali PASS. Failure receipts, sumber awal, serta amendments tetap utuh. Jangan menjalankan verifier legacy yang sengaja diarsipkan sebagai failure path.','',
        f"Fresh replay tambahan melatih ulang **3komponen dari raw CSV saja**, tanpa cache/CSVprediksi/model binary lama, dalam{replay['seconds']:.1f}detik. Seluruh fitted states dan prediction hash identik dengan manifest accepted. [Replay receipt](replay_QA.json), [checkpoint final](checkpoint_final_QA.json), [manifests/receipts](kaggle/).", '',
        'Dependencies: `python -m pip install -r experiments/optimization_v2/requirements.txt`. Data resmi tetap lokal di data/raw/{classification,regression} atau resolver TP2_DATA_DIR. Untuk mereproduksi winner tanpa derived caches:', '',
        '```powershell',
        'python scripts/optimization_v2_replay.py --output experiments/runs/optimization_v2_replay/new_run.csv --receipt experiments/runs/optimization_v2_replay/new_run.json',
        '```','',
        'Script menolak overwrite dan tidak mengunggah. Manifest/input/source hashes memeriksa resep accepted; exact equality hanya diklaim pada environment yang diperiksa. Full temporal audit memerlukan derived control vectors yang dapat dibangun dari raw melalui docs/CLASSIFICATION_ROUND1.md dan docs/REGRESSION_RESEARCH.md. Canonical frozen outputs dipertahankan; fresh research replay perlu namespace versi baru. Raw data, binaries dan prediction caches tidak masukGit.','',
        '## Prior-experiment audit and implications','',audit,'',
        '## Completed Classification score-head studies','',
        'Rincian mesin dan active verifiers ada di classification_calibration/, classification_stack/, classification_text/ dan classification_unigram/. Tabel di atas merupakan ringkasan; tidak ada official EDA/notebook yang diganti. Method references: [calibration](https://scikit-learn.org/stable/modules/calibration), [stacking](https://scikit-learn.org/stable/modules/ensemble), [CatBoost text](https://catboost.ai/docs/en/features/text-features).','',
        '## Stop point','',
        'Kampanye berhenti setelah beberapa hipotesis berbeda gagal untuk Classification dan satu challenger Regression yang kuat dikonfirmasi internal/public. Memperluas grid atau mengutak-atik bobot pada outcome/public yang sama akan menaikkan bias adaptif; belum ada dasar cukup untuk memakai slot tambahan. Ini checkpoint terbaik yang beralasan dari eksperimen ini, bukan klaim global optimum atau private-leaderboard guarantee. Model awal tetap tersedia sebagai fallback. Tidak ada SCeLE, ZIP, atau perubahan akademis.','']
    # Two separate panels prevent pilot cohorts being mistaken for full-six results.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    full_names=['Recent base','Recent NB','Recent stackNB','Expanding base','Expanding CBstack']
    studies=[(c,k)for k in ['calibrated_base','calibrated_nb','stack_base_nb']]+[(s,k)for k in ['expanding_base','expanding_stack_cb']]
    full=np.array([[100*(d['windows'][w][k]['all']['macro_f1']-d['windows'][w]['champion']['all']['macro_f1'])for w in ['W1','W2','W3','W4','W5','W6']]for d,k in studies])
    views=['W1','W4','H12_E','H12_L']
    pilot=np.array([[100*n['views'][w]['delta']['all_macro_f1']for w in views],*[ [100*u['continuation_gate'][k]['individual_view_delta_macro_f1'][w]for w in views]for k in ['unigram_counts','unigram_binary']]])
    fig,axes=plt.subplots(2,1,figsize=(10.5,7.7),layout='constrained')
    for ax,data,labels,cols,title in [(axes[0],full,full_names,['W1','W2','W3','W4','W5','W6'],'Score heads: six primary windows'),(axes[1],pilot,['Native CatBoost','Unigram TFIDF','Unigram binary'],views,'Representation pilots: two fits, separate H12 views')]:
        im=ax.imshow(data,cmap='RdYlGn',vmin=-2,vmax=2,aspect='auto');ax.set_yticks(range(len(labels)),labels);ax.set_xticks(range(len(cols)),cols);ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):ax.text(j,i,f'{data[i,j]:+.2f}',ha='center',va='center',fontsize=10)
    fig.colorbar(im,ax=axes,label='Delta Macro F1 ×100; positive is better',shrink=.85)
    fig.savefig(OUT/'classification_policy_deltas.png',dpi=160);plt.close(fig)
    (OUT/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Final report/Classification diagnostic generated.')

if __name__=='__main__':main()
