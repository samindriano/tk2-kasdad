"""Build the candidate research notebook and aggregate comparison views.

The notebook reruns four Train-only investigations before drawing these views.
No frozen research artifact, paused final notebook or model is written here.
"""
from pathlib import Path
import argparse
import json
import textwrap

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/'notebooks/samuel/eda_candidate_discovery.ipynb'


def build_views(root=None):
    root=Path(root or ROOT); out=root/'reports/eda_candidates'
    load=lambda p:json.loads((out/p).read_text(encoding='utf-8'))
    a=load('text/stats.json'); c=load('missingness/statistics.json')
    d=load('categories/stats.json'); f=load('targets/stats.json')
    exact=a['text_diversity']['exact_token_count_sensitivity']['features'][0]
    stream=c['signal_stream_availability']
    screen=[
        {'ID':'A','arah':'Keragaman token pada panjang sama','hipotesis':'Repetisi/variasi teks berkaitan dengan outcome',
         'bukti':f"Gap artifact tertimbang {100*exact['upper_vs_lower_diversity_artifact_delta']:.2f} pp; {exact['positive_token_count_strata']}/{exact['supported_token_counts']} strata searah",
         'keputusan':'Shortlist; kandidat utama untuk review'},
        {'ID':'B','arah':'text_load versus panjang stream aktual','hipotesis':'Metadata volume mencerminkan informasi lain dari ukuran teks',
         'bukti':f"Spearman text_load–token count {a['text_load_coherence']['spearman_text_load_tokens']:.4f}",
         'keputusan':'Tidak diteruskan sebagai pertanyaan utama; proxy panjang sangat kuat'},
        {'ID':'C','arah':'Ketersediaan informasi / stream kosong','hipotesis':'Availability berkaitan dengan profil outcome, bukan sekadar kelengkapan umum',
         'bukti':f"Gap rank pooled {stream['threat_mean_delta']:+.3f}; setelah pembobotan volume sekitar −1 hingga 0",
         'keputusan':'Shortlist stream availability; skor completeness umum ditolak'},
        {'ID':'D','arah':'Heterogenitas keluarga pattern_code','hipotesis':'Kategori pola mempunyai profil outcome berbeda',
         'bukti':f"115 kode n≥30 mencakup {100*d['categories']['pattern_code']['support_counts']['30']['nonmissing_category_row_coverage']:.1f}% Train",
         'keputusan':'Bukti cukup; bukan 3 shortlist utama karena makna kode anonim'},
        {'ID':'E','arah':'Konteks organisasi dan modul','hipotesis':'Perbedaan antar-ekosistem memberi konteks outcome',
         'bukti':f"Known-code coverage n≥30: org {100*d['categories']['org_code']['support_counts']['30']['nonmissing_category_row_coverage']:.1f}%; module {100*d['categories']['module_code']['support_counts']['30']['nonmissing_category_row_coverage']:.1f}%",
         'keputusan':'Dieliminasi: banyak singleton, coverage lemah, semantik anonim'},
        {'ID':'F','arah':'Keselarasan kedua outcome TP2','hipotesis':'Indikasi artefak sejalan dengan prioritas, tetapi tidak saling menggantikan',
         'bukti':f"{f['high_quartile']['no_observed_artifact']:,}/{f['high_quartile']['n']:,} rank≥75 berlabel0 ({100*f['high_quartile']['no_observed_artifact_share']:.2f}%)",
         'keputusan':'Shortlist; paling mudah dijelaskan, insight lebih deskriptif'}]
    overlaps=[]
    for row in screen:
        overlaps.append({'ID':row['ID'],
            'Rousan': 'Target sama, fitur utama teks; tanpa EF/impact' if row['ID'] in ['A','B'] else 'Bukan EF/impact→artifact',
            'Abhi':'Bukan scope_pressure→threat; tidak memakai impact sebagai kontrol',
            'Rizky':'Bukan telemetry×module_spread→artifact; module_code ≠ module_spread' if row['ID']=='E' else 'Bukan telemetry/module_spread; keduanya bukan fitur utama/kontrol',
            'Samuel lama':'Tidak meneliti impact–threat antarperiode; cycle hanya sensitivity check',
            'putusan':'Berbeda substantif; tidak mengubah pembagian resmi'})
    comparisons=[
        {'kandidat':'A · keragaman token','insight':'Asosiasi tetap pada panjang persis sama',
         'dukungan':'71 panjang; 37.681 support rows, 18.365 baris dibandingkan',
         'robustness':'80/120/200 minimum support; 3 ukuran keragaman; arah beragam',
         'demo':'Definisi sederhana, tetapi perlu menjelaskan ties dan pembobotan',
         'kelemahan':'Tidak memberi makna token; exact-length×cycle hanya 5 cycle',
         'relevansi':'Konteks Classification berbasis teks, tanpa klaim gain model disebabkan EDA'},
        {'kandidat':'C · stream availability','insight':'Gap marginal rank mengecil setelah komposisi volume',
         'dukungan':'3.404 kosong, 47.204 tersedia; 906 kosong di luar rentang tersedia',
         'robustness':'5/10/20 bin, common range, cycle, pattern, availability lain',
         'demo':'Menarik untuk confounding; butuh menjelaskan common support',
         'kelemahan':'Tanda sesudah pembobotan sensitif bin; bukan penjelasan kausal',
         'relevansi':'Konteks informasi yang hilang, terpisah dari kesalahan model'},
        {'kandidat':'F · dua outcome','insight':'Asosiasi positif tidak berarti outcome dapat dipertukarkan',
         'dukungan':'50.608 verified pairs; 5.029/12.643 rank tinggi berlabel0',
         'robustness':'33 cycle, leave-one-cycle-out, ambang rank, common pattern',
         'demo':'Paling intuitif: ECDF dan komposisi outcome',
         'kelemahan':'Asosiasi dasarnya relatif diharapkan; definisi high/low deskriptif',
         'relevansi':'Menjelaskan perbedaan dua task, bukan fitur silang untuk model'},
        {'kandidat':'EDA lama · impact–threat temporal','insight':'Marginal stabil menyembunyikan perubahan kondisional',
         'dukungan':'50.608 Train; common impact×EF 48.630 (96,09%)',
         'robustness':'33 cycle, periodisasi 2/3/4, composition, tails, cluster sensitivity',
         'demo':'Lebih banyak konsep: korelasi, standardisasi, rank geometry',
         'kelemahan':'Perlu waktu menjelaskan nonmonotonisitas dan batas temporal',
         'relevansi':'Kuat untuk generalisasi Regression; sudah selesai dan tetap valid'}]
    recommendation={
        'status':'RECOMMENDED_CANDIDATE_ONLY_USER_SELECTION_PENDING',
        'candidate':'A',
        'question':'Pada laporan dengan jumlah token yang sama, apakah keragaman token dalam signal_stream berkaitan dengan proporsi artifact_signal yang teramati?',
        'answer':f"Pada common support yang diuji, proporsi tertimbang meningkat dari {100*exact['average_low_tail_artifact_rate']:.2f}% menjadi {100*exact['average_high_tail_artifact_rate']:.2f}% (gap {100*exact['upper_vs_lower_diversity_artifact_delta']:.2f} pp); efek tidak seragam pada seluruh panjang.",
        'why':'Pertanyaan lebih fokus, fitur mudah didefinisikan, bukti EDA berdiri sendiri, dan berbeda dari anggota lain. Bukan dipilih dari leaderboard.',
        'replacement_worthwhile':'Layak bila tujuan Samuel ialah demo yang lebih fokus pada karakteristik teks; bukan keunggulan statistik universal atas EDA lama. EDA lama mempunyai coverage dan robustness yang sangat kuat.',
        'official_question_changed':False,'final_notebook_preparation_resumed':False,
        'key_limit':'Temuan eksploratif; tidak ada makna token/kausalitas atau validasi independen; 6,73% teks kosong dikecualikan dan tidak semua panjang mempunyai dukungan.'}
    payload={'screening':screen,'overlap':overlaps,'comparison':comparisons,'recommendation':recommendation}
    (out/'candidate_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,key in [('candidate_screening.csv','screening'),('non_overlap.csv','overlap'),('shortlist_comparison.csv','comparison')]:
        pd.DataFrame(payload[key]).to_csv(out/name,index=False)
    folder=out/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    table=pd.read_csv(out/'text/exact_length_type_token_ratio.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    rates=100*pd.Series([exact['average_low_tail_artifact_rate'],exact['average_high_tail_artifact_rate']])
    bars=axes[0].bar(['Keragaman rendah','Keragaman tinggi'],rates,color=['#276b86','#b45d36'],width=.6)
    for bar,rate,n in zip(bars,rates,[exact['low_tail_compared_rows'],exact['high_tail_compared_rows']]):
        axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{rate:.2f}%\nn banding={n:,}',ha='center',fontsize=10)
    axes[0].set(title='Panjang persis sama, proporsi berbeda',ylabel='artifact_signal=1, tertimbang per panjang (%)',ylim=(0,60))
    axes[1].scatter(table.token_count,100*table.artifact_delta,s=15+table.n/35,color='#276b86',alpha=.65)
    axes[1].axhline(0,color='black',lw=1)
    axes[1].axhline(100*exact['upper_vs_lower_diversity_artifact_delta'],color='#b45d36',ls='--',label='Gap tertimbang')
    axes[1].set(title='64/71 panjang searah; ada counterexample',xlabel='Jumlah token persis (strata n≥120)',ylabel='Gap tinggi − rendah (poin persentase)')
    axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
    fig.suptitle('A · 37.681 baris pada 71 panjang; bobot sama untuk kedua grup: n_panjang / total n_support',fontsize=11)
    fig.tight_layout();fig.savefig(folder/'A_exact_length_diversity.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4))
    sens=a['text_diversity']['exact_token_count_sensitivity']['support_threshold_sensitivity']
    ax.plot([r['min_rows_per_exact_token_count']for r in sens],[100*r['support_weighted_stratum_delta']for r in sens],'-o',color='#276b86')
    for r in sens:
        ax.annotate(f"{r['support_rows']:,} support rows",(r['min_rows_per_exact_token_count'],100*r['support_weighted_stratum_delta']),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
    ax.set(title='A · Besaran asosiasi bertahan pada tiga batas dukungan',xlabel='Minimum observasi per jumlah token persis',ylabel='Gap artifact, tinggi − rendah (pp)',ylim=(0,18),xlim=(65,215))
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(folder/'A_support_sensitivity.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return payload


def build_notebook():
    cells=[]
    def md(s):cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s):cells.append(nbformat.v4.new_code_cell(textwrap.dedent(s).strip()))
    md('''
    # Samuel — Independent EDA Discovery TP2

    **A06 KasdudLovers · SamuelIndriano · 31 Oktober 2026 · comparative research untuk review**

    Tujuan saya adalah menilai apakah ada pertanyaan EDA pengganti yang lebih fokus dan mudah
    dijelaskan, tanpa menganggap EDA lama lemah. Enam arah diperiksa dengan data, lalu A (keragaman
    token), C (availability), dan F (keselarasan outcome) diinvestigasi lebih dalam.
    **Rekomendasi: A**, dengan cakupan dan counterexample di bawah. Pertanyaan resmi belum diganti.

    Tidak ada modeling atau Kaggle submission dalam notebook ini. Keberhasilan model sebelumnya
    hanya menjadi konteks hipotesis; semua bukti baru dihitung melalui EDA tersendiri.
    ''')
    md('''
    ## Data dan rancangan

    PDF resmi TP2 (57 halaman, SHA awal `386fa635…`) mewajibkan pertanyaan EDA individual,
    eksplorasi, visualisasi, dan interpretasi (p.2); tidak mewajibkan metode canggih tertentu.
    `signal_stream` adalah token teknis anonim, dapat kosong/noisy (p.16); `text_load`
    adalah indikator volume narasi (p.17). Makna target TP2 dan metrik berasal dari konfirmasi
    kompetisi di `docs/COMPETITION_INFO.md`, bukan formula target yang ditebak dari data.

    Gunakan Python/dependensi dalam `requirements.txt`, serta kedua official Train di
    `data/raw/{regression,classification}/Train.csv`, atau `TP2_DATA_DIR` ke folder induknya.
    Test, cache eksperimen, file sementara, dan kredensial tidak dibutuhkan.
    Notebook ini menjalankan ulang empat skrip analisis dari data mentah dan menyimpan
    tabel/grafik agregat hanya di `reports/eda_candidates/`.

    Semua Train telah dilihat pada riset sebelumnya. Bins, support rules, dan kandidat dipilih
    dalam discovery; ini **eksploratif**, bukan praregistrasi atau validasi konfirmatori independen.
    Tidak ada token/kategori dipilih berdasarkan target ekstrem. Cycle dipakai sebagai
    pemeriksaan sensitivitas, bukan pertanyaan temporal impact–threat yang lama.
    ''')
    code('''
    from pathlib import Path
    import sys, json, hashlib
    from contextlib import redirect_stdout
    from io import StringIO
    import pandas as pd
    import numpy as np
    from IPython.display import display, Image
    ROOT = Path.cwd().resolve()
    while not (ROOT/'src/utils/paths.py').exists() and ROOT != ROOT.parent: ROOT = ROOT.parent
    if not (ROOT/'src/utils/paths.py').exists(): raise FileNotFoundError('Buka notebook dari dalam repository.')
    if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
    from src.utils.paths import task_files
    from scripts.eda_candidates_text import run as run_text
    from scripts.eda_candidates_missingness import run as run_missing
    from scripts.eda_candidates_categories import run as run_categories
    from scripts.eda_candidates_targets import run as run_targets
    from scripts.build_eda_candidate_discovery import build_views
    REPORT=ROOT/'reports/eda_candidates'
    pd.set_option('display.max_colwidth',100)
    pd.set_option('display.precision',5)
    def show(path): display(Image(filename=str(REPORT/path)))
    train_reg=pd.read_csv(task_files('regression')['train'])
    train_cls=pd.read_csv(task_files('classification')['train'])
    shared=sorted(set(train_reg)&set(train_cls))
    assert train_reg.shape==train_cls.shape==(50608,19)
    assert train_reg[shared].equals(train_cls[shared]) and len(shared)==18
    assert train_reg.record_id.is_unique and train_reg.record_id.equals(train_cls.record_id)
    display(pd.DataFrame({'Train':['Regression','Classification'],'baris':[len(train_reg),len(train_cls)],
                          'target':['threat_rank','artifact_signal'],'cycle':['47–79','47–79']}))
    print('Shared features identik, termasuk ID; join label one-to-one aman untuk EDA deskriptif saja.')
    ''')
    md('''
    ## Discovery A–B: teks dan volume laporan

    Token didefinisikan sebagai regex `[A-Za-z0-9]+`, lowercase. Keragaman utama adalah
    **jumlah token unik / jumlah token**; ia mengukur repetisi, bukan mutu atau makna laporan.
    Rasio ini tergantung panjang teks, sehingga membandingkan rasio mentah antarsemua laporan
    bisa menyesatkan. Saya memeriksa broad length bins, lalu jumlah token yang persis sama.
    `text_load` versus panjang aktual menjadi arah B yang berbeda, dan diuji tanpa diasumsikan menarik.
    ''')
    code('''
    with redirect_stdout(StringIO()):
        text=run_text(ROOT, REPORT/'text')
    display(pd.Series(text['availability'],name='availability'))
    display(pd.DataFrame(text['text_diversity']['per_length_quintile'])[
        ['length_bin','n','low_diversity_n','high_diversity_n','artifact_rate_low','artifact_rate_high']])
    show(Path('text/text_length_coherence.png'))
    ''')
    md('''
    **Arah B melemah:** `text_load` sangat berkorelasi dengan jumlah token (Spearman 0,9829)
    dan panjang karakter (0,9834), pada 47.204 stream tersedia. Ini konteks data yang berguna,
    tetapi pertanyaan “apakah volume berhubungan dengan panjang” terlalu dekat dengan definisinya.
    Arah A justru layak diteruskan karena outcome berbeda pada tingkat repetisi yang berbeda.
    ''')
    md('''
    ## Discovery C: informasi hilang dan kelengkapan

    Hitung missing pada tujuh field: org, module, pattern, stream, trace, org_spread,
    module_spread. `telemetry_flag` dikecualikan agar tidak mengambil inti EDA Rizky.
    Missing stream mengikuti NaN atau blank setelah strip. Skor kelengkapan umum dan stream
    kosong dibedakan; hubungan missingness dengan outcome tidak membuktikan MCAR/MAR atau sebab-akibat.
    ''')
    code('''
    missing=run_missing(ROOT, REPORT/'missingness')
    display(pd.read_csv(REPORT/'missingness/outcomes_by_missing_count.csv').head(6))
    show(Path('missingness/figures/availability_count_and_outcomes.png'))
    ''')
    md('''
    **Hasil negatif yang penting:** complete versus any-missing memberi gap mean rank hanya
    +0,384 untuk any-missing, dengan arah campuran per-cycle; prevalence artifact 35,11% versus
    34,27%. Kelompok dengan 4–5 missing sangat kecil. Skor completeness umum bukan kandidat kuat.
    Stream kosong tersendiri memberi pola lebih besar yang perlu diperiksa komposisinya, bukan
    dijadikan bukti bahwa “semakin tidak lengkap semakin berbahaya”.
    ''')
    md('''
    ## Discovery D–E: kategori pola, organisasi, dan modul

    Semua kode yang memenuhi dukungan 30/100/300 diperiksa; tidak memilih beberapa kategori
    dengan mean paling ekstrem. Missing bukan “kode yang diketahui” dan dilaporkan terpisah.
    Subtract mean outcome setiap cycle hanya menghasilkan outcome yang dicenter terhadap cycle;
    itu bukan standardisasi penuh komposisi kategori atau partial η² regresi multivariabel.
    ''')
    code('''
    categories=run_categories(ROOT, REPORT/'categories')
    support=[]
    for column, item in categories['categories'].items():
        for minimum, value in item['support_counts'].items():
            support.append({'field':column,'minimum n':minimum,'known codes':value['nonmissing_categories'],
                            'coverage':value['nonmissing_category_row_coverage']})
    display(pd.DataFrame(support))
    show(Path('categories/figures/category_code_support_coverage.png'))
    ''')
    md('''
    **D didukung data tetapi kurang mudah ditafsirkan:** 115 pattern codes dengan n≥30
    mencakup 80,2% Train; heterogenitas outcome tetap terlihat pada beberapa support cutoffs.
    Namun kode anonim tidak menjelaskan keluarga kerentanan secara domain, dan cycle-centering
    belum menyingkirkan composition confounding. D tidak masuk tiga shortlist utama.
    **E dieliminasi:** org/module sangat jarang, median ukuran kategori 1; coverage known-code
    n≥30 hanya 50,4%/23,6%. Rentang mean ekstrem tidak cukup menjadi insight.
    ''')
    md('''
    ## Discovery F: apakah kedua outcome selalu sejalan?

    Dua Train telah diverifikasi mempunyai ID dan seluruh fitur bersama yang identik.
    Saya menggabungkan **dua label yang teramati** untuk EDA deskriptif, tidak menjadikan
    target suatu task sebagai fitur task lain. Label `artifact_signal=0` berarti indikasi
    artefak tidak teramati, bukan bukti artefak tidak ada. Rank adalah skor prioritas 0–100,
    bukan probabilitas insiden. Geometri percentile per-cycle merupakan pola teramati;
    formula pembentukan label belum diketahui.
    ''')
    code('''
    targets=run_targets(ROOT, REPORT/'targets')
    display(pd.DataFrame(targets['groups']).round(4))
    show(Path('targets/target_concordance.png'))
    summary=build_views(ROOT)
    display(pd.DataFrame(summary['screening'])[['ID','arah','bukti','keputusan']])
    ''')
    md('''
    ## Shortlist A: keragaman pada jumlah token sama

    Di setiap jumlah token, gunakan **average percentile rank** keragaman; grup rendah
    ≤0,25 dan tinggi ≥0,75. Ties dipertahankan, tidak dipisahkan secara arbitrer. Panjang
    harus memiliki n≥120 dan setiap tail n≥20; panjang yang tidak memiliki variasi/dukungan
    dikeluarkan. Angka support 37.681 mencakup semua baris pada strata yang memenuhi syarat;
    hanya **9.011 rendah + 9.354 tinggi** masuk kontras. Kedua grup memakai bobot strata
    yang sama, n_panjang / total n_support, agar komposisi panjang dibandingkan setara.
    ''')
    code('''
    exact=text['text_diversity']['exact_token_count_sensitivity']
    display(pd.DataFrame(exact['features'])[
        ['feature','supported_token_counts','support_rows','low_tail_compared_rows',
         'high_tail_compared_rows','upper_vs_lower_diversity_artifact_delta','positive_token_count_strata']])
    show(Path('figures/A_exact_length_diversity.png'))
    show(Path('figures/A_support_sensitivity.png'))
    ''')
    md('''
    **Jawaban A:** prevalence artifact tertimbang 24,19%→38,39%, selisih **14,20 poin
    persentase**. Pooled prevalence tail mentah 24,35%→38,83% adalah estimand berbeda;
    jangan mencampur angka mentah dengan bobot common-length. Arah positif pada 64/71
    panjang, bukan seluruh panjang: efek per panjang berkisar −20,59 sampai +55,31 pp.
    Support mencakup 79,83% stream-present atau 74,46% semua Train; bukan semua data.

    Dengan n minimum 80/120/200, gap tertimbang +14,61/+14,20/+13,48 pp. Entropy dan inverse
    repeat-share memberi +10,05/+14,87 pp; ketiganya saling terkait, bukan replikasi independen.
    Pada broad-length-bin comparison, arah positif terlihat pada 30/33 cycle. Tetapi
    exact length×cycle hanya cukup pada 5 cycle (gate lebih longgar: n≥60/panjang dan ≥10/tail),
    sehingga tidak ada klaim kontrol panjang
    dan waktu sekaligus untuk seluruh Train. Resampling cycle mengasumsikan exchangeability,
    tidak menangani serial dependence, dan bukan confidence interval konfirmatori.
    Threat-rank tidak mempunyai stabilitas arah yang sama; rekomendasi A difokuskan pada artifact.
    Komposisi kategori/konteks lain tetap mungkin menjelaskan sebagian asosiasi.
    ''')
    md('''
    ## Shortlist C: gap missing-stream dan komposisi volume

    Tanpa stream (3.404 baris), mean/median rank 57,605/61,302; dengan stream 49,487/49,152.
    Gap pooled mean **+8,118** tidak hilang hanya dengan mengecek cycle: arah positif 32/33.
    Namun 76,7% stream kosong terkumpul pada decile `text_load` terendah.
    Berikut saya samakan distribusi bin volume melalui bobot share bin seluruh Train.
    Ini perbandingan deskriptif yang kasar, bukan identifikasi pengaruh missing text.
    ''')
    code('''
    stream=missing['signal_stream_availability']
    display(pd.read_csv(REPORT/'missingness/signal_stream_by_text_load_decile.csv'))
    show(Path('missingness/figures/text_load_confounding_check.png'))
    show(Path('missingness/figures/signal_stream_threat_ecdf.png'))
    ''')
    md('''
    **Jawaban C:** gap besar marginal mengecil setelah komposisi volume dipertimbangkan:
    +0,253/−0,887/−1,448 poin pada 5/10/20 quantile bins. Ini bukan pembalikan tanda yang
    robust: tandanya sensitif jumlah bin, sedangkan penyusutan besarnya lebih konsisten.
    Nominal support per bin 100% bukan matching sempurna: **906 baris kosong** mempunyai
    text_load di bawah minimum stream-present. Effective n grup kosong setelah weighting
    sekitar 953–1.049, lebih kecil daripada 3.404. Rentang volume yang sama menyisakan 2.498
    missing rows; joint volume×pattern min20 hanya mencakup sekitar 13% semua Train.
    Masih ada heterogenitas dan residual confounding. High-tail ≥95 hampir sama (~5,05%)
    pada dua grup; gap mean bukan bukti seluruh distribusi naik atau tail lebih berbahaya.
    Ini kandidat kuat untuk mengajarkan pentingnya komposisi, tetapi demo lebih rumit daripada A.
    ''')
    md('''
    ## Shortlist F: sejalan secara rata-rata, bukan identik

    Mean rank label1 65,995 versus label0 41,543 (gap 24,452); arah gap positif pada
    33/33 cycle. Point-biserial 0,403 mengukur asosiasi, bukan akurasi model atau kausalitas.
    Pada batas deskriptif ≥75, **5.029/12.643 (39,78%)** berlabel0. Pada ≤25,
    **987/12.638 (7,81%)** berlabel1. Batas-batas alternatif juga disimpan; cutoff ini
    bukan keputusan keamanan atau risk threshold resmi.
    ''')
    code('''
    display(pd.DataFrame(targets['threshold_sensitivity']).fillna('—'))
    show(Path('targets/target_discordance.png'))
    display(pd.Series(targets['pattern_supported'],name='pattern common support'))
    ''')
    md('''
    Pola F tidak disebabkan satu cycle atau satu kategori: leave-one-cycle-out mean gap
    24,121–24,735; 59 pattern groups dengan ≥30 baris tiap kelas juga searah, cakupan 88,28%.
    Filter ini memakai **hitungan label per kelas**, bukan target-blind. Gap 24,170 pada
    subset adalah group-size-weighted mean gap dalam satuan rank, bukan standardized effect
    size. F paling sederhana dijelaskan, tetapi insight utamanya lebih deskriptif daripada A/C.
    Tidak ada bukti arah sebab-akibat atau izin memakai label silang saat inference.
    ''')
    md('''
    ## Perbandingan, non-overlap, dan rekomendasi

    Semua kandidat diperiksa terhadap empat topik: Rousan EF/impact→artifact; Abhi
    scope_pressure→threat; Rizky telemetry×module_spread→artifact; Samuel lama
    impact–threat antarcycle. Kesamaan target diizinkan; fitur utama dan fenomenanya berbeda.
    Tidak ada kandidat yang memakai ketiga pasangan utama anggota lain sebagai fokus/kontrol.
    ''')
    code('''
    display(pd.DataFrame(summary['overlap']))
    display(pd.DataFrame(summary['comparison']))
    print('REKOMENDASI UNTUK REVIEW:',summary['recommendation']['question'])
    assert summary['recommendation']['official_question_changed'] is False
    assert summary['recommendation']['final_notebook_preparation_resumed'] is False
    assert exact['features'][0]['support_rows']==37681
    assert targets['n']==50608 and missing['overall']['n']==50608
    print('Semua analisis berasal dari dua Train resmi; tidak ada fit model, cache privat, atau upload.')
    ''')
    md('''
    **Satu rekomendasi terbaik: A**, “Pada laporan dengan jumlah token yang sama, apakah
    keragaman token dalam signal_stream berkaitan dengan proporsi artifact_signal yang teramati?”
    Grafik utama: exact-length weighted prevalence dan seluruh 71 kontras panjang;
    sensitivity support menyertai interpretasinya. Bukti tidak bergantung pada keberhasilan model teks.

    **Apakah worth it mengganti?** Layak bila prioritas Samuel adalah pertanyaan yang lebih
    fokus dan demo dengan dua konsep utama—repetisi teks dan proporsi outcome.
    Saya tidak menyatakan A lebih unggul secara statistik universal: EDA lama sudah sangat
    kuat, mempunyai common support 96,09%, periodisasi/EF/cycle robustness dan rank geometry
    yang lebih luas. A mengorbankan coverage dan tidak menjawab temporal generalization.
    Jika ingin mempertahankan pekerjaan yang sudah matang, EDA lama tetap layak.
    Pemilihan resmi tetap menunggu review Samuel, bukan dilakukan notebook ini.

    Penelitian ini belum membuktikan mekanisme kerentanan, kualitas bahasa, sebab-akibat,
    formula generator target, private leaderboard performance atau kemampuan generalisasi.
    Laporan lengkap: [Discovery](../../reports/eda_candidates/DISCOVERY_FINDINGS.md),
    [Perbandingan](../../reports/eda_candidates/CANDIDATE_COMPARISON.md),
    [Reproduksi/QA](../../reports/eda_candidates/REPRODUCIBILITY.md).
    EDA lama dan seluruh checkpoint/model/submission dipertahankan utuh.
    ''')
    nb=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'research_status':'EXPLORATORY_USER_SELECTION_PENDING'})
    nbformat.validate(nb);nbformat.write(nb,NB)
    return NB


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    build_views();build_notebook()
    if args.execute:
        from nbclient import NotebookClient
        notebook=nbformat.read(NB,as_version=4)
        NotebookClient(notebook,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbformat.write(notebook,NB)
    print(NB)
