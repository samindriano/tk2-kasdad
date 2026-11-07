"""Build Samuel's formal raw-data availability EDA notebook."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "samuel" / "12_samuel_formal_eda_availability.ipynb"


def md(value):
    return nbf.v4.new_markdown_cell(value.strip())


def py(value):
    return nbf.v4.new_code_cell(value.strip())


def build():
    cells = [
        md('''
# EDA Individual Samuel — Teks Kosong dan Terisi

## Pertanyaan penelitian

**Apakah persentase laporan berlabel `artifact_signal = 1` berbeda antara `signal_stream` kosong dan terisi, setelah mempertimbangkan waktu dan volume laporan?**

Kita membandingkan laporan yang teksnya kosong dengan laporan yang teksnya tersedia. Setelah itu, kita memeriksa apakah perbedaannya tetap terlihat saat waktu dan volume laporan ikut diperhatikan. Analisis memakai official Classification Train TP2. Teks kosong berarti isi `signal_stream` tidak tersedia pada record tersebut; status ini tidak menilai kualitas teks.
'''),
        md('''
## 1. Data dan definisi

`artifact_signal = 1` berarti ada indikasi artefak eksploit publik yang teramati. Nilai 0 tidak membuktikan bahwa artefak pasti tidak ada. Teks disebut **kosong (missing)** jika nilainya null, string kosong, atau hanya spasi. Selain itu disebut **terisi (present)**.

Waktu pengamatan dibagi menjadi tiga periode `cycle_code`: 47–57, 58–68, dan 69–79. Volume laporan memakai nilai `text_load` yang disediakan dataset, bukan jumlah token yang dihitung dari `signal_stream`. Analisis hanya memakai Train.

Untuk membuat perbandingan lebih seimbang, laporan dikelompokkan menurut waktu dan rentang volume. Bobot setiap kelompok mengikuti proporsinya terhadap seluruh baris yang memenuhi syarat. Bobot ini dipakai sama untuk laporan kosong dan terisi. Cara ini disebut **standardisasi**. Setiap kelompok harus memiliki sedikitnya **20 laporan kosong dan 20 laporan terisi**. Batas rentang volume dihitung tanpa memakai label target atau status kosong/terisi.

Rentang volume memakai kelompok kuantil, yaitu pembagian data berdasarkan urutan nilainya. Pembagian ini masih kasar: berada dalam rentang yang sama tidak berarti dua laporan memiliki nilai `text_load` yang persis sama.
'''),
        py(r'''
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
PROJECT_ROOT = next((p for p in candidates if (p / "src" / "utils" / "paths.py").is_file()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Jalankan notebook dari dalam repository TP2.")
sys.path.insert(0, str(PROJECT_ROOT))

from src.eda.availability import (
    add_global_quantile_bins, prepare_availability_frame,
    prevalence_table, standardized_contrast, wilson_interval,
)
from src.models.classification_round2_models import stream_missing_mask
from src.utils.paths import task_files

train_path = task_files("classification")["train"]
if train_path is None:
    raise FileNotFoundError(
        "Letakkan official Classification Train.csv di data/raw/classification/ "
        "atau atur TP2_DATA_DIR ke folder induk data."
    )
train = pd.read_csv(train_path)
analysis = prepare_availability_frame(train, stream_missing_mask(train))
assert len(analysis) == len(train)
assert set(analysis["availability"].unique()) == {"present", "missing"}
assert analysis["cycle_code"].min() == 47 and analysis["cycle_code"].max() == 79

COLORS = {"present": "#1f4e79", "missing": "#d17a22"}
plt.rcParams.update({
    "figure.dpi": 120, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})
print(
    f"Sumber: official Classification Train.csv | {len(train):,} baris × {train.shape[1]} kolom | "
    f"cycle {analysis.cycle_code.min()}–{analysis.cycle_code.max()}"
)
'''),
        md('''
## 2. Gambaran data

Tabel merangkum denominator, jumlah label positif, dan sebaran volume narasi pada setiap status availability. Angka dihitung dari Train saat notebook dijalankan.
'''),
        py(r'''
grouped = prevalence_table(analysis, ["availability"]).set_index("availability").loc[["present", "missing"]]
overview = pd.DataFrame({
    "Baris (n)": grouped["n"].astype(int),
    "artifact_signal = 1 (n)": grouped["positive"].astype(int),
    "Proporsi artifact (%)": grouped["prevalence"] * 100,
    "text_load median": analysis.groupby("availability")["text_load"].median().loc[["present", "missing"]],
    "text_load Q1": analysis.groupby("availability")["text_load"].quantile(.25).loc[["present", "missing"]],
    "text_load Q3": analysis.groupby("availability")["text_load"].quantile(.75).loc[["present", "missing"]],
})
overview.index.name = "Availability"
display(overview.round(2))
print(
    f"Total Train {len(analysis):,}; label positif {analysis.artifact_signal.sum():,} "
    f"({analysis.artifact_signal.mean():.2%}); cycle unik {analysis.cycle_code.nunique()}."
)
'''),
        md('''
## 3. Perbandingan awal: semua laporan

Kita mulai dengan membandingkan persentase label 1 pada seluruh laporan kosong dan terisi. Interval Wilson 95% pada grafik menggambarkan ketidakpastian persentase masing-masing kelompok; interval tersebut bukan interval untuk selisih kedua kelompok.
'''),
        py(r'''
marginal_rows = []
for availability in ("present", "missing"):
    part = analysis.loc[analysis.availability.eq(availability)]
    successes, n = int(part.artifact_signal.sum()), int(len(part))
    low, high = wilson_interval(successes, n)
    marginal_rows.append({
        "Ketersediaan": availability, "artifact_signal = 1 (n)": successes,
        "Denominator (n)": n, "Proporsi (%)": 100 * successes / n,
        "Wilson 95% bawah (%)": 100 * low, "Wilson 95% atas (%)": 100 * high,
    })
marginal = pd.DataFrame(marginal_rows)
marginal_indexed = marginal.set_index("Ketersediaan")
marginal_gap_pp = marginal_indexed.loc["missing", "Proporsi (%)"] - marginal_indexed.loc["present", "Proporsi (%)"]
display(marginal.round(2))
display(Markdown(
    f"Proporsi artifact pada Train adalah **{marginal_indexed.loc['present', 'Proporsi (%)']:.2f}%** "
    f"saat stream tersedia dan **{marginal_indexed.loc['missing', 'Proporsi (%)']:.2f}%** saat missing. "
    f"Selisih missing − present adalah **{marginal_gap_pp:+.2f} poin persentase (pp)**."
))

fig, ax = plt.subplots(figsize=(7.5, 4.3))
x = np.arange(len(marginal))
rates = marginal["Proporsi (%)"].to_numpy()
lower = rates - marginal["Wilson 95% bawah (%)"].to_numpy()
upper = marginal["Wilson 95% atas (%)"].to_numpy() - rates
ax.bar(x, rates, color=[COLORS[name] for name in marginal["Ketersediaan"]], width=.58, zorder=2)
ax.errorbar(x, rates, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#30343b", capsize=4, lw=1.2, zorder=3)
for i, row in marginal.iterrows():
    ax.text(i, row["Wilson 95% atas (%)"] + 1.2, f"n={int(row['Denominator (n)']):,}", ha="center", fontsize=9)
ax.set_xticks(x, marginal["Ketersediaan"])
ax.set_ylabel("Proporsi artifact_signal = 1 (%)")
ax.set_title("Proporsi marginal menurut ketersediaan signal_stream")
ax.set_ylim(0, 52)
ax.grid(axis="y", color="#d9dde3", linewidth=.7, zorder=0)
ax.text(.99, .03, "Error bars: Wilson 95% per kelompok", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#40444b")
fig.tight_layout()
plt.show()
'''),
        md('''
## 4. Perbedaan menurut periode dan cycle

Prevalence dapat berubah sepanjang waktu. Grafik kiri memperlihatkan selisih per cycle (missing − present); grafik kanan membandingkan proporsi dan denominator pada tiga periode. Selisih positif tidak berlaku pada semua periode atau cycle.
'''),
        py(r'''
period_prevalence = prevalence_table(analysis, ["period", "availability"])
period_pivot = period_prevalence.pivot(index="period", columns="availability", values="prevalence").loc[["47–57", "58–68", "69–79"]]
period_n = period_prevalence.pivot(index="period", columns="availability", values="n").loc[period_pivot.index]
period_summary = pd.DataFrame({
    "Present n": period_n["present"].astype(int),
    "Present (%)": 100 * period_pivot["present"],
    "Missing n": period_n["missing"].astype(int),
    "Missing (%)": 100 * period_pivot["missing"],
    "Gap missing − present (pp)": 100 * (period_pivot["missing"] - period_pivot["present"]),
})
display(period_summary.round(2))

cycle_prevalence = prevalence_table(analysis, ["cycle_code", "availability"])
cycle_rates = cycle_prevalence.pivot(index="cycle_code", columns="availability", values="prevalence")
cycle_n = cycle_prevalence.pivot(index="cycle_code", columns="availability", values="n")
cycle_gap = 100 * (cycle_rates["missing"] - cycle_rates["present"])
cycle_supported = cycle_n["missing"].ge(20) & cycle_n["present"].ge(20)
assert cycle_supported.all(), "Exact-cycle graph must not include unsupported groups"
positive_cycles = int((cycle_gap > 0).sum())
negative_cycles = int((cycle_gap < 0).sum())

fig, (ax_cycle, ax_period) = plt.subplots(1, 2, figsize=(12.2, 4.4), gridspec_kw={"width_ratios": [1.45, 1]})
for marker, condition, label in [
    ("o", cycle_gap >= 0, "Gap ≥ 0 pp"),
    ("x", cycle_gap < 0, "Gap < 0 pp"),
]:
    ax_cycle.scatter(cycle_gap.index[condition], cycle_gap.loc[condition], marker=marker, s=34,
                     color=COLORS["present"], label=label, linewidths=1.2)
ax_cycle.axhline(0, color="#343a40", lw=1)
ax_cycle.set_xticks(np.arange(47, 80, 3))
ax_cycle.set_xlabel("cycle_code")
ax_cycle.set_ylabel("Gap missing − present (pp)")
ax_cycle.set_title(f"Per-cycle: {positive_cycles} positif, {negative_cycles} negatif")
ax_cycle.legend(frameon=False, loc="best")
ax_cycle.grid(axis="y", color="#e1e4e8", linewidth=.7)

period_x = np.arange(len(period_pivot))
for availability in ("present", "missing"):
    values = 100 * period_pivot[availability].to_numpy()
    ax_period.plot(period_x, values, marker="o", lw=2, color=COLORS[availability], label=availability)
    for j, (value, n) in enumerate(zip(values, period_n[availability].to_numpy())):
        offset = (6, 6) if availability == "present" else (6, -14)
        ax_period.annotate(f"n={int(n):,}", (j, value), xytext=offset, textcoords="offset points",
                           fontsize=8, color=COLORS[availability])
ax_period.set_xticks(period_x, period_pivot.index)
ax_period.set_xlabel("Periode cycle")
ax_period.set_ylabel("Proporsi artifact_signal = 1 (%)")
ax_period.set_title("Periode dan availability")
ax_period.set_ylim(0, 62)
ax_period.legend(frameon=False)
ax_period.grid(axis="y", color="#e1e4e8", linewidth=.7)
fig.tight_layout()
plt.show()

display(Markdown(
    f"Gap per periode ialah **{period_summary.iloc[0]['Gap missing − present (pp)']:+.2f} pp**, "
    f"**{period_summary.iloc[1]['Gap missing − present (pp)']:+.2f} pp**, dan "
    f"**{period_summary.iloc[2]['Gap missing − present (pp)']:+.2f} pp**. "
    f"Pada cycle dengan n≥20 untuk masing-masing status, {positive_cycles}/{len(cycle_gap)} gap positif "
    f"dan {negative_cycles}/{len(cycle_gap)} negatif; rentangnya {cycle_gap.min():+.2f} sampai {cycle_gap.max():+.2f} pp."
))
'''),
        md('''
## 5. Volume narasi (text_load)

text_load berbeda menurut availability, sehingga perbandingan marginal dapat mencampur kelompok dengan komposisi volume berbeda. Histogram memakai density agar skala tidak didominasi kelompok present yang memiliki denominator jauh lebih besar.
'''),
        py(r'''
text_summary = analysis.groupby("availability")["text_load"].agg(
    n="size", minimum="min", q1=lambda x: x.quantile(.25), median="median",
    q3=lambda x: x.quantile(.75), maximum="max"
).loc[["present", "missing"]]
overlap_low = max(text_summary.loc["present", "minimum"], text_summary.loc["missing", "minimum"])
overlap_high = min(text_summary.loc["present", "maximum"], text_summary.loc["missing", "maximum"])
outside_missing = int((analysis.availability.eq("missing") & ~analysis.text_load.between(overlap_low, overlap_high)).sum())
outside_present = int((analysis.availability.eq("present") & ~analysis.text_load.between(overlap_low, overlap_high)).sum())
display(text_summary.round(2))

fig, ax = plt.subplots(figsize=(8.4, 4.4))
bins = np.linspace(analysis.text_load.min(), analysis.text_load.max(), 34)
for availability in ("present", "missing"):
    values = analysis.loc[analysis.availability.eq(availability), "text_load"]
    ax.hist(values, bins=bins, density=True, alpha=.48, color=COLORS[availability],
            label=f"{availability} (n={len(values):,})")
    ax.axvline(values.median(), color=COLORS[availability], lw=1.8, linestyle="--")
ax.axvspan(overlap_low, overlap_high, color="#78828f", alpha=.10, label="Rentang nilai yang tumpang tindih")
ax.set_xlabel("text_load")
ax.set_ylabel("Kepadatan")
ax.set_title("Distribusi text_load menurut ketersediaan stream")
ax.legend(frameon=False)
ax.grid(axis="y", color="#e1e4e8", linewidth=.7)
fig.tight_layout()
plt.show()

display(Markdown(
    f"Median text_load adalah **{text_summary.loc['present', 'median']:.2f}** saat stream tersedia dan "
    f"**{text_summary.loc['missing', 'median']:.2f}** saat missing. Rentang nilai yang sama-sama teramati "
    f"adalah **{overlap_low:.2f}–{overlap_high:.2f}**; di luarnya terdapat {outside_missing:,} baris missing "
    f"dan {outside_present:,} baris present."
))
'''),
        md('''
## 6. Membandingkan komposisi waktu dan volume yang sama

Pada perbandingan awal, kedua kelompok bisa memiliki komposisi waktu dan volume yang berbeda. Di sini kita menyamakan bobot kelompok waktu/volume untuk keduanya, lalu menghitung ulang persentase label 1. Inilah perbandingan yang distandardisasi.

Pemeriksaan pertama memakai setiap `cycle_code` secara terpisah dan mencakup seluruh baris, karena setiap cycle memiliki sedikitnya 20 laporan dari masing-masing status. Pemeriksaan berikutnya memakai tiga periode dan membagi `text_load` ke dalam 5 atau 10 kelompok kuantil. Jumlah baris yang memenuhi syarat tetap dilaporkan.
'''),
        py(r'''
cycle_standardized, cycle_cells = standardized_contrast(analysis, ["cycle_code"], min_support=20)
q5_frame, q5_edges = add_global_quantile_bins(analysis, 5)
period_q5, period_q5_cells = standardized_contrast(q5_frame, ["period", "text_load_q5"], min_support=20)
q10_frame, q10_edges = add_global_quantile_bins(analysis, 10)
period_q10, period_q10_cells = standardized_contrast(q10_frame, ["period", "text_load_q10"], min_support=20)

comparison_rows = [{
    "Kontras": "Marginal (seluruh Train)",
    "Present (%)": marginal_indexed.loc["present", "Proporsi (%)"],
    "Missing (%)": marginal_indexed.loc["missing", "Proporsi (%)"],
    "Gap missing − present (pp)": marginal_gap_pp,
    "Sel didukung": "—",
    "Baris present tercakup": int(analysis.availability.eq("present").sum()),
    "Baris missing tercakup": int(analysis.availability.eq("missing").sum()),
    "Coverage keseluruhan (%)": 100.0,
}]
for label, result in [
    ("Exact cycle", cycle_standardized),
    ("Periode × kuantil text_load (5)", period_q5),
    ("Periode × kuantil text_load (10)", period_q10),
]:
    comparison_rows.append({
        "Kontras": label,
        "Present (%)": 100 * result["standardized_prevalence"]["present"],
        "Missing (%)": 100 * result["standardized_prevalence"]["missing"],
        "Gap missing − present (pp)": result["standardized_missing_minus_present_pp"],
        "Sel didukung": f"{result['cells_supported']}/{result['cells_total']}",
        "Baris present tercakup": result["covered_rows_by_availability"]["present"],
        "Baris missing tercakup": result["covered_rows_by_availability"]["missing"],
        "Coverage keseluruhan (%)": 100 * result["coverage_all_rows"],
    })
contrasts = pd.DataFrame(comparison_rows)
display(contrasts.round(2))

support_by_group = pd.DataFrame([
    {"Kontras": "Exact cycle",
     "Coverage present (%)": 100 * cycle_standardized["coverage_by_availability"]["present"],
     "Coverage missing (%)": 100 * cycle_standardized["coverage_by_availability"]["missing"]},
    {"Kontras": "Periode × kuantil 5",
     "Coverage present (%)": 100 * period_q5["coverage_by_availability"]["present"],
     "Coverage missing (%)": 100 * period_q5["coverage_by_availability"]["missing"]},
    {"Kontras": "Periode × kuantil 10",
     "Coverage present (%)": 100 * period_q10["coverage_by_availability"]["present"],
     "Coverage missing (%)": 100 * period_q10["coverage_by_availability"]["missing"]},
])
display(support_by_group.round(2))
print(f"Minimum support: 20 baris per availability per sel. Jumlah interval kuantil: 5-bin={len(q5_edges)-1}; 10-bin={len(q10_edges)-1}.")
'''),
        py(r'''
fig, ax = plt.subplots(figsize=(9.6, 4.5))
labels = ["Marginal\nsemua baris", "Exact cycle", "Periode ×\nkuantil 5", "Periode ×\nkuantil 10"]
gaps = contrasts["Gap missing − present (pp)"].to_numpy()
colors = ["#424a54", "#1f4e79", "#d17a22", "#9a7b32"]
x = np.arange(len(gaps))
ax.axhline(0, color="#30343b", lw=1)
ax.vlines(x, 0, gaps, color=colors, lw=2.2)
ax.scatter(x, gaps, color=colors, s=55, zorder=3)
for i, row in contrasts.iterrows():
    value = row["Gap missing − present (pp)"]
    ax.annotate(f"{value:+.2f} pp", (i, value),
                xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylabel("Gap missing − present (pp)")
ax.set_title("Kontras artifact setelah menyamakan komposisi cell")
ax.grid(axis="y", color="#e1e4e8", linewidth=.7)
ax.text(.01, .02, "Standardisasi deskriptif; tanpa interval kontras atau klaim signifikansi.",
        transform=ax.transAxes, fontsize=8, color="#454a50")
fig.tight_layout()
plt.show()
'''),
        md('''
## 7. Cara membaca hasil dan batasannya

Selisih dihitung sebagai **persentase pada laporan kosong dikurangi persentase pada laporan terisi**. Nilai positif berarti label 1 lebih sering teramati pada kelompok kosong; nilai negatif berarti lebih sering pada kelompok terisi. Satuan **poin persentase (pp)** menyatakan selisih dua persentase.

Perbandingan yang disesuaikan hanya menyamakan kelompok waktu/volume yang kita tetapkan. Hasilnya menunjukkan hubungan pada data, bukan bukti sebab-akibat.
'''),
        py(r'''
q10_unsupported = period_q10_cells.loc[~period_q10_cells.supported]
display(Markdown(
    f"**Interpretasi utama.** Secara marginal, proporsi label positif pada stream missing lebih tinggi "
    f"{marginal_gap_pp:+.2f} pp. Setelah menyamakan distribusi exact cycle, gap menjadi "
    f"{cycle_standardized['standardized_missing_minus_present_pp']:+.2f} pp dengan coverage "
    f"{cycle_standardized['coverage_all_rows']:.2%}. Setelah menyamakan periode dan kuantil volume, gap "
    f"menjadi {period_q5['standardized_missing_minus_present_pp']:+.2f} pp untuk 5 bin dan "
    f"{period_q10['standardized_missing_minus_present_pp']:+.2f} pp untuk 10 bin "
    f"(coverage 10-bin {period_q10['coverage_all_rows']:.2%}; {period_q10['cells_supported']}/"
    f"{period_q10['cells_total']} sel didukung). Arah asosiasi deskriptif sensitif pada komposisi waktu "
    f"dan volume; ini bukan bukti bahwa missingness atau volume menyebabkan label."
))
display(Markdown(
    f"**Support.** Minimum per sel adalah n≥20 untuk present dan missing. Exact cycle mencakup "
    f"{cycle_standardized['supported_rows_total']:,}/{len(analysis):,} baris; periode×5 mencakup "
    f"{period_q5['supported_rows_total']:,}/{len(analysis):,}; periode×10 mencakup "
    f"{period_q10['supported_rows_total']:,}/{len(analysis):,}. Ada {len(q10_unsupported)} sel 10-bin "
    f"yang tidak didukung. Denominator 10-bin yang tercakup adalah "
    f"{period_q10['covered_rows_by_availability']['missing']:,}/"
    f"{period_q10['total_rows_by_availability']['missing']:,} missing dan "
    f"{period_q10['covered_rows_by_availability']['present']:,}/"
    f"{period_q10['total_rows_by_availability']['present']:,} present."
))
display(Markdown(
    f"Rentang text_load yang sama-sama teramati adalah {overlap_low:.2f}–{overlap_high:.2f}; ada "
    f"{outside_missing:,} baris missing dan {outside_present:,} present di luar rentang itu. Karena "
    f"standardisasi memakai bins kasar, dukungan dalam suatu bin tidak berarti setiap nilai text_load "
    f"memiliki pasangan pembanding tepat."
))
display(Markdown(
    "**Batas interpretasi.** Hasil ini adalah asosiasi pada Train; ia tidak mengidentifikasi proses "
    "pengumpulan, makna teks anonim, kualitas teks, atau sebab-akibat. Ia tidak menghilangkan seluruh "
    "confounding, tidak menjamin pola serupa pada official Test, dan tidak menguji selisih dengan inferensi "
    "statistik. Stream kosong tidak menyumbang isi teks. Semua cycle Train telah dipakai dalam pengembangan "
    "eksperimen sehingga hasil ini bukan konfirmasi independen."
))
display(Markdown(
    "**Kaitan dengan Classification.** Diagnosis model terdahulu secara terpisah menemukan hybrid C2_B "
    "lebih lemah pada stream missing (Macro F1 0.710427), sedangkan structured HGB C1_B mencapai 0.781712; "
    "C4_HGB kemudian memakai routing availability. EDA raw ini dibuat setelah diagnosis model, jadi ia "
    "memberi konteks komposisi availability, bukan penyebab atau bukti tunggal bahwa routing meningkatkan skor."
))
'''),
        md('''
## 8. Kesimpulan

Secara keseluruhan, label `artifact_signal = 1` lebih sering teramati pada laporan dengan teks kosong: **38,13%**, dibanding **34,47%** pada laporan dengan teks terisi. Selisihnya **+3,66 poin persentase**. Jika hanya komposisi cycle yang disamakan, selisihnya tetap positif, yaitu **+3,96 poin persentase**.

Namun, setelah komposisi periode dan kelompok volume laporan disamakan, arahnya berbalik: persentase pada kelompok kosong sekitar **2,2–2,3 poin persentase lebih rendah**. Jadi, hasil perbandingan bergantung pada komposisi laporan yang dibandingkan. Kesimpulan sederhana dari seluruh data saja belum cukup untuk menggambarkan polanya.

Penyesuaian volume masih memakai rentang yang kasar; analisis 10 kelompok mencakup **94,33%** baris. Hubungan juga berbeda antarperiode/cycle. Hasil ini tidak membuktikan bahwa teks kosong menyebabkan perubahan label dan belum memastikan pola yang sama pada official Test.
'''),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    notebook.metadata["title"] = "KASDAD TP2 — Samuel Formal EDA Availability"
    nbf.validate(notebook)
    return notebook


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build(), OUTPUT)
    print(f"Built {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

