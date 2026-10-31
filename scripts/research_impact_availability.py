"""Raw-data audit of signal_stream availability and artifact_signal composition.

Descriptive only. Reads official Classification Train, never Test or predictions.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "impact_eda" / "availability"
MIN_PER_STATE_PER_CYCLE = 20
PERIODS = [(47, 57), (58, 68), (69, 79)]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> dict:
    if n == 0:
        return {"low": None, "high": None}
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"low": float(max(0, center - half)), "high": float(min(1, center + half))}


def summarize_group(df: pd.DataFrame, label: str) -> dict:
    y = df["artifact_signal"].astype(int)
    n = int(len(y))
    positive = int(y.sum())
    interval = wilson(positive, n)
    text = df["text_load"].astype(float)
    return {
        "availability": label,
        "n": n,
        "artifact_0": int(n - positive),
        "artifact_1": positive,
        "artifact_prevalence": float(positive / n) if n else None,
        "artifact_wilson95": interval,
        "text_load_median": float(text.median()),
        "text_load_q25": float(text.quantile(.25)),
        "text_load_q75": float(text.quantile(.75)),
    }


def main() -> None:
    from src.models.classification_round2_models import stream_missing_mask
    from src.utils.paths import task_files

    train_path = task_files("classification")["train"]
    if train_path is None:
        raise FileNotFoundError("Official Classification Train not found via task_files().")
    input_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
    df = pd.read_csv(train_path)
    raw_shape = [int(df.shape[0]), int(df.shape[1])]
    raw_columns = list(df.columns)
    expected = {"record_id", "cycle_code", "signal_stream", "artifact_signal", "text_load"}
    absent = sorted(expected - set(df.columns))
    if absent:
        raise ValueError(f"Required raw columns absent: {absent}")
    if df.record_id.duplicated().any():
        raise ValueError("record_id is not unique")
    target = pd.to_numeric(df.artifact_signal, errors="raise")
    if not target.isin([0, 1]).all() or target.isna().any():
        raise ValueError("artifact_signal must be complete binary 0/1")
    miss = stream_missing_mask(df)
    if len(miss) != len(df):
        raise AssertionError("Frozen stream_missing_mask misaligned")
    df = df.copy()
    df["_availability"] = np.where(miss, "missing", "present")
    df["_artifact"] = target.astype(int)
    df["_cycle"] = pd.to_numeric(df.cycle_code, errors="raise").astype(int)

    overall = [summarize_group(df.loc[~miss], "present"), summarize_group(df.loc[miss], "missing")]
    overall_delta = overall[1]["artifact_prevalence"] - overall[0]["artifact_prevalence"]

    by_period = []
    for start, end in PERIODS:
        part = df[df._cycle.between(start, end)]
        groups = [summarize_group(part[part._availability.eq(name)], name)
                  for name in ["present", "missing"]]
        by_period.append({"period": [start, end], "n": int(len(part)), "groups": groups,
                          "missing_minus_present_pp": 100 * (groups[1]["artifact_prevalence"] - groups[0]["artifact_prevalence"])})

    cycles = []
    for cycle, part in df.groupby("_cycle", sort=True):
        groups = {name: summarize_group(part[part._availability.eq(name)], name)
                  for name in ["present", "missing"]}
        supported = all(groups[name]["n"] >= MIN_PER_STATE_PER_CYCLE for name in groups)
        delta = groups["missing"]["artifact_prevalence"] - groups["present"]["artifact_prevalence"] if all(groups[name]["n"] for name in groups) else None
        cycles.append({"cycle": int(cycle), "n": int(len(part)), "groups": groups,
                       "missing_minus_present_pp": None if delta is None else 100 * delta,
                       "supported_min20_each_state": bool(supported)})
    volume_sensitivity = make_volume_sensitivity(df)

    supported_cycles = [c for c in cycles if c["supported_min20_each_state"]]
    supported_ids = {c["cycle"] for c in supported_cycles}
    common = df[df._cycle.isin(supported_ids)]
    # Direct standardization to the cycle distribution among all rows in the
    # supported-cycle population: each group's cycle-specific rates get the
    # same pooled cycle weights N_c / sum(N_c). This is descriptive only.
    denom = len(common)
    std_present = 0.0
    std_missing = 0.0
    for c in supported_cycles:
        weight = c["n"] / denom
        std_present += weight * c["groups"]["present"]["artifact_prevalence"]
        std_missing += weight * c["groups"]["missing"]["artifact_prevalence"]
    cycle_effects = [c["missing_minus_present_pp"] for c in supported_cycles]
    signs = {"positive": sum(x > 1e-12 for x in cycle_effects),
             "negative": sum(x < -1e-12 for x in cycle_effects),
             "zero": sum(abs(x) <= 1e-12 for x in cycle_effects)}

    stats = {
        "title": "Classification signal_stream availability and artifact_signal composition",
        "method": "Raw-data descriptive analysis; no model scores, predictions, Test data, or Test labels.",
        "source": {"path_role": "official Classification Train resolved with src.utils.paths.task_files",
                   "sha256": input_sha, "rows": raw_shape[0], "columns": raw_columns,
                   "record_id_unique": True},
        "reproduction": {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "python": platform.python_version(),
                         "packages": {name: importlib.metadata.version(name)
                                      for name in ["numpy", "pandas", "matplotlib", "scikit-learn"]}},
        "definitions": {"availability": "Round2 stream_missing_mask: null, blank, or whitespace-only stream is missing",
                        "outcome": "artifact_signal in {0,1}", "uncertainty_interval": "Wilson 95%; descriptive per-group intervals, not an adjusted contrast CI",
                        "minimum_exact_cycle_support": f"minimal {MIN_PER_STATE_PER_CYCLE} baris pada masing-masing status availability",
                        "standardization": "within each supported cycle compute missing-present prevalence; weight each cycle by its share of all rows in the pooled supported-cycle population"},
        "schema_audit": {"shape": raw_shape,
                         "duplicate_record_id": int(df.record_id.duplicated().sum()),
                         "artifact_signal_missing": int(df.artifact_signal.isna().sum()),
                         "signal_stream_null": int(df.signal_stream.isna().sum()),
                         "signal_stream_blank_or_whitespace_nonnull": int(((df.signal_stream.notna()) & df.signal_stream.astype(str).str.strip().eq("")).sum()),
                         "signal_stream_missing_under_frozen_mask": int(miss.sum()),
                         "cycle_min_max": [int(df._cycle.min()), int(df._cycle.max())],
                         "official_test_opened": False},
        "overall_groups": overall,
        "overall_missing_minus_present_pp": float(100 * overall_delta),
        "periods": by_period,
        "cycles": cycles,
        "text_load_sensitivity": volume_sensitivity,
        "exact_cycle_standardization": {"minimum_n_each_state": MIN_PER_STATE_PER_CYCLE,
            "supported_cycles": len(supported_cycles), "all_cycles": len(cycles),
            "cycle_ids": sorted(supported_ids), "covered_rows": int(len(common)),
            "coverage_all_rows": float(len(common) / len(df)),
            "covered_missing_rows": int(common._availability.eq("missing").sum()),
            "coverage_missing_group": float(common._availability.eq("missing").sum() / miss.sum()),
            "covered_present_rows": int(common._availability.eq("present").sum()),
            "coverage_present_group": float(common._availability.eq("present").sum() / (~miss).sum()),
            "standardized_present_prevalence": float(std_present),
            "standardized_missing_prevalence": float(std_missing),
            "standardized_missing_minus_present_pp": float(100 * (std_missing - std_present)),
            "supported_cycle_direction_counts": signs,
            "cycle_gap_range_pp": [float(min(cycle_effects)), float(max(cycle_effects))],
            "cycle_gap_median_pp": float(np.median(cycle_effects))},
        "interpretation_limits": [
            "Associations are descriptive and do not show that missingness causes either label.",
            "Cycle standardization controls only cycle composition; it does not control text_load or the record-generation process.",
            "Wilson intervals describe each raw prevalence separately; no adjusted contrast interval or hypothesis test is claimed.",
            "Cycle and period results use development Train repeatedly examined elsewhere; they are sensitivity descriptions, not independent confirmation.",
            "Missing versus present may represent different information-collection processes; missingness is not a text-quality measure."],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_figures(overall, by_period, cycles, stats)
    write_notes(stats)
    print(json.dumps({"rows": len(df), "sha256": input_sha,
        "overall_gap_pp": stats["overall_missing_minus_present_pp"],
        "standardized_gap_pp": stats["exact_cycle_standardization"]["standardized_missing_minus_present_pp"],
        "supported_cycles": len(supported_cycles), "coverage": stats["exact_cycle_standardization"]["coverage_all_rows"]}, indent=2))


def standardize_cells(df: pd.DataFrame, keys: list[str], scheme: str, min_support: int = 20) -> dict:
    grouped = (df.groupby(keys + ["_availability"], observed=True)
                 .agg(n=("_artifact", "size"), positives=("_artifact", "sum")).reset_index())
    domains = [sorted(df[key].dropna().unique().tolist()) for key in keys]
    all_cells = pd.MultiIndex.from_product(domains, names=keys).to_frame(index=False)
    cells = []
    for _, row in all_cells.iterrows():
        dims = {key: row[key] for key in keys}
        state = {}
        for name in ["present", "missing"]:
            mask = np.ones(len(grouped), dtype=bool)
            for key in keys:
                mask &= grouped[key].eq(dims[key]).to_numpy()
            mask &= grouped["_availability"].eq(name).to_numpy()
            part = grouped.loc[mask]
            n = int(part.n.sum()) if len(part) else 0
            positives = int(part.positives.sum()) if len(part) else 0
            state[name] = {"n": n, "artifact_1": positives, "prevalence": positives / n if n else None}
        cells.append({**dims, **state,
                      "supported": state["present"]["n"] >= min_support and state["missing"]["n"] >= min_support})
    supported = [cell for cell in cells if cell["supported"]]
    coverage = {name: sum(cell[name]["n"] for cell in supported) for name in ["present", "missing"]}
    pooled_n = sum(coverage.values())
    rates = {"present": 0.0, "missing": 0.0}
    for cell in supported:
        weight = (cell["present"]["n"] + cell["missing"]["n"]) / pooled_n
        for name in rates:
            rates[name] += weight * cell[name]["prevalence"]
    compared_raw = {
        name: sum(cell[name]["artifact_1"] for cell in supported) / coverage[name]
        for name in ["present", "missing"]}
    def details(cell: dict) -> dict:
        detail = {key: int(cell[key]) for key in keys}
        detail.update({"n_present": cell["present"]["n"], "artifact_1_present": cell["present"]["artifact_1"],
                       "n_missing": cell["missing"]["n"], "artifact_1_missing": cell["missing"]["artifact_1"]})
        detail["supported"] = bool(cell["supported"])
        return detail
    return {
        "scheme": scheme, "strata": keys, "minimum_n_each_state_per_cell": min_support,
        "cells_total": len(cells), "cells_supported": len(supported),
        "supported_cell_details": [details(cell) for cell in supported],
        "unsupported_cell_details": [details(cell) for cell in cells if not cell["supported"]],
        "covered_rows_total": int(pooled_n), "coverage_all_rows": float(pooled_n / len(df)),
        "covered_present_rows": int(coverage["present"]),
        "total_present_rows": int((df["_availability"] == "present").sum()),
        "coverage_present_group": float(coverage["present"] / (df["_availability"] == "present").sum()),
        "covered_missing_rows": int(coverage["missing"]),
        "total_missing_rows": int((df["_availability"] == "missing").sum()),
        "coverage_missing_group": float(coverage["missing"] / (df["_availability"] == "missing").sum()),
        "supported_sample_raw_prevalence": compared_raw,
        "standardized_reference_prevalence": rates,
        "standardized_missing_minus_present_pp": float(100 * (rates["missing"] - rates["present"])),
        "compared_denominator_definition": "The standardized rates use only supported-cell rows; report both group-specific denominator/coverage and pooled supported-cell rows above.",
        "weight_definition": "Within a supported cell, weight=(n_present+n_missing)/total rows in supported cells; identical weights apply to both cell-specific rates.",
    }


def make_volume_sensitivity(df: pd.DataFrame) -> dict:
    temp = df.copy()
    temp["_period"] = np.select(
        [temp["_cycle"].between(47, 57), temp["_cycle"].between(58, 68), temp["_cycle"].between(69, 79)],
        [0, 1, 2], default=-1).astype(int)
    if (temp["_period"] < 0).any():
        raise ValueError("Train includes cycles outside periods 47–57, 58–68, and 69–79.")
    if temp["text_load"].isna().any():
        raise ValueError("text_load has nulls; define target-blind missing-volume handling before binning.")
    edges_by_q, volume_only, period_volume = {}, [], []
    cycle_only = standardize_cells(temp, ["_cycle"], "exact_cycle", 20)
    for q in [5, 10]:
        labels, edges = pd.qcut(temp["text_load"], q=q, labels=False, retbins=True, duplicates="raise")
        temp[f"_volume_q{q}"] = labels.astype(int)
        edges_by_q[str(q)] = [float(v) for v in edges]
        volume_only.append(standardize_cells(temp, [f"_volume_q{q}"], f"global_{q}_quantile_bins", 20))
        period_volume.append(standardize_cells(temp, ["_period", f"_volume_q{q}"],
                                                f"period_x_global_{q}_quantile_bins", 20))

    present_range = [float(temp.loc[temp["_availability"].eq("present"), "text_load"].min()),
                     float(temp.loc[temp["_availability"].eq("present"), "text_load"].max())]
    missing_range = [float(temp.loc[temp["_availability"].eq("missing"), "text_load"].min()),
                     float(temp.loc[temp["_availability"].eq("missing"), "text_load"].max())]
    common_low, common_high = max(present_range[0], missing_range[0]), min(present_range[1], missing_range[1])
    in_common = temp["text_load"].between(common_low, common_high, inclusive="both").to_numpy()
    is_missing = temp["_availability"].eq("missing").to_numpy()
    return {
        "binning_rule": "Global qcut quantiles of text_load on all official Train rows. Binning uses neither artifact_signal nor availability.",
        "quantile_edges": edges_by_q,
        "common_text_load_support": {
            "present_min_max": present_range, "missing_min_max": missing_range,
            "intersection_inclusive": [common_low, common_high],
            "missing_rows_outside_intersection": int((is_missing & ~in_common).sum()),
            "present_rows_outside_intersection": int((~is_missing & ~in_common).sum()),
            "rows_inside_intersection": int(in_common.sum())},
        "comparisons": {"cycle_only": cycle_only, "volume_only": volume_only, "period_x_volume": period_volume},
        "interpretation": "Different standardized contrasts can reflect composition and cell-support changes. They are descriptive sensitivity checks, not causal adjustment or proof that text_load confounds availability; supported-row and group-specific coverage must accompany the contrast.",
    }


def make_figures(overall: list, periods: list, cycles: list, stats: dict) -> None:
    colors = {"present": "#2878a3", "missing": "#cf623b"}
    fig, ax = plt.subplots(figsize=(9.0, 5.0), layout="constrained")
    items = [("Overall", overall[0], overall[1])]
    items += [(f"Cycle {p['period'][0]}–{p['period'][1]}", p["groups"][0], p["groups"][1]) for p in periods]
    y = np.arange(len(items))
    for label, group_index, offset in [("present", 1, -0.11), ("missing", 2, 0.11)]:
        vals = [item[group_index]["artifact_prevalence"] for item in items]
        lo = [item[group_index]["artifact_wilson95"]["low"] for item in items]
        hi = [item[group_index]["artifact_wilson95"]["high"] for item in items]
        xerr = np.array([[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]])
        ax.errorbar(vals, y + offset, xerr=xerr, fmt="o", capsize=3, color=colors[label], label=label)
    ax.set_yticks(y, [it[0] for it in items]); ax.set_xlim(0.25, 0.56)
    ax.set_xlabel("Proporsi artifact_signal = 1 (Wilson 95% interval)")
    ax.set_title("Komposisi artifact menurut ketersediaan signal_stream")
    ax.grid(axis="x", alpha=.22); ax.legend(title="signal_stream")
    fig.savefig(OUT / "prevalence_by_availability.png", dpi=170); plt.close(fig)

    supported = [c for c in cycles if c["supported_min20_each_state"]]
    fig, ax = plt.subplots(figsize=(10.0, 5.0), layout="constrained")
    x = np.array([c["cycle"] for c in supported]); d = np.array([c["missing_minus_present_pp"] for c in supported])
    size = np.array([c["groups"]["missing"]["n"] for c in supported])
    ax.axhline(0, color="#424242", lw=1)
    ax.scatter(x, d, s=18 + 0.045 * size, color=np.where(d >= 0, "#cf623b", "#2878a3"), alpha=.82,
               linewidths=.45, edgecolors="white")
    for j, (start, end) in enumerate(PERIODS):
        if j:
            ax.axvline(start - .5, ls=":", color="#888888", lw=1)
        ax.text((start + end) / 2, .97, f"{start}–{end}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="#555555")
    st = stats["exact_cycle_standardization"]["standardized_missing_minus_present_pp"]
    ax.axhline(st, ls="--", color="#6941a5", lw=1.4, label=f"Cycle-standardized gap: {st:+.2f} pp")
    ax.set_xlabel("cycle_code (marker area increases with missing-stream denominator)")
    ax.set_ylabel("Within-cycle difference in artifact prevalence (pp)")
    ax.set_title("Arah gap missing − present bervariasi antar-cycle")
    ax.grid(axis="y", alpha=.2); ax.legend()
    fig.savefig(OUT / "cycle_gaps.png", dpi=170); plt.close(fig)


def write_notes(stats: dict) -> None:
    g = stats["overall_groups"]
    z = stats["exact_cycle_standardization"]
    v = stats["text_load_sensitivity"]
    c = v["comparisons"]
    per = stats["periods"]
    v5, v10 = c["volume_only"]
    j5, j10 = c["period_x_volume"]
    lines = [
        "# Raw-data EDA: ketersediaan `signal_stream` dan komposisi label",
        "",
        "**Pertanyaan:** apakah komposisi `artifact_signal` berbeda saat `signal_stream` tersedia dibanding kosong, dan apakah selisih mentah bertahan setelah menyamakan distribusi `cycle_code`? Ini analisis deskriptif dari official Classification Train.",
        "",
        f"Train memiliki {stats['source']['rows']:,} baris unik (SHA-256 `{stats['source']['sha256']}`); aturan availability sama dengan wrapper frozen: null, kosong, atau whitespace-only dianggap missing. Terdapat {g[1]['n']:,} baris missing ({100*g[1]['n']/stats['source']['rows']:.2f}%) dan {g[0]['n']:,} baris present.",
        "",
        f"Raw prevalence artifact adalah **{100*g[1]['artifact_prevalence']:.2f}%** pada stream missing (n={g[1]['n']:,}; Wilson 95% {100*g[1]['artifact_wilson95']['low']:.2f}–{100*g[1]['artifact_wilson95']['high']:.2f}%) dan **{100*g[0]['artifact_prevalence']:.2f}%** pada stream present (n={g[0]['n']:,}; Wilson 95% {100*g[0]['artifact_wilson95']['low']:.2f}–{100*g[0]['artifact_wilson95']['high']:.2f}%). Gap marginalnya **{stats['overall_missing_minus_present_pp']:+.2f} poin persentase**.",
        "",
        "### Periode temporal",
    ]
    for p in per:
        lines.append(f"- Cycle {p['period'][0]}–{p['period'][1]}: present {100*p['groups'][0]['artifact_prevalence']:.2f}% (n={p['groups'][0]['n']:,}); missing {100*p['groups'][1]['artifact_prevalence']:.2f}% (n={p['groups'][1]['n']:,}); gap {p['missing_minus_present_pp']:+.2f} pp.")
    lines += [
        "",
        f"### Standardisasi menurut exact cycle",
        "",
        f"Menggunakan cycle dengan ≥{MIN_PER_STATE_PER_CYCLE} baris pada masing-masing status stream, terdapat {z['supported_cycles']}/{stats['schema_audit']['cycle_min_max'][1]-stats['schema_audit']['cycle_min_max'][0]+1} cycle yang memenuhi dukungan. Common cycle set mencakup {z['covered_rows']:,}/{stats['source']['rows']:,} baris ({100*z['coverage_all_rows']:.2f}% overall), termasuk {z['covered_missing_rows']:,}/{g[1]['n']:,} missing ({100*z['coverage_missing_group']:.2f}%) dan {z['covered_present_rows']:,}/{g[0]['n']:,} present ({100*z['coverage_present_group']:.2f}%). Bobot setiap cycle ialah proporsi seluruh baris pada common cycle set di cycle itu; jadi kedua status dibandingkan pada distribusi cycle yang sama.",
        "",
        f"Setelah standardisasi cycle, prevalence menjadi {100*z['standardized_present_prevalence']:.2f}% present dan {100*z['standardized_missing_prevalence']:.2f}% missing; gap **{z['standardized_missing_minus_present_pp']:+.2f} pp**. Arah gap positif pada {z['supported_cycle_direction_counts']['positive']}, negatif pada {z['supported_cycle_direction_counts']['negative']}, dan nol pada {z['supported_cycle_direction_counts']['zero']} cycle yang didukung (rentang {z['cycle_gap_range_pp'][0]:+.2f} hingga {z['cycle_gap_range_pp'][1]:+.2f} pp). Cycle mix saja tidak menghilangkan gap marginal, tetapi ini belum memperhitungkan volume teks.",
        "",
        "### Sensitivitas terhadap volume teks",
        "",
        "Selanjutnya text_load dibagi ke kuantil global 5 dan 10; batas bin dihitung dari seluruh Train tanpa menggunakan target atau availability. Bobot dalam setiap standardisasi sama bagi kedua status: ukuran cell (n_present+n_missing) dibagi total baris pada semua cell yang didukung. Cell harus memiliki minimal 20 baris missing dan 20 present.",
        "",
        f"- Cycle saja, exact cycle 33/33: {z['covered_rows']:,}/50,608 baris (100.00%; missing {z['covered_missing_rows']:,}/3,404, present {z['covered_present_rows']:,}/47,204), gap **{z['standardized_missing_minus_present_pp']:+.2f} pp**.",
        f"- Volume saja, 5 kuantil: {v5['cells_supported']}/{v5['cells_total']} bins; {v5['covered_rows_total']:,}/50,608 (100.00%; missing {v5['covered_missing_rows']:,}/3,404, present {v5['covered_present_rows']:,}/47,204), gap **{v5['standardized_missing_minus_present_pp']:+.2f} pp**. Dengan 10 kuantil: {v10['cells_supported']}/{v10['cells_total']} bins; full coverage, gap **{v10['standardized_missing_minus_present_pp']:+.2f} pp**.",
        f"- Periode×volume, 3×5 bins: {j5['cells_supported']}/{j5['cells_total']} cells; {j5['covered_rows_total']:,}/50,608 (100.00%; missing {j5['covered_missing_rows']:,}/3,404, present {j5['covered_present_rows']:,}/47,204), gap **{j5['standardized_missing_minus_present_pp']:+.2f} pp**.",
        f"- Periode×volume, 3×10 bins: {j10['cells_supported']}/{j10['cells_total']} cells; {j10['covered_rows_total']:,}/50,608 ({100*j10['coverage_all_rows']:.2f}%; missing {j10['covered_missing_rows']:,}/3,404 ({100*j10['coverage_missing_group']:.2f}%), present {j10['covered_present_rows']:,}/47,204 ({100*j10['coverage_present_group']:.2f}%)), gap **{j10['standardized_missing_minus_present_pp']:+.2f} pp**. Dua cells tidak memenuhi dukungan karena masing-masing hanya berisi 17 dan 15 laporan missing; kedua cell dan denominatornya tercatat di stats.json.",
        "",
        f"Gap berubah arah: positif setelah penyamaan cycle, tetapi negatif pada volume-only serta period×volume yang didukung. Rentang volume present {v['common_text_load_support']['present_min_max'][0]:.4f}–{v['common_text_load_support']['present_min_max'][1]:.4f}; missing {v['common_text_load_support']['missing_min_max'][0]:.4f}–{v['common_text_load_support']['missing_min_max'][1]:.4f}; intersection {v['common_text_load_support']['intersection_inclusive'][0]:.4f}–{v['common_text_load_support']['intersection_inclusive'][1]:.4f}. Di luar intersection terdapat {v['common_text_load_support']['missing_rows_outside_intersection']:,} missing dan {v['common_text_load_support']['present_rows_outside_intersection']:,} present rows. Ini menunjukkan asosiasi raw peka terhadap komposisi volume dan definisi support; tidak membuktikan text_load sebagai confounder kausal.",
        "",
        "### Apa artinya untuk modeling",
        "",
        "Raw EDA memberi tahu bahwa status ketersediaan stream berkaitan dengan komposisi label dan berubah menurut waktu; ini belum memberi tahu apakah stream kosong menyebabkan label atau merupakan sinyal proses pengumpulan. Pada diagnostics model yang dibuat kemudian, C2_B hybrid memiliki missing-stream Macro F1 0.710427, sedangkan C1_B structured HGB mencapai 0.781712. C4_HGB merutekan missing rows ke HGB dan menaikkan aggregate primary 0.820633→0.825481 serta H12 0.788581→0.796810. Angka model tersebut adalah bukti retrospektif terpisah, bukan hasil raw EDA ini atau gain baru dari pertanyaan EDA; routing sudah menjadi bagian model final.",
        "",
        "text_load berbeda secara deskriptif (lihat median dan kuartil per status di stats.json). Kontras standardisasi adalah deskriptif tanpa CI adjusted; Wilson intervals di atas merangkum setiap rate mentah secara terpisah. Seluruh cycle sudah berulang kali digunakan dalam pengembangan, jadi perbandingan temporal bukan konfirmasi independen.",
        "",
        "**Nilai EDA untuk tugas:** ini dapat menjadi cerita EDA-ke-keputusan modeling yang konkret, asalkan dituturkan jujur sebagai profil ketersediaan dan heterogenitas yang kemudian memotivasi routing. Kelemahannya, hipotesis menjadi lebih kuat setelah diagnostics model; jangan menyajikannya seolah discovery pra-modeling atau mengulang skor sebagai analisis EDA. `signal_stream` missing juga proses availability, bukan teks ber-TTR rendah.",
        "",
        "Grafik utama: `prevalence_by_availability.png` (rates dan Wilson intervals menurut keseluruhan/periode) dan `cycle_gaps.png` (arah tiap cycle; ukuran titik mengikuti denominator missing). Machine-readable detail, schema, raw hashes, seluruh denominator, dan batas interpretasi ada di `stats.json`.",
    ]
    (OUT / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
