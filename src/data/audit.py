"""Read-only CSV profiling. Aggregate evidence only; never fits a model."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.utils.paths import task_files


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_numeric(series):
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()


def profile(frame):
    columns = {}
    for name in frame:
        series = frame[name]
        item = {'dtype': str(series.dtype), 'missing': int(series.isna().sum()),
                'missing_rate': float(series.isna().mean()) if len(series) else None, 'unique_nonmissing': int(series.nunique()),
                'blank_strings': int(series.astype('string').str.fullmatch(r'\s*').fillna(False).sum())}
        if pd.api.types.is_numeric_dtype(series):
            values = finite_numeric(series)
            item['nonfinite_excluding_missing'] = int(np.isinf(series.to_numpy(dtype=float)).sum())
            item['summary'] = {str(k): float(v) if np.isfinite(v) else None for k, v in values.describe(percentiles=[.01, .25, .5, .75, .99]).items()}
            if len(values):
                lower, upper = values.quantile([.25, .75])
                iqr = upper - lower
                item['iqr_fences'] = [float(lower-1.5*iqr), float(upper+1.5*iqr)]
                item['iqr_outlier_count'] = int(((values < lower-1.5*iqr) | (values > upper+1.5*iqr)).sum())
        else:
            counts = series.value_counts(dropna=True)
            item['category_frequency_summary'] = {
                'nonmissing_rows': int(counts.sum()),
                'most_frequent_count': int(counts.iloc[0]) if len(counts) else 0,
                'top10_row_share_nonmissing': float(counts.head(10).sum()/counts.sum()) if counts.sum() else None,
                'singleton_category_count': int((counts == 1).sum()),
            }
            # Do not export raw vulnerability text or record identifiers.
            if name != 'signal_stream' and item['unique_nonmissing'] <= 50:
                item['category_counts'] = {str(k): int(v) for k, v in counts.items()}
            item['length_quantiles'] = {str(k): float(v) if np.isfinite(v) else None for k, v in series.dropna().astype(str).str.len().quantile([0, .5, .95, 1]).items()}
        columns[name] = item
    result = {'shape': list(frame.shape), 'columns': columns, 'exact_duplicate_rows': int(frame.duplicated().sum())}
    if 'record_id' in frame:
        result['duplicate_ids'] = int(frame.record_id.duplicated().sum())
        result['missing_ids'] = int(frame.record_id.isna().sum())
    if 'cycle_code' in frame:
        result['cycle_counts'] = {str(k): int(v) for k, v in frame.cycle_code.value_counts(dropna=False).sort_index().items()}
    return result


def compare_splits(train, test, target):
    common = [c for c in test if c in train]
    features = [c for c in common if c not in ('record_id', target)]
    result = {'train_only_columns': sorted(set(train)-set(test)),
              'test_only_columns': sorted(set(test)-set(train)),
              'feature_order_matches': [c for c in train if c != target] == list(test),
              'dtype_mismatches': {c: [str(train[c].dtype), str(test[c].dtype)] for c in common if train[c].dtype != test[c].dtype},
              'columns': {}}
    if 'record_id' in common:
        result['overlapping_ids'] = int(len(set(train.record_id.dropna()) & set(test.record_id.dropna())))
    result['train_duplicate_feature_rows_excluding_id_target'] = int(train.duplicated(subset=features).sum()) if features else None
    result['test_duplicate_feature_rows_excluding_id_target'] = int(test.duplicated(subset=features).sum()) if features else None
    if features:
        # Exact pandas merge on full feature values, NaN matching NaN. No probabilistic hash matching.
        matches = train[features].drop_duplicates().merge(test[features].drop_duplicates(), on=features, how='inner')
        result['shared_distinct_feature_rows'] = len(matches)
    for name in common:
        if name == 'record_id':
            continue
        a, b = train[name], test[name]
        item = {'missing_rate_train': float(a.isna().mean()), 'missing_rate_test': float(b.isna().mean()),
                'missing_rate_delta_test_minus_train': float(b.isna().mean()-a.isna().mean())}
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = finite_numeric(a), finite_numeric(b)
            if len(x) and len(y):
                item.update({'ks_statistic': float(ks_2samp(x, y).statistic),
                             'train_median': float(x.median()), 'test_median': float(y.median()),
                             'test_outside_train_range': int(((y < x.min()) | (y > x.max())).sum())})
        else:
            x, y = a.dropna().astype(str), b.dropna().astype(str)
            item['test_unseen_category_count'] = int(len(set(y)-set(x)))
            item['test_unseen_category_row_rate_nonmissing'] = float((~y.isin(set(x))).mean()) if len(y) else None
            p, q = x.value_counts(normalize=True), y.value_counts(normalize=True)
            categories = p.index.union(q.index)
            item['total_variation_nonmissing'] = float((p.reindex(categories, fill_value=0)-q.reindex(categories, fill_value=0)).abs().sum()/2)
        result['columns'][name] = item
    return result


def audit_task(task, root=None):
    paths = task_files(task, root)
    missing = [f'{task}/{name}' for name, path in paths.items() if path is None]
    frames = {name: pd.read_csv(path) for name, path in paths.items() if path is not None}
    result = {'task': task, 'status': 'COMPLETE' if not missing else 'PARTIAL_INPUTS' if frames else 'BLOCKED_MISSING_DATA',
              'missing_roles': missing, 'target': None, 'target_verification': 'UNVERIFIED', 'files': {}}
    for name, frame in frames.items():
        result['files'][name] = {'name': paths[name].name, 'bytes': paths[name].stat().st_size,
                                  'sha256': sha256(paths[name]), **profile(frame)}
    if 'train' in frames and 'test' in frames:
        train, test = frames['train'], frames['test']
        candidates = sorted(set(train)-set(test))
        sample_candidates = sorted(set(frames['sample'])-{'record_id'}) if 'sample' in frames else []
        result['target_candidates_train_minus_test'] = candidates
        result['sample_prediction_columns'] = sample_candidates
        if (len(candidates) == 1 and candidates == sample_candidates
                and set(frames['sample'].columns) == {'record_id', candidates[0]}):
            result['target'], result['target_verification'] = candidates[0], 'VERIFIED_TRAIN_TEST_SAMPLE_SCHEMA'
        else:
            result['status'] = 'BLOCKED_TARGET_UNVERIFIED'
        target = result['target']
        result['comparison'] = compare_splits(train, test, target)
        result['predictor_count_excluding_id'] = len(test.columns.difference(['record_id']))
        if target:
            y = train[target]
            result['target_summary'] = {'dtype': str(y.dtype), 'nonmissing': int(y.notna().sum()), 'unique': int(y.nunique())}
            frequencies = y.value_counts()
            result['target_summary']['top10_value_counts'] = {str(k): int(v) for k,v in frequencies.head(10).items()}
            result['target_summary']['top1_row_share'] = float(frequencies.iloc[0]/len(y)) if len(y) else None
            result['target_summary']['top10_row_share'] = float(frequencies.head(10).sum()/len(y)) if len(y) else None
            if pd.api.types.is_numeric_dtype(y):
                finite_y = finite_numeric(y)
                result['target_summary']['quantiles'] = {str(k): float(v) if np.isfinite(v) else None for k,v in finite_y.quantile([0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]).items()}
                result['target_summary']['integer_like_count_atol_1e8'] = int(np.isclose(finite_y, np.round(finite_y), rtol=0, atol=1e-8).sum())
            if y.nunique() <= 100:
                result['target_summary']['counts'] = {str(k): int(v) for k, v in y.value_counts(dropna=False).sort_index().items()}
            result['numeric_target_associations'] = {}
            if pd.api.types.is_numeric_dtype(y):
                for col in train.select_dtypes(include='number'):
                    if col in (target, 'record_id'):
                        continue
                    pair = train[[col, target]].replace([np.inf, -np.inf], np.nan).dropna()
                    if pair[col].nunique() > 1 and pair[target].nunique() > 1:
                        result['numeric_target_associations'][col] = {'n': len(pair), 'pearson': float(pair.corr().iloc[0,1]), 'spearman': float(pair.corr(method='spearman').iloc[0,1])}
            result['proxy_checks'] = {}
            if task == 'classification':
                for col in ('audit_lane', 'checksum_hint'):
                    if col in train and train[col].nunique() <= 300 and not pd.api.types.is_numeric_dtype(train[col]):
                        table = pd.crosstab(train[col].fillna('<MISSING>'), y, dropna=False)
                        result['proxy_checks'][col] = {'contingency_counts': {str(k): {str(a): int(b) for a,b in v.items()} for k,v in table.to_dict().items()}, 'note': 'Association only. Feature generation and availability before outcome UNKNOWN.'}
        if 'sample' in frames:
            sample = frames['sample']
            result['sample_alignment'] = {'rows_match_test': len(sample) == len(test),
                                          'id_order_matches_test': 'record_id' in sample and 'record_id' in test and sample.record_id.equals(test.record_id)}
        result['audit_issues'] = []
        if not result['comparison']['feature_order_matches']:
            result['audit_issues'].append('Feature schema/order differs between train and test.')
        if target and train[target].isna().any():
            result['audit_issues'].append('Missing target values in train.')
        if any(p.get('duplicate_ids', 0) or p.get('missing_ids', 0) for p in result['files'].values()):
            result['audit_issues'].append('Missing or duplicate record IDs present.')
        if result['comparison'].get('overlapping_ids', 0):
            result['audit_issues'].append('Train/test record IDs overlap.')
        if 'sample_alignment' in result and not all(result['sample_alignment'].values()):
            result['audit_issues'].append('Sample rows or ID order differ from test.')
    return result


def compare_tp1(task, current, tp1_root):
    old_paths = task_files(task, tp1_root)
    result = {'source': 'User-provided local TP1 reference, read-only', 'splits': {}}
    for split in ('train', 'test'):
        new_path = current[split]
        old_path = old_paths[split]
        if new_path is None or old_path is None:
            result['splits'][split] = {'status': 'MISSING_INPUT'}
            continue
        old, new = pd.read_csv(old_path), pd.read_csv(new_path)
        item = {'tp1_sha256': sha256(old_path), 'tp1_shape': list(old.shape),
                'added_columns_tp2': sorted(set(new)-set(old)), 'removed_columns_tp2': sorted(set(old)-set(new))}
        if 'record_id' in old and 'record_id' in new and old.record_id.is_unique and new.record_id.is_unique:
            a, b = old.set_index('record_id'), new.set_index('record_id')
            ids = a.index.intersection(b.index)
            item['shared_ids'] = len(ids)
            item['same_id_set'] = set(a.index) == set(b.index)
            item['shared_column_mismatch_counts'] = {}
            for col in a.columns.intersection(b.columns):
                x, y = a.loc[ids, col], b.loc[ids, col]
                item['shared_column_mismatch_counts'][col] = int((~(x.eq(y) | (x.isna() & y.isna()))).sum())
        result['splits'][split] = item
    return result


def environment():
    return {'python': platform.python_version(), 'packages': {p: importlib.metadata.version(p) for p in ['pandas','numpy','scipy','matplotlib','nbformat','nbclient']}}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def markdown_report(result):
    task = result['task']
    lines = [f'# Audit dataset TP2 — {task}', '', f"Status: **{result['status']}**.", '',
             'Grain: satu baris kandidat record; semantik satu vulnerability per record belum dikonfirmasi data dictionary.', '',
             f"Target: `{result['target']}`; evidence: {result['target_verification']}.", '']
    if result['missing_roles']:
        lines += ['Input belum tersedia: '+', '.join(result['missing_roles'])+'.', '',
                  'Diperlukan file resmi Train.csv, Test.csv, SampleSubmission.csv di folder task. Data task lain tidak dipakai sebagai pengganti. Target, missingness, distribusi, leakage, dan kesiapan EDA belum dapat diverifikasi.', '']
    if result['files']:
        lines += ['| File | Rows | Columns | Duplicate rows | Duplicate ID |','|---|---:|---:|---:|---:|']
        for role, p in result['files'].items():
            lines.append(f"| {role} | {p['shape'][0]:,} | {p['shape'][1]} | {p['exact_duplicate_rows']} | {p.get('duplicate_ids', 'N/A')} |")
        lines += ['', 'Nama, dtype setiap kolom, missing, kuantil, IQR, kategori, hash SHA-256 dan hitungan cycle lengkap terdapat dalam JSON pendamping. IQR adalah penanda deskriptif, bukan alasan otomatis menghapus data.', '']
    if 'comparison' in result:
        c = result['comparison']
        lines += [f"Prediktor selain ID: {result['predictor_count_excluding_id']}; fitur berurutan cocok: {c['feature_order_matches']}; ID overlap: {c.get('overlapping_ids', 'UNKNOWN')}; shared distinct feature rows: {c.get('shared_distinct_feature_rows', 'UNKNOWN')}.", '',
                  f"Duplicate fitur train/test tanpa ID dan target: {c['train_duplicate_feature_rows_excluding_id_target']}/{c['test_duplicate_feature_rows_excluding_id_target']}.", '']
        if result.get('target_summary', {}).get('counts'):
            total = result['target_summary']['nonmissing']
            lines += ['Distribusi target train: '+', '.join(f'{k}: {v:,} ({v/total:.2%})' for k,v in result['target_summary']['counts'].items())+'.', '']
        elif result.get('target_summary', {}).get('quantiles'):
            s = result['target_summary']
            lines += [f"Target numerik train: {s['unique']:,} nilai unik; kuantil {s['quantiles']}; top-10 nilai mencakup {s['top10_row_share']:.2%} baris. Angka ini deskriptif; nama/range teramati tidak menetapkan definisi domain target.", '']
        lines += ['| Kolom | Missing train | Missing test | KS numeric / TV categorical |','|---|---:|---:|---:|']
        for col, item in c['columns'].items():
            shift = item.get('ks_statistic', item.get('total_variation_nonmissing'))
            lines.append(f"| {col} | {item['missing_rate_train']:.2%} | {item['missing_rate_test']:.2%} | {shift:.4f} |" if shift is not None else f'| {col} | N/A | N/A | N/A |')
        lines += ['', 'KS = perbedaan maksimum empirical CDF; TV = total variation proporsi kategori nonmissing. Nilai ini deskriptif, tanpa klaim signifikansi atau penyebab drift; kategori teks ber-cardinality tinggi perlu dibaca bersama unseen rate dan panjang teks di JSON.', '',
                  '## Risiko dan implikasi', '',
                  '- HIGH / confidence tinggi: missing dan format heterogen memerlukan parser terdokumentasi; jangan fit imputasi atau encoding pada gabungan train/test.',
                  '- HIGH / confidence pada ketidakpastian: asal pembentukan audit_lane, checksum_hint, dan fitur lintas-target belum diketahui. Korelasi tidak membuktikan atau menyingkirkan leakage; verifikasi ketersediaan sebelum outcome dengan asisten/data dictionary.',
                  '- cycle_code adalah kode urut; arti tanggal/interval belum diketahui. Pisahkan train/validation menurut urutan cycle jika semantik waktu dikonfirmasi; jangan gunakan test untuk memilih fitur, threshold, atau model.',
                  '- impact_load dan entry_friction sebagai fitur TP2 tidak otomatis leakage hanya karena merupakan target TP1. Jangan menggabungkan label TP1 tambahan; gunakan hanya kolom resmi task TP2.',
                  '- Target test tidak tersedia, sehingga target shift train–test tidak dapat diuji. Data dictionary, lineage dan timing fitur tetap UNKNOWN.', '',
                  'Tidak ada training, hyperparameter tuning, atau submission pada audit ini.']
    return '\n'.join(lines).rstrip()+'\n'
