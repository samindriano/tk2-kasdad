"""Summarize the preregistered retention rules, never select new weights."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/simplification_v1'

def main():
 r=json.loads((OUT/'RESULTS.json').read_text());t='0.53230976129477'
 cls={};reg={}
 counts={'REFERENCE':4,'NO_MEMORY':4,'NO_POS':3,'NO_MEMORY_NO_POS':3,'LR_CB':2,'POS_CB':3,'HGB_MISSING':4}
 for n in r['classification']['UNIQUE']['policies'][t]:
  views={v:x['policies'][t][n] for v,x in r['classification'].items()}
  group=[g for x in views.values() for g in x['groups']]
  unknown=sum(len(x['unsupported']) for x in views.values())
  checks={'primary_views_harm_le_0005':all(views[v]['gain']>=-.0005 for v in ['E67_H12','E75_W6','UNIQUE']),
    'each_E67_window_harm_le_002':all(views[v]['gain']>=-.002 for v in ['E67_W4','E67_W5','E67_W6']),
    'group_f1_harm_le_01':all(g['f1_harm']<=.01 for g in group),
    'group_recall_harm_le_02':all(g['recall_harm']<=.02 for g in group),'no_unsupported_groups':unknown==0}
  cls[n]={'predictors':counts[n],'retention':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
    'unique_gain':views['UNIQUE']['gain'],'E67_H12_gain':views['E67_H12']['gain'],'E75_W6_gain':views['E75_W6']['gain'],
    'max_group_f1_harm':max(g['f1_harm'] for g in group),'max_group_recall_harm':max(g['recall_harm'] for g in group),
    'unique_threshold05_gain':r['classification']['UNIQUE']['policies']['0.5'][n]['gain']}
 for n in r['regression']['D1']['candidates']:
  x={v:d['candidates'][n] for v,d in r['regression'].items()}
  pool=sum(r['regression'][v]['n']*x[v]['gain'] for v in x)/15953
  raw=[(v,x[v]['raw_space_gain']) for v in x]
  rawpool=sum(r['regression'][v]['n']*g for v,g in raw)/15953 if all(g is not None for _,g in raw) else None
  groups=[g for v in x.values() for g in v['groups'].values()]
  checks={'pool_harm_le_02':pool>=-.02,'each_window_harm_le_05':all(v['gain']>=-.05 for v in x.values()),
    'group_harm_le_25':all(g['gain'] is not None and g['gain']>=-.25 for g in groups)}
  reg[n]={'predictors':4 if n in ['NO_LGB_SVD','NO_RIDGE','NO_WORD','NO_CHAR','NO_POS'] else 5,
    'retention':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'pooled_gain':pool,
    'D1_gain':x['D1']['gain'],'D2_gain':x['D2']['gain'],'raw_pooled_gain':rawpool,
    'max_supported_group_harm':max(-g['gain'] for g in groups if g['gain'] is not None),
    'raw_retention':('PASS' if rawpool>=-.02 and all(g>=-.05 for _,g in raw) else 'FAIL') if rawpool is not None else 'NOT_APPLICABLE'}
 summary={'classification':cls,'regression':reg,'selection_method':'Frozen fixed ablations; no combinations/weight search; repeated development evidence only'}
 with (OUT/'DECISIONS.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
 for task,data,key,title in [('classification',cls,'unique_gain','Classification: fixed component removals'),('regression',reg,'pooled_gain','Regression: fixed component removals')]:
  fig,ax=plt.subplots(figsize=(10,5.8));names=list(data);values=[data[n][key] for n in names]
  ax.barh(names,values,color=['#157f79' if data[n]['retention']=='PASS' else '#bb6658' for n in names]);ax.axvline(0,color='black',lw=.8)
  ax.set_xlabel('Macro F1 gain vs deployed-policy reference' if task=='classification' else 'MAE improvement vs same-history hinge')
  for i,n in enumerate(names):ax.annotate(f" {values[i]:+.6f} | {data[n]['predictors']} predictors",(values[i],i),ha='left' if values[i]>=0 else 'right',va='center',fontsize=8)
  ax.margins(x=.6);ax.set_title(title+'\nGreen: passes predeclared retention and group guards');fig.tight_layout();fig.savefig(OUT/f'{task.upper()}_ABLATIONS.png',dpi=160);plt.close(fig)
 lines=['# Dual-task model simplification — root findings','',
 '**Decision:** prioritize a POS-free controlled benchmark for each task; retain the deployed champions pending actual application/public testing. This is a zero-fit development audit, not a replacement decision.','',
 'The requested audit was performed directly by root in an isolated worktree. Protocol/code/input hashes were frozen before scoring. Six Classification simplifications and seven Regression simplifications were compared, with no weight search, new fitting, raw dataset/Test access or Kaggle upload.','',
 '## Classification','', '| Candidate | Distinct predictors | E67 H12 F1 gain | E75 W6 gain | Unique pool gain | Retention |','|---|---:|---:|---:|---:|---|']
 for n,d in cls.items():lines.append(f"| {n} | {d['predictors']} | {d['E67_H12_gain']:+.6f} | {d['E75_W6_gain']:+.6f} | {d['unique_gain']:+.6f} | {d['retention']} |")
 lines += ['', 'NO_POS removes the positional gamma and token-third transform; it keeps C2B LR, sparse LightGBM, CatBoost missing fallback and A02 residual-neighbor memory. This reduces distinct predictors4→3 and removes positional feature assembly, but retains donor memory/search. All three primary temporal views improve. Worst supported subgroup F1 harm is '+f"{cls['NO_POS']['max_group_f1_harm']:.6f}"+'; worst positive-recall harm is '+f"{cls['NO_POS']['max_group_recall_harm']:.6f}"+' (close to the0.02 guard). No unsupported group cells were omitted.',
 'Removing A02 memory saves donor storage/search but loses about0.00253 pool F1. A compact LR+CatBoost route loses about0.00572. Replacing missing CatBoost with HGB harms the latest window and supported missing subgroup; it is a fixed-threshold transport check, not an optimized HGB conclusion.',
 'Threshold0.50 missing sensitivity for NO_POS has unique F1 gain '+f"{cls['NO_POS']['unique_threshold05_gain']:+.6f}"+'. The operational threshold0.53230976129477 was selected historically and is not re-tuned here.',
 '**Public counterevidence:** A02 without the later POS blend was already a historical submission (reported public0.82930), whereas the later POS/A02 operational champion reached0.83225. Thus better local NO_POS is not proof it will retain public performance; do not treat it as an entirely new architecture. The precise current missing policy differs from older submissions. A deployment comparison must hash/reconstruct the exact new policy rather than resubmit an old file blindly.',
 '', '![Classification ablations](CLASSIFICATION_ABLATIONS.png)','', '## Regression','', '| Candidate | Predictors | D1 gain | D2 gain | Pooled MAE gain | Raw-space gain | Retention |','|---|---:|---:|---:|---:|---:|---|']
 for n,d in reg.items():
  raw='—' if d['raw_pooled_gain'] is None else f"{d['raw_pooled_gain']:+.6f}"
  lines.append(f"| {n} | {d['predictors']} | {d['D1_gain']:+.6f} | {d['D2_gain']:+.6f} | {d['pooled_gain']:+.6f} | {raw} | {d['retention']} |")
 lines += ['', 'NO_POS leaves the four-leaf R06 ensemble, then applies the existing same-history hinge present-only. Both windows improve; missing predictions retain their original policy within floating arithmetic tolerance. Target-tail groups also improve. This removes one400-tree LightGBM and positional token-third features (runtime savings not measured).',
 'NO_CHAR is a secondary candidate: removing the character learner saves an entire character TF-IDF representation and400-tree LightGBM; pooled gain is small, D1 worsens about0.02617, and the low-target group worsens about0.24745, close to the0.25 guard. It passes the predeclared retention rule but has materially weaker stability than NO_POS. No combined NO_POS+NO_CHAR candidate was tested; do not infer their gains add.',
 'LGB-SVD, Ridge and word learner removals fail retention. Ridge has a relatively small weight but still contributes useful complementarity; weight size alone does not establish redundancy. Removing calibration loses about0.05908 MAE. Returning to GLOBAL loses0.00636; hinge is a cheap three-parameter correction and should remain in the simplification direction.',
 '### Numerical flattening, separate from model removal','',
 'Present DIRECT can be written as0.31640625·mappedLGB +0.10546875·mappedRidge +0.140625·mappedWord +0.1875·mappedChar +0.25·mappedPosition before final clipping. Missing uses0.421875/0.140625/0.1875/0.25/0, respectively. Cached LGB/Ridge values are already mapped. The positional rank50 branch remains un-clipped until blending. Original clip/rank boundaries are preserved. Nested-versus-flat replay maximum difference is2.84e-14 at both origins: numerically equivalent, not bit-identical and not a model-performance gain. Deployment refactoring should preserve original operation order if bit identity matters.',
 '', '![Regression ablations](REGRESSION_ABLATIONS.png)', '', '## Evidence and limits','',
 'Classification uses20,903 E67 H12 rows and8,491 E75 W6 rows; their newest cycles overlap. The unique pool is E67 cycles68–75 plus E75 cycles76–79, n20,903. Regression causal windows contain7,462 and8,491 rows, pooled15,953. No final full-OOF fit coefficients are evaluated on their own calibration training rows.','',
 'Component removals reuse existing trained leaf predictions and renormalize prescribed weights; they are executable inference ablations, not fresh end-to-end refits. Regression hinge parameters were trained for the original ensemble, so transported-calibration results can advantage/disadvantage a removal. Raw-space results are also shown; no recalibration occurred. Cached data and labels have been used repeatedly in prior research; retention PASS means the explicit local tolerances, not statistical equivalence or a guarantee of Kaggle scores.','',
 'Resource use: initial available '+str(r['resource']['initial_available_bytes'])+' bytes, final available '+str(r['resource']['final_available_bytes'])+' bytes, estimate256MiB incremental; shared OS lock. No measured deployment latency/peak-memory reductions are claimed. Distinct-predictor counts omit preprocessing state and neighbor memory, which must remain explicit. Four tests cover identity alignment, complete-cycle tied ranks, metric cross-check and non-commuting clipping. Every Classification F1 is checked against sklearn; every Regression MAE against a separate scalar absolute-error calculation. All original full-model vectors and nested/flat recipes replay within1e-10.','',
 '## Bounded next decision','',
 'Prioritize1) REG four-leaf R06 plus unchanged hinge,2) CLS A02-only present plus unchanged operational CatBoost missing rule. First verify exact E79 artifacts/feature dependencies and original training history; no new model training should be necessary if existing leaf inference artifacts suffice. Compare exact policies with the current submitted CSVs before considering a benchmark upload. REG NO_CHAR may be a separate low-priority efficiency benchmark. Do not start combinatorial deletion, retraining or distillation until these single-removal results are understood. Preserve the current champions and all ongoing Luna experiments.','',
 'Reproduce: `python -m unittest tests.test_simplification_audit_v1`; first immutable run uses `python -m scripts.simplification_audit_v1 --freeze`, then `--run`, then `python -m scripts.report_simplification_audit_v1`. Existing outputs are exclusive-write. External roots may be passed explicitly via CLI; all frozen source/cache paths and hashes are listed in INPUT_FREEZE.json. Raw datasets and model binaries are not committed.']
 (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
