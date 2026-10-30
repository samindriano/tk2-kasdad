"""Build and Restart/Run All Samuel's compact Regression Round2 research notebook."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md('''# Samuel — Regression Round 2

KASDAD TP2 · Feature attribution, training history sensitivity & temporal generalization

## Motivation & hypotheses

Round 1: expanded structured HGB improves primary MAE but harms early H12. We ask which feature groups contribute, whether more recent observations matter at matched sample size, and which residual segments remain difficult. Conditional-median reasoning from REG-Lite B remains relevant to MAE; its TP1 multiclass recipe is not copied.

## Context & Methods

Same six 4-cycle primary windows and two H12 views, 38,588 records per family. H12 overlaps primary. The whole Train has already been explored; this is adaptive development evidence, not an untouched test. Fixed absolute-error HGB, 200 iterations, 15 leaves, learning rate .05, minimum leaf 30, L2 1, seed 42, no early stopping, two threads. Five controlled group removals; recent 8/12 histories; exact-size random past samples with two fixed seeds. Up to four adaptive interactions were separately frozen before their fits.

### Key assumptions

Cycle order represents earlier/later observations, not a verified calendar. Additional structured features retain supplied-schema eligibility and provisional operational timing. Matched row counts do not remove composition or temporal confounding. Official Test features are descriptive only; the research pipeline never predicts Test. The separate initial-submission track uses older frozen recipes and does not select these experiments.'''),
        code('''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Image

ROOT = Path.cwd().resolve()
while not (ROOT/'src/utils/paths.py').is_file() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT/'src/utils/paths.py').is_file():
    raise FileNotFoundError('Open this notebook inside the TP2 repository.')
sys.path.insert(0, str(ROOT))
from src.models.regression_round2_evaluation import verify_freeze, load_protocol, run_round2
from src.models.regression_round2_reporting import load_results, comparison_frame, build_report
freeze = verify_freeze()
protocol = load_protocol()
# Reconstruct only the already-selected frozen recipes if ignored local vectors are missing.
result_path = ROOT/'reports/regression_round2/results.json'
old = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
needed = [ROOT/'experiments/runs/regression_round2_v1'/f"fit{end}__{r['id']}.npz"
          for end in [55,59,63,67,71,75] for r in (old['recipes'] if old else protocol['recipes'])]
if not old or not all(p.exists() for p in needed):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        run_round2(stage='all')
results = load_results()
summary = comparison_frame(results)
print('Freeze:', freeze['signature'])
print(len(results['recipes']), 'recipes;', len(results['jobs']), 'window evaluations')
print('Protected checkpoint files:', len(freeze['protected_checkpoint_files']))'''),
        md('''## tl;dr

The table below identifies the lowest observed primary and H12 MAE separately. These are review candidates, not final model decisions. **Full expanded recent-8 leads primary; expanding without category codes leads H12.** Category removal helps early H12 but harms late H12 and the high target tail. Recent history does not improve long-horizon results consistently. Removing cycle becomes less costly with recent history, while still trailing the full recent model. Read these counterexamples before interpreting pooled gains.'''),
        code('''for metric in ['Primary MAE', 'H12 MAE']:
    best = summary.loc[summary[metric].idxmin()]
    print(f"Lowest observed {metric}: {best['Recipe']} = {best[metric]:.6f}")
display(summary[['Recipe','Primary MAE','Equal-cycle MAE','H12 MAE','H12_E','H12_L']].round(5))'''),
        md('''## Experimental design

The two exact controls delegate to Round 1. Feature removals drop all derived columns associated with a raw feature. Encoders/imputers are fitted solely on selected past rows. Recent histories preserve complete cycles; matched samples are target-blind individual-row samples from expanding history, with exactly the recent-8 count. Their two seeds measure limited sample sensitivity.

Feature/history interactions are explicitly adaptive. Their numerical selection rationale and immutable signature are shown below; no test features or Kaggle scores are used for that choice.'''),
        code('''display(pd.DataFrame([{'family':f, **w} for f,k in [('primary','primary_windows'),('H12','secondary_windows')] for w in protocol[k]]))
selection = json.loads((ROOT/'experiments/regression_round2/interaction_selection.json').read_text(encoding='utf-8'))
print('Interaction selection:', selection['signature'])
print(selection['payload']['rationale'])
display(pd.DataFrame(results['recipes']))'''),
        md('''## Results — feature contribution

Negative removal-minus-full MAE means that removing the group helped this fitted model. A zero or positive result does not establish universal feature usefulness. R1-D's W1/H12 early counterexample is retained.

The reusable plotting code lives in `src/models/regression_round2_reporting.py`; each figure is regenerated from saved metrics and verified vectors, then embedded for portable viewing.'''),
        code('''figure_paths = build_report()
diagnostics = json.loads((ROOT/'reports/regression_round2/behavior_diagnostics.json').read_text(encoding='utf-8'))
display(Image(filename=str(ROOT/'reports/regression_round2/figures/regression_round2_ablation_window_deltas.png')))'''),
        md('''## Results — history and feature × history

Equal sample count partially separates data volume from period selection. Training composition, category novelty and dependence still vary. Interactions compare a removal against the full model under the same history, then against the corresponding expanding-history contrast; they are predictive refit contrasts.'''),
        code('''display(Image(filename=str(ROOT/'reports/regression_round2/figures/regression_round2_history_comparison.png')))
display(pd.DataFrame(diagnostics['interactions']).drop(columns=['interpretation']).round(5))
if diagnostics['interactions']:
    display(Image(filename=str(ROOT/'reports/regression_round2/figures/regression_round2_feature_history_interactions.png')))'''),
        md('''## Results — residuals, tails and horizon

Signed error is prediction minus actual. Actual target bins are evaluation-only and never used as an inference gate. Small pooled bias can conceal overprediction at low targets and underprediction at high targets. H12 lead profiles change era and horizon together.'''),
        code('''display(summary[['Recipe','RMSE','Signed error','EF0','EF1','Low tail','High tail']].round(5))
display(Image(filename=str(ROOT/'reports/regression_round2/figures/regression_round2_target_region_residuals.png')))
display(Image(filename=str(ROOT/'reports/regression_round2/figures/regression_round2_horizon_profiles.png')))
display(Image(filename=str(ROOT/'reports/regression_round2/figures/regression_round2_cycle_and_tail_errors.png')))'''),
        md('''## Results — descriptive generalization risk

These checks use official Test predictors after experiment selection. They do not reveal Test labels or establish actual future MAE. Unknown category handling prevents a preprocessing failure; it does not guarantee that unknown categories are predictable. Numeric cycle trees partition observed support and do not extrapolate a learned time slope.'''),
        code('''risk = diagnostics['official_test_feature_risk']
print('Cycle gap:', risk['cycle_gap'])
display(pd.DataFrame([{'feature':name,'train missing rate':v['train_missing_rate'],'test missing rate':v['test_missing_rate'],
                      'test unknown n':v.get('test_unknown_n'),'KS distance':v.get('descriptive_ks_distance')}
                     for name,v in risk['features'].items()]).round(5))'''),
        md('''## Interpretation, limitations & next steps

Read [FINDINGS.md](../../reports/regression_round2/FINDINGS.md) for the evidence-led interpretation, counterexamples and Round 3 recommendations, and [COMPARISON_TABLES.md](../../reports/regression_round2/COMPARISON_TABLES.md) for every candidate/window/cycle and subgroup. The complete exact-impact tables and conditional IQR diagnostics remain in JSON to keep this notebook short.

Absolute-error fitting targets conditional medians. Tail shrinkage alone is not a reason to apply percentile calibration: ranking quality, conditional spread and MAE trade-offs still need a separate preregistered study. Feature ablation is conditional predictive evidence, and matched-size sampling is not a causal estimate of recency. No automatic model zoo, target transform, text experiment or Round 3 follows this milestone.

## Checks

Independent verification reconciles saved vectors, original control equality, fit-only preprocessing and temporal ordering. The receipt below records what was actually checked; representative repeat evidence is claimed only when present and identity-bound.'''),
        code('''receipt = json.loads((ROOT/'reports/regression_round2/verification.json').read_text(encoding='utf-8'))
assert receipt['status'] == 'PASS'
assert receipt['freeze_signature'] == freeze['signature']
import hashlib
assert receipt['results_sha256'] == hashlib.sha256(result_path.read_bytes()).hexdigest()
display(pd.Series({k:v for k,v in receipt.items() if not isinstance(v,(dict,list))}))
print('Representative repeat:', receipt['repeat_verification'])
print('No official-test predictions, classification refits or Kaggle selection occur in this notebook.')''')]
    nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}})
    path = ROOT/'notebooks/samuel/07_samuel_regression_round2.ipynb'
    nbformat.write(nb, path)
    NotebookClient(nb, timeout=1200, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print(path)


if __name__ == '__main__':
    main()
