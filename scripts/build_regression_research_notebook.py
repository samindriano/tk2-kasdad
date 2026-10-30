"""Build and execute the presentation notebook for the autonomous research phase."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md('''# Samuel — Autonomous Regression Research

KASDAD TP2 · Prediction calibration, text information and temporal generalization

## Motivation & hypotheses

The initial R1-D public MAE was13.53473 versus the user-supplied official baseline12.23275. This phase asks whether prediction-only cycle ranking, text features and a simple stronger estimator improve MAE while preserving temporal evaluation and explainability. The formal EDA question remains unchanged.

## Context & Methods

Same six expanding 4-cycle windows and two H12 views. Each family has38,588 validation records; H12 reuses primary records. Prior EDA/rounds have used all these cycles, so evidence is adaptive development, not an untouched test. Stage-specific protocols/code/input identities are frozen before each stage. Mapping weights and LightGBM complexities use only past inner labels.

### Key assumptions

Prediction-only percentile mapping uses all supplied rows of a cycle and average ties. Supplied batches may not exhaust the real source universe. It does not prove the official target generator. Added structured/text features retain supplied-schema eligibility with operational timing provisional. No Test labels or fabricated zero-cohort labels are used. Public Kaggle feedback is limited and separate; no more than six additional regression submissions are authorized.'''),
      code('''from pathlib import Path
import sys,json,hashlib
import pandas as pd
from IPython.display import display,Image

ROOT=Path.cwd().resolve()
while not (ROOT/'src/utils/paths.py').exists() and ROOT!=ROOT.parent: ROOT=ROOT.parent
if not (ROOT/'src/utils/paths.py').exists(): raise FileNotFoundError('Open inside the TP2 repository.')
sys.path.insert(0,str(ROOT))
from src.models.regression_research_common import verify_stage
from scripts.report_regression_research import load_stages,comparisons,build
for stage in ['stage1_ranking','stage23_models','stage4_refinement','stage5_lgb_text','shift_diagnostics','scope_sensitivity']:
    receipt=verify_stage(stage)
    print(stage,receipt['signature'][:16],len(receipt['protected_files']),'protected files')
stages=load_stages();comparison=comparisons(stages)
print('Prediction caches are local and ignored; no model weights are committed.')'''),
      md('''## tl;dr

Pure percentile ranking harms the original structured controls. Text adds information, and mapping helps compressed Ridge scores more than HGB scores. SVD32+LightGBM has the lowest observed primary MAE; hybrid Ridge remains the lowest pooled H12 candidate. The exact current public scores are shown separately below; neither internal winner nor public winner is automatically final.'''),
      code('''cols=['Stage','Recipe','Policy','Primary MAE','Equal-cycle MAE','H12 MAE','H12_E','H12_L']
display(comparison[(comparison.Policy=='inner')|((comparison.Stage=='stage1_ranking')&(comparison.Policy=='raw'))][cols].round(5))
figures=build()
display(Image(filename=str(ROOT/'reports/regression_research/figures/research_model_comparison.png')))'''),
      md('''## Results — prediction-only cycle ranking

These weights were fixed before outer evaluation. They are diagnostic comparisons, not per-window oracle selection. Pure ranking changes score spacing and marginal distribution; it cannot repair ranking inversions. Incomplete-reference stress compares a subset against the complete supplied-cycle map on identical retained rows.'''),
      code('''display(Image(filename=str(ROOT/'reports/regression_research/figures/research_rank_mapping.png')))
display(Image(filename=str(ROOT/'reports/regression_research/figures/research_batch_sensitivity.png')))'''),
      md('''## Results — text and model contribution

Text-only Ridge is weak. Same-family structured Ridge is a meaningful negative control: its imputer/scaler/OHE/alpha are identical to hybrid Ridge, but every stream is replaced with a constant empty string during fit and prediction. Hybrid's gain therefore includes text plus availability information. HGB+SVD retains the original native structured categories. LightGBM uses the same SVD32 representation, isolating the model×information interaction without a new text vocabulary recipe.

Stage4 tested128 latent dimensions, no-scope representation and the structured Ridge control. Stage5 reused two previously defined LightGBM complexities; inner validation chose parameters and rank weight jointly. No deep architecture, stacking or large ensemble was used.'''),
      code('''display(Image(filename=str(ROOT/'reports/regression_research/figures/research_window_heatmap.png')))
display(Image(filename=str(ROOT/'reports/regression_research/figures/research_stream_subgroups.png')))'''),
      md('''## Results — errors and target geometry

Actual target bins and tails are evaluation-only; they never select an inference gate. Absolute-error fitting remains median-oriented. A compressed conditional prediction distribution need not copy the outcome distribution: broad conditional spread can reflect uncertainty or unmeasured heterogeneity. Ranking/calibration that improves tails can still worsen central MAE or missing-stream performance.'''),
      code('''display(comparison[comparison.Policy=='inner'][['Recipe','Primary MAE','RMSE','Bias','EF0','EF1','Low tail','High tail','Missing','Present','Spearman']].round(5))
display(Image(filename=str(ROOT/'reports/regression_research/figures/research_residuals.png')))'''),
      md('''## Results — novel scope support

No Train count is zero;6,278 Test rows have at least one scope/spread zero and all lie below the Train pressure minimum. Zero parses as numeric but its semantics are unknown. No zero-to-median replacement was applied. The no-scope validation ablation was not a consistent winner.

The unscored sensitivity probe sets three counts to zero and pressure below Train support on copied validation inputs. Its prediction deltas quantify model reliance under an artificial input scenario. Labels may no longer describe that altered record, so no perturbed MAE is computed and no claim is made about Test zero-cohort accuracy.'''),
      code('''shift=json.loads((ROOT/'reports/regression_research/shift_diagnostics.json').read_text(encoding='utf-8'))
sensitivity=json.loads((ROOT/'reports/regression_research/scope_sensitivity.json').read_text(encoding='utf-8'))
print('See SHIFT_DIAGNOSTICS.md for exact source identities and predictor-only evidence.')
display(pd.Series({k:v for k,v in sensitivity.items() if not isinstance(v,(dict,list))}))
print('Unscored sensitivity report:',ROOT/'reports/regression_research/scope_sensitivity.json')'''),
      md('''## Results — limited Kaggle feedback

Each shown additional candidate has a recorded recipe, full-Train fit, validated29,421-row CSV, input/output hashes and an at-most-once submission reservation. CSVs remain ignored. Public scores cover only the public subset; the user describes roughly30% public/70% private, and private scores remain unavailable. Candidate recipes were not refit to public labels, and remaining authorization is not a quota to consume.'''),
      code('''data=json.loads((ROOT/'reports/regression_research/report_data.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{'candidate':k,'public MAE':v,'Kaggle ref':data['kaggle_refs'].get(k)} for k,v in data['kaggle_scores'].items()]))
print('Official baseline (user supplied):12.23275')
display(Image(filename=str(ROOT/'reports/regression_research/figures/research_kaggle_scores.png')))'''),
      md('''## Interpretation, limitations & next steps

Read [FINDINGS.md](../../reports/regression_research/FINDINGS.md) for the final evidence-led review, counterexamples and recommended next step. [COMPARISON_TABLES.md](../../reports/regression_research/COMPARISON_TABLES.md) keeps all fixed and inner-selected candidates. The formal EDA's conditional shifts remain valid; predictive improvements do not identify the target-generation formula or resolve source semantics.

No permanent final model decision or Classification optimization follows automatically. Review H12 early risk, EF/tails, missing text and provided-batch assumptions alongside the public benchmark. Further work should be a single justified hypothesis with fixed guardrails, rather than small leaderboard-driven changes.

## Checks

The independent verifier recomputes mappings using pandas ranks and checks saved fit/validation row positions, fingerprints, inner choices, MAE/bias/tails/EF/cycle metrics and subgroup support. It does not refit models or read Test labels.'''),
      code('''checks=[]
for name in ['stage1','stage23','stage4','stage5']:
    r=json.loads((ROOT/f'reports/regression_research/verification_{name}.json').read_text(encoding='utf-8'))
    assert r['status']=='PASS'
    checks.append({'stage':name,'status':r['status'],'checks':str(r.get('checks',{}))[:260]})
display(pd.DataFrame(checks))
print('No training, upload, threshold selection or Test prediction occurs in this notebook.')''')]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    p=ROOT/'notebooks/samuel/08_samuel_regression_research.ipynb';nbformat.write(nb,p)
    NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.validate(nb);nbformat.write(nb,p);print(p)


if __name__=='__main__':main()
