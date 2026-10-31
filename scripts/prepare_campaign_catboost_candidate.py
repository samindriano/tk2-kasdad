"""CatBoost factory adapter for the unchanged campaign preparation pipeline.

The first-wave preparation source is preserved after accepted submissions.
This explicit entry point supplies the actual frozen CatBoost factory API.
"""
from pathlib import Path
import argparse,sys,importlib.metadata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import prepare_campaign_candidate as preparation
from src.models.campaign_catboost import build_catboost_campaign

original_factory=preparation.component_model
original_exclusive=preparation.exclusive
def factory(task,component):
    if task=='regression'and component['model']=='catboost':return build_catboost_campaign()
    return original_factory(task,component)

def exclusive(path,document):
    if document.get('status')=='PREPARED_AND_QA_PASS':
        document['packages']['catboost']=importlib.metadata.version('catboost')
    return original_exclusive(path,document)

def main():
    p=argparse.ArgumentParser();p.add_argument('decision',type=Path);args=p.parse_args()
    decision=preparation.read(args.decision)
    assert decision['task']=='regression'and any(c['model']=='catboost'for c in decision['recipe']['components'])
    preparation.component_model=factory
    preparation.exclusive=exclusive
    preparation.prepare(args.decision)
if __name__=='__main__':main()
