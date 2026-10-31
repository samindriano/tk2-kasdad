"""Synthetic file-contract/transaction tests; no network or model scoring."""
from pathlib import Path
from types import SimpleNamespace
from enum import Enum
from contextlib import redirect_stdout
from io import StringIO
import csv,json,tempfile,unittest
from unittest.mock import patch
import scripts.autonomous_kaggle as gate

class Status(Enum):COMPLETE='COMPLETE'
class SubmissionTests(unittest.TestCase):
    def test_sdk_enum_normalizes_terminal_status(self):
        s=SimpleNamespace(ref=1,date='synthetic',file_name='unit.csv',description='unit',status=Status.COMPLETE,team_name='kasdud lovers',public_score='.5')
        self.assertEqual(gate.simple(s,True)['status'],'COMPLETE')

    def test_file_contract_rejects_fractional_class_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);out=root/'reports/autonomous_eda/kaggle';out.mkdir(parents=True)
            sub=root/'submissions/autonomous_eda';sub.mkdir(parents=True)
            config=root/'experiments/autonomous_eda';config.mkdir(parents=True)
            data=root/'data';data.mkdir()
            ids=[str(i)for i in range(29421)]
            for role in ['train','test','sample']:
                with (data/f'{role}.csv').open('w',newline='')as f:
                    w=csv.writer(f)
                    if role=='test':w.writerow(['record_id']);w.writerows((i,)for i in ids)
                    else:w.writerow(['record_id','artifact_signal']);w.writerows((i,0)for i in ids)
            path=sub/'unit.csv'
            def write(values):
                with path.open('w',newline='')as f:
                    w=csv.writer(f);w.writerow(['record_id','artifact_signal']);w.writerows(values)
            write([(i,0)for i in ids]);sel=config/'selection.json'
            (root/'fixture.py').write_text('# synthetic source')
            source={'fixture.py':gate.sha(root/'fixture.py')}
            selected={'candidate_id':'unit','eligible_for_external_evaluation':True,'recipe':{'kind':'synthetic'},'submission_message':'synthetic unit'}
            sel.write_text(json.dumps({'candidates':{'classification':[selected]},'source_code_sha256':source}))
            manifest=out/'unit.json'
            official={r:gate.sha(data/f'{r}.csv')for r in ['train','test','sample']}
            def save():
                m={'status':'PREPARED_AND_QA_PASS','task':'classification','competition':gate.slug('classification'),'candidate_id':'unit',
                    'main_selection':selected,'recipe':selected['recipe'],'submission_message':selected['submission_message'],
                    'selection_file_sha256':gate.sha(sel),'source_code_sha256':source,
                    'csv_path':'submissions/autonomous_eda/unit.csv','csv_sha256':gate.sha(path),'input_sha256':{r:gate.sha(data/f'{r}.csv')for r in ['train','test','sample']},
                    'input_relative_paths':{r:f'data/{r}.csv'for r in ['train','test','sample']},
                    'ordered_test_ids_sha256':__import__('hashlib').sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
                    'fit_preprocessing_only_train':True,'test_labels_used':False,'candidate_differs_from_accepted_baseline':True}
                manifest.write_text(json.dumps(m))
            with patch.multiple(gate,ROOT=root,OUT=out),patch.dict(gate.EXPECTED_INPUTS,{'classification':official}):
                save();gate.validate('classification',manifest)
                altered=json.loads(manifest.read_text());altered['recipe']={'kind':'unfrozen'};manifest.write_text(json.dumps(altered))
                with self.assertRaises(AssertionError):gate.validate('classification',manifest)
                save();altered=json.loads(manifest.read_text());altered['source_code_sha256']={};manifest.write_text(json.dumps(altered))
                with self.assertRaises(AssertionError):gate.validate('classification',manifest)
                bad=[(i,0)for i in ids];bad[10]=(ids[10],.5);write(bad);save()
                with self.assertRaises(AssertionError):gate.validate('classification',manifest)
                bad=[(i,0)for i in ids];bad[10]=(ids[9],0);write(bad);save()
                with self.assertRaises(AssertionError):gate.validate('classification',manifest)
                write([(i,0)for i in ids])
                # Even if an authority hash fixture is changed, Test labels are rejected independently.
                with (data/'test.csv').open('w',newline='')as f:
                    w=csv.writer(f);w.writerow(['record_id','artifact_signal']);w.writerows((i,0)for i in ids)
                official['test']=gate.sha(data/'test.csv');save()
                with self.assertRaisesRegex(AssertionError,'Test must not contain target columns'):gate.validate('classification',manifest)

    def test_two_upload_limit_and_idempotency_with_mock_api(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);out=root/'reports/autonomous_eda/kaggle';out.mkdir(parents=True)
            config=root/'experiments/autonomous_eda';config.mkdir(parents=True)
            (config/'milestone.json').write_text(json.dumps({'max_successful_new_submissions':{'classification':2}}))
            class FakeAPI:
                def __init__(self):self.calls=0;self.rows={}
                def competition_submit(self,file,message,competition,quiet=True):
                    self.calls+=1;ref=9000000+self.calls
                    self.rows[ref]=SimpleNamespace(ref=ref,date='synthetic',file_name=Path(file).name,description=message,status=Status.COMPLETE,team_name='kasdud lovers',public_score='.5')
                    return SimpleNamespace(ref=ref)
                def get_submission(self,ref):return self.rows[ref]
            api=FakeAPI()
            def files(candidate):
                p=root/f'{candidate}.csv';p.write_text('synthetic fixture')
                manifest=root/f'{candidate}.json';manifest.write_text('{}')
                return {'candidate_id':candidate,'submission_message':'synthetic '+candidate},p,manifest
            with patch.multiple(gate,ROOT=root,OUT=out),patch.object(gate,'preflight',return_value={'historical_submissions_without_other_scores':[],'num_total':0}),redirect_stdout(StringIO()):
                m,p,manifest=files('one')
                with patch.object(gate,'validate',return_value=(m,p)):
                    gate.upload(api,'classification',manifest)
                    with self.assertRaises(RuntimeError):gate.upload(api,'classification',manifest)
                m,p,manifest=files('two')
                with patch.object(gate,'validate',return_value=(m,p)):gate.upload(api,'classification',manifest)
                m,p,manifest=files('three')
                with patch.object(gate,'validate',return_value=(m,p)):
                    with self.assertRaises(AssertionError):gate.upload(api,'classification',manifest)
            self.assertEqual(api.calls,2)

    def test_ambiguous_acceptance_reconciles_without_second_upload(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);out=root/'reports/autonomous_eda/kaggle';out.mkdir(parents=True)
            config=root/'experiments/autonomous_eda';config.mkdir(parents=True)
            (config/'milestone.json').write_text(json.dumps({'max_successful_new_submissions':{'classification':2}}))
            p=root/'ambiguous.csv';p.write_text('synthetic');manifest=root/'ambiguous.json';manifest.write_text('{}')
            m={'candidate_id':'ambiguous','submission_message':'synthetic ambiguous'}
            receipt=SimpleNamespace(ref=9000999,date='synthetic',file_name=p.name,description=m['submission_message'],status=Status.COMPLETE,team_name='kasdud lovers',public_score='.5')
            calls=[]
            def accepted_then_connection_failed(*args,**kwargs):calls.append(args);raise OSError('synthetic connection failure after acceptance')
            api=SimpleNamespace(competition_submit=accepted_then_connection_failed,competition_submissions=lambda *a,**k:[receipt],get_submission=lambda ref:receipt)
            with patch.multiple(gate,ROOT=root,OUT=out),patch.object(gate,'validate',return_value=(m,p)),patch.object(gate,'preflight',return_value={'historical_submissions_without_other_scores':[],'num_total':0}),redirect_stdout(StringIO()):
                with self.assertRaises(OSError):gate.upload(api,'classification',manifest)
                with self.assertRaises(RuntimeError):gate.upload(api,'classification',manifest)
                gate.poll(api,'classification','ambiguous')
            state=json.loads((out/'classification__ambiguous_ledger.json').read_text())
            self.assertEqual(len(calls),1);self.assertEqual(state['status'],'COMPLETE');self.assertEqual(state['ref'],'9000999')

if __name__=='__main__':unittest.main()
