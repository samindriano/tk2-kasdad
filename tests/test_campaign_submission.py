"""Mock transaction checks for the isolated live-quota campaign gate."""
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import json,tempfile,unittest
from unittest.mock import patch
from scripts import campaign_kaggle as gate

class CampaignSubmissionTests(unittest.TestCase):
    def test_shared_last_slot_is_preserved_and_budget_is_not_a_quota(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);out=root/'reports';config=root/'config';config.mkdir();out.mkdir()
            (config/'milestone.json').write_text(json.dumps({'research_upload_budget':{'classification':2}}))
            output=root/'candidate.csv';output.write_text('mock only');manifest=root/'candidate.json';manifest.write_text('{}')
            m={'candidate_id':'candidate','submission_message':'mock campaign'};calls=[]
            api=SimpleNamespace(competition_submit=lambda *a,**k:calls.append(a))
            live={'num_allowed_now':1,'num_total':7,'historical_submissions_without_other_scores':[]}
            with patch.multiple(gate,ROOT=root,OUT=out,CONFIG=config),patch.object(gate,'validate',return_value=(m,output)),patch.object(gate,'preflight',return_value=live):
                with self.assertRaisesRegex(AssertionError,'Leave at least one'):gate.upload(api,'classification',manifest)
            self.assertEqual(calls,[]);self.assertFalse(list(out.glob('*ledger.json')))

    def test_acceptance_timeout_reconciles_and_never_double_uploads(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);out=root/'reports';config=root/'config';config.mkdir();out.mkdir()
            (config/'milestone.json').write_text(json.dumps({'research_upload_budget':{'classification':2}}))
            output=root/'candidate.csv';output.write_text('mock only');manifest=root/'candidate.json';manifest.write_text('{}')
            m={'candidate_id':'candidate','submission_message':'mock campaign'};calls=[]
            result=SimpleNamespace(ref=9000001,date='mock',file_name=output.name,description=m['submission_message'],status='COMPLETE',team_name='kasdud lovers',public_score='.5')
            def accepted(*args,**kwargs):calls.append(args);raise OSError('simulated dropped response')
            api=SimpleNamespace(competition_submit=accepted,competition_submissions=lambda *a,**k:[result],get_submission=lambda ref:result)
            live={'num_allowed_now':8,'num_total':7,'historical_submissions_without_other_scores':[]}
            with patch.multiple(gate,ROOT=root,OUT=out,CONFIG=config),patch.object(gate,'validate',return_value=(m,output)),patch.object(gate,'preflight',return_value=live),redirect_stdout(StringIO()):
                with self.assertRaises(OSError):gate.upload(api,'classification',manifest)
                with self.assertRaises(AssertionError):gate.upload(api,'classification',manifest)
                gate.poll(api,'classification','candidate')
            self.assertEqual(len(calls),1);receipt=json.loads((out/'classification__candidate_ledger.json').read_text());self.assertEqual(receipt['status'],'COMPLETE')

    def test_campaign_budget_refuses_additional_candidate_reservation(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);out=root/'reports';config=root/'config';config.mkdir();out.mkdir()
            (config/'milestone.json').write_text(json.dumps({'research_upload_budget':{'regression':1}}))
            (out/'regression__first_ledger.json').write_text('{}')
            output=root/'second.csv';output.write_text('mock');manifest=root/'second.json';manifest.write_text('{}')
            with patch.multiple(gate,ROOT=root,OUT=out,CONFIG=config),patch.object(gate,'validate',return_value=({'candidate_id':'second'},output)):
                with self.assertRaises(AssertionError):gate.upload(SimpleNamespace(),'regression',manifest)

if __name__=='__main__':unittest.main()
