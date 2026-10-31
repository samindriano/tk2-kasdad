"""Shared score-head mathematics with explicit structured/text input semantics."""
from src.models.optimization_v2_classification import TemporalScoreHead

class StructuredTextHead(TemporalScoreHead):
    def __init__(self,kind):
        if kind not in ['expanding_base','expanding_stack_cb']:raise ValueError('Unknown expanding head')
        super().__init__('calibrated_base'if kind=='expanding_base'else'stack_base_nb')
        self.recipe_id=kind

    def audit_state(self):
        result=super().audit_state()
        result['recipe_id']=self.recipe_id
        result['score_columns']=['C2_B present score','CatBoost structured score']
        result['kind']='two-logit-stack'if self.recipe_id=='expanding_stack_cb'else'one-logit-control'
        return result
