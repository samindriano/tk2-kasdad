import numpy as np
import pytest
from scripts.optimization_v2_prepare import mix

def test_bad_weights_or_row_alignment_fail_before_artifact():
    with pytest.raises(AssertionError):mix([np.array([1.,2.])],[{'weight':.7,'rank_weight':.5}],np.array([80,80]))
    with pytest.raises(AssertionError):mix([np.array([1.])],[{'weight':1.,'rank_weight':.5}],np.array([80,80]))
    with pytest.raises(AssertionError):mix([np.array([np.nan,2.])],[{'weight':1.,'rank_weight':.5}],np.array([80,80]))

def test_cycle_mapping_and_convex_clip_contract():
    p=mix([np.array([-5.,20.,200.]),np.array([10.,10.,10.])],
          [{'weight':.75,'rank_weight':0.},{'weight':.25,'rank_weight':0.}],np.array([80,80,81]))
    # The frozen component policy clips each mapped component before mixing.
    np.testing.assert_array_equal(p,[2.5,17.5,77.5])
