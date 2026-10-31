from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_cls_residual_fullsix import metric, train_edges


def test_metric_uses_fixed_half_threshold_and_class_f1s():
    y = np.array([0, 1, 0, 1], dtype=np.int8)
    p = np.array([0.49, 0.50, 0.8, 0.2])
    m = metric(y, p)
    assert m["threshold"] == 0.5
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (1, 1, 1, 1)
    assert m["macro_f1"] == 0.5
    assert m["n"] == 4


def test_fit_edges_use_only_cycles_at_or_before_cutoff():
    frame = pd.DataFrame({
        "cycle_code": [47, 48, 49, 50, 51, 52, 53],
        "entry_friction": [1., 2., 3., 4., 5., 1000., 2000.],
    })
    edges = train_edges(frame, 51, "entry_friction")
    np.testing.assert_allclose(edges, np.quantile([1., 2., 3., 4., 5.], [.25, .5, .75]))
    assert max(edges) < 1000


def test_sparse_and_residual_metrics_require_aligned_nonempty_vectors():
    try:
        metric(np.array([0, 1]), np.array([0.4]))
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned metric vectors should fail closed")
