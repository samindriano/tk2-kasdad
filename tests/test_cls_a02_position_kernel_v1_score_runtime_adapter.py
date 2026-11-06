from __future__ import annotations

import importlib
import subprocess

import pytest

from scripts.score_cls_a02_position_kernel_v1_runtime_adapter_01 import (
    inject_subprocess_compatibility,
    unresolved_standard_module_names,
)


def test_only_frozen_scorer_subprocess_global_is_missing():
    assert unresolved_standard_module_names() == {"subprocess"}


def test_runtime_adapter_injects_only_subprocess_and_git_helper_works():
    scorer = importlib.import_module("scripts.score_cls_a02_position_kernel_v1")
    previous = getattr(scorer, "subprocess", None)
    try:
        if hasattr(scorer, "subprocess"):
            delattr(scorer, "subprocess")
        inject_subprocess_compatibility(scorer)
        assert scorer.subprocess is subprocess
        git_history = scorer.subprocess_run_git_rev_list().splitlines()
        assert git_history and len(git_history[0]) == 40
    finally:
        if previous is None:
            if hasattr(scorer, "subprocess"):
                delattr(scorer, "subprocess")
        else:
            scorer.subprocess = previous


def test_adapter_refuses_to_override_existing_scorer_global():
    scorer = importlib.import_module("scripts.score_cls_a02_position_kernel_v1")
    previous = getattr(scorer, "subprocess", None)
    try:
        scorer.subprocess = subprocess
        with pytest.raises(RuntimeError, match="refusing adapter injection"):
            inject_subprocess_compatibility(scorer)
    finally:
        if previous is None:
            delattr(scorer, "subprocess")
        else:
            scorer.subprocess = previous
