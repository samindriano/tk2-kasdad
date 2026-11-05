"""Saved-state, target-free E79 feature compatibility replay.

This checkpoint only verifies that frozen C2_B/A02 feature transforms still
reproduce the immutable full-Train matrix digests. It never fits an estimator,
builds a DF map, reads labels, scores predictions, or opens Test data.
"""
from __future__ import annotations

import argparse
import ast
import builtins
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import symtable
import time
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits

from src.models import classification_models as classification_models
from src.models import optimization_v3_positional_text_pilot_v1 as positional


AUTH_COMMIT = "71192b65c4ba0b1e31fd9273476d00b28f96b343"
AUTH_PATH = "reports/optimization_v3/cls_pos_global_strength_e79_v1/SOL_COMPATIBILITY_REPLAY_AUTHORIZATION.md"
AUTH_SHA256 = "9d0729d1f19793dfb79cc3024be868e44f8ff97020147cd0af2d91ff5df6b6e3"
ATTESTATION_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/SOURCE_COMPATIBILITY_ATTESTATION.json"
PROTOCOL_REL = "experiments/optimization_v3/cls_pos_global_strength_e79_v3/protocol.json"
CORRECTION_NOTE_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/CORRECTION_NOTE.md"
R1_ATTESTATION_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v1/SOURCE_COMPATIBILITY_ATTESTATION.json"
R1_ATTESTATION_FILE_SHA256 = "6839ef1228cfaffbefa80d9ede9fbddf09d589685d95faa83ee225af2c60703f"
R1_ATTEMPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v1/source_compatibility_replay/ATTEMPT.started.json"
R1_ATTEMPT_SHA256 = "9f9f33f573280cab6a57a7fa95e1e6baa17813f4e6a91e04b75d3f91af5de7c7"
R1_RECEIPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v1/source_compatibility_replay/REPLAY_RECEIPT.json"
R1_RECEIPT_SHA256 = "a754f0efb85f89bb5991d732518b78d4a3e0e765c77ae4303331c8b86262fdf8"
R1_RUNNER_REL = "scripts/optimization_v3_cls_pos_global_strength_e79_v1.py"
R1_RUNNER_SHA256 = "e3d4f30c15733e7ccd2ace4429277ea006794294bd5cf449681faad420dd9aa9"
R1_VERIFIER_REL = "scripts/verify_optimization_v3_cls_pos_global_strength_e79_v1.py"
R1_VERIFIER_SHA256 = "337fa46569647b6875b2e49b81d446144ce794a1637eceeea5a166ea1e04f350"
R1_TEST_REL = "tests/test_cls_pos_global_strength_e79_v1.py"
R1_TEST_SHA256 = "b63ccf49501118bdaf2f3e40aa82b25511b2171b5f9dfa6368f2492a88d03856"
R1_PROTOCOL_REL = "experiments/optimization_v3/cls_pos_global_strength_e79_v1/protocol.json"
R1_PROTOCOL_SHA256 = "a78e9e193df3fff51eca3c07ecc6a62bad4d17a973a5d847ee453b5b88fc56df"
R2_ATTESTATION_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v2/SOURCE_COMPATIBILITY_ATTESTATION.json"
R2_ATTESTATION_FILE_SHA256 = "7ee26bdb1f0a32dba081e15999f882a0334953d02ba3f17cbdb68af56346817e"
R2_ATTESTATION_INTERNAL_SHA256 = "efd340d5ced47ce4959cfc19c89ebc621a0223651621e4f8929673843c32a8b9"
R2_ATTEMPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v2/source_compatibility_replay/ATTEMPT.started.json"
R2_ATTEMPT_SHA256 = "2f175f4a3bbbb2b53898df0969a9eef7eb7d4599898281705c54ec8c0c8a8c49"
R2_RECEIPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v2/source_compatibility_replay/REPLAY_RECEIPT.json"
R2_RECEIPT_SHA256 = "1a001f7843dfa19356adab62b0eadd21aecf1b0be8fa83f3db63a9a354c68ff9"
R2_RECEIPT_INTERNAL_SHA256 = "7245c94a097f3440fa43938b2d798646f7d31d2f1324354144867cae54762031"
R2_RUNNER_REL = "scripts/optimization_v3_cls_pos_global_strength_e79_v2.py"
R2_RUNNER_SHA256 = "08654eab445b5b3e51574dfa4d7d416f3684670d4dc3722cb821ea44a4d1c59e"
R2_VERIFIER_REL = "scripts/verify_optimization_v3_cls_pos_global_strength_e79_v2.py"
R2_VERIFIER_SHA256 = "ac06d3f4136d93d74638b2535aef728f5f14ed9a970bca9f1f03da2753770b8e"
R2_TEST_REL = "tests/test_cls_pos_global_strength_e79_v2.py"
R2_TEST_SHA256 = "c84eb32d6957f9b9a0041f45ae7038c2f2feb16905a4205119029ff911cbbeb7"
R2_PROTOCOL_REL = "experiments/optimization_v3/cls_pos_global_strength_e79_v2/protocol.json"
R2_PROTOCOL_SHA256 = "6cdd19cfdce8351470bf5fd2f3549cf5794a6fc382247c69b05cfa29420c81a9"
R2_CORRECTION_NOTE_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v2/CORRECTION_NOTE.md"
R2_CORRECTION_NOTE_SHA256 = "870b55b360a70486e3efcc0e362532ce4fe76367f376b9ca5ae2a3d4df0e9406"
TRAIN_REL = "data/raw/classification/Train.csv"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_ROWS = 50_608
EXPECTED_CYCLES = tuple(range(47, 80))
BASE_SHAPE = (50_608, 20_427)
BASE_NNZ = 3_855_766
BASE_SHA256 = "9d47aa36ff15a0cbeead8cf36049f0a699f7618d0ea8890806a2d82140cff59b"
POSITION_SHAPE = (50_608, 17_259)
POSITION_NNZ = 1_638_979
POSITION_SHA256 = "32268550b8220dbfb43f5c92c33cd84554f0bced446d1459c5c55c3eabbf9942"
ORDERED_TRAIN_IDS_SHA256 = "9a96fa28e72b08a6022dd005aed877dadcc5e3dd5053200bcdd12711312c26b2"
VOCABULARY_SIZE = 5_753
POSITION_COLUMNS = "three segment-major blocks: first, middle, final; within each block saved A02 vocabulary-index order filtered to unigrams"

A02_MODEL_REL = "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/model.joblib"
A02_MODEL_SHA256 = "2791eddf410c7e81b119bde2a0b1d33a694eed4de84027c61cee97f6668dcb6e"
A02_RECEIPT_REL = "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/fit_receipt.json"
A02_RECEIPT_SHA256 = "11a7e46b6595e92a421e358b8e7e3a5ffbac8a6db69c64c9c33fd3d5c5bd7d77"
CGEOM_PREFLIGHT_REL = "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/INTEGRATION_PREFLIGHT_R1.json"
CGEOM_PREFLIGHT_SHA256 = "087e1eb82fa65f046aebfb57fd365d95dc417bdf6841b250b1b60440f506e243"
UNIT_ROOT_REL = "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/units"
CGEOM_RUN_ROOT_REL = "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1"
RUN_ROOT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/source_compatibility_replay"
ATTEMPT_REL = f"{RUN_ROOT_REL}/ATTEMPT.started.json"
REPLAY_RECEIPT_REL = f"{RUN_ROOT_REL}/REPLAY_RECEIPT.json"

PRODUCERS = {
    "structured": "88ae68653a02898051a2ac12f780e8b12ec9c2d4",
    "scaler_ohe": "4480f74ac909da4641fb9278dd9b536f66d8889d",
}
CGEOM_SCRIPT_REL = "scripts/optimization_v3_a02_cgeom_benchmark_v1.py"
CGEOM_TEST_REL = "tests/test_optimization_v3_a02_cgeom_benchmark_v1.py"
EXPECTED_PRODUCER_BLOBS = {
    PRODUCERS["structured"]: {
        CGEOM_SCRIPT_REL: "abc348edbace18cc0b6c08a78746a07cbbafbf694fdd3f1b9a5c0fb513281e5a",
        CGEOM_TEST_REL: "15c7fd2e11a18147bde05a39dbc44c1d43497b633379102dc9d401d28c26298d",
    },
    PRODUCERS["scaler_ohe"]: {
        CGEOM_SCRIPT_REL: "ea011537a29ffd75aa7695dfb7df7a0bc4af4a311cf05158b6b2a8c53a03d694",
        CGEOM_TEST_REL: "ed96e928e1da80fcd8cd5b57e56975a6ac3b524bab2cd4e534dd7554a3d930a5",
    },
}
EXPECTED_FIT_PAYLOAD_BODY_SHA256 = "1efc9576dbc0ec08b4c29ee66951a3b0d78a1b1d269170c751d2c3dc08e3b698"
HELPER_FILE_PINS = {
    "src/models/optimization_v3_positional_text_pilot_v1.py": "390d9585a91440cc83314d2fa58acfebf374371032308937489019c2bd242c36",
    "src/models/classification_models.py": "6fa123f5da93c2d1d1fc2d2eae0a04ead2988848123ece9b206e97ae5021c646",
    "src/models/round1_models.py": "766e0c2e183c5aa653a5c5b81aae4d646e99cecdcd043bead3eadb99edd0a7a4",
    "src/models/positional_text_features.py": "d4ef29f5b8942e04f221cdad5e9d93f7a96f76f485cae8bdc80fbbc8557cb24e",
}
HELPER_FUNCTION_PINS = {
    "src/models/optimization_v3_positional_text_pilot_v1.py": {
        "fit_c2b_structured_state": "27b87a0098dde702edb92262381c69d96f439a9e4be1d6b8801957b3fc3a0d30",
        "fit_c2b_scaler": "c8167da0ecf7c0dfd6c5bf85e2e7d901b943cae53fee276fe2ff4b802074d21d",
        "_c2b_category_tokens": "3e8560cb02f59e00506049384e1f0dd627309a3edc1faaeb45dbebceeea1eded",
        "fit_c2b_ohe": "70a1b1bed24a7a8e723269e188fe2e0d62376c8d2581cbaad3b3b0da8e359b75",
        "fit_c2b_tfidf": "3544c5e7cf0c00fe07aed75726cee081e13a6631cd291e83b542b80124701912",
        "transform_c2b_structured": "c031d695f7ca82b192fbe37daa280d98f3673c740479bfef75fe9bbd6a6aada6",
        "transform_c2b_text": "9912fb731e527e7e3c863cce671dc74680941463210a7a946fce5126b0712a60",
        "build_position_block": "0d7bc6da82869e7d57263368c1fbeafccac9adfd6c6f0da7ce1869d9f06cd5d5",
        "validate_feature_frame": "79b0948f4d7c9683882d884e8bdf0250b14a925b7f2187d0e09014d42709e950",
    }
}

UNIT_ASSETS = {
    "c2b_lr": (
        "A02_CGEOM_BENCHMARK_V1_E79_C2B_LR_INTEGRATION_R1",
        "40b9588e1cd2f93350fe09ffc69efa9bf0bcb6927e6b8a2bc0192a4a6235ca80",
        "d076a1b2bcc29f14aa4be9c6d7621f4b0217b79f528ea9a376765329f4b2c9b9",
    ),
    "c2b_structured": (
        "A02_CGEOM_BENCHMARK_V1_E79_C2B_STRUCTURED",
        "3347262f1df86a70011bebe01e8ec7df38bff1aedf0da8edbf30ab4d4c3a537a",
        "8da5f7ae163577aaecc8db94fd6130188ffeb270e52a30b8cd5925dce8414840",
    ),
    "c2b_scaler": (
        "A02_CGEOM_BENCHMARK_V1_E79_C2B_SCALER",
        "d45db93e7a02c7b3c2662347d0b2b31a87838df3080b0b6150856c75df4883d9",
        "183df7d8a8f6f6936d0d832f293a26e627d9d923dc626b1fb3ec773bcbb2af1b",
    ),
    "c2b_ohe": (
        "A02_CGEOM_BENCHMARK_V1_E79_C2B_OHE",
        "85c6f0f7fb4b318d3ffd11bff5f4944cba8a7ea1124355b493d0e92509c3bb77",
        "c86e8682fda527dee706231f12078c2b4839c0acdab7b05cd1224daba5e22fae",
    ),
    "uniform_pos": (
        "A02_CGEOM_BENCHMARK_V1_E79_POS_BAG_OFFSET_GAMMA_INTEGRATION_R1",
        "d741d016f990d8742208edc919f5c58305b7579980d059b202609df4e5b3e650",
        "f08f1bd3733a20fb8745d40c82190543a48af6bade1df5684ccfef3198c34580",
    ),
}

EXPLICIT_SOURCE_VARIANCES = {
    "c2b_structured": {
        CGEOM_SCRIPT_REL: {
            "receipt_expected": "abc348edbace18cc0b6c08a78746a07cbbafbf694fdd3f1b9a5c0fb513281e5a",
            "current_expected": "d6af3f3e18787c9cd1e5c52b1463fb30e18398cee6e3dc838e5227cf30a50a3f",
        },
        CGEOM_TEST_REL: {
            "receipt_expected": "15c7fd2e11a18147bde05a39dbc44c1d43497b633379102dc9d401d28c26298d",
            "current_expected": "9a651df8fed669c10865244ad5fa60f37039c130d68a22c0bf54007f17ba298a",
        },
    },
    "c2b_scaler": {
        CGEOM_SCRIPT_REL: {
            "receipt_expected": "ea011537a29ffd75aa7695dfb7df7a0bc4af4a311cf05158b6b2a8c53a03d694",
            "current_expected": "d6af3f3e18787c9cd1e5c52b1463fb30e18398cee6e3dc838e5227cf30a50a3f",
        },
        CGEOM_TEST_REL: {
            "receipt_expected": "ed96e928e1da80fcd8cd5b57e56975a6ac3b524bab2cd4e534dd7554a3d930a5",
            "current_expected": "9a651df8fed669c10865244ad5fa60f37039c130d68a22c0bf54007f17ba298a",
        },
    },
    "c2b_ohe": {
        CGEOM_SCRIPT_REL: {
            "receipt_expected": "ea011537a29ffd75aa7695dfb7df7a0bc4af4a311cf05158b6b2a8c53a03d694",
            "current_expected": "d6af3f3e18787c9cd1e5c52b1463fb30e18398cee6e3dc838e5227cf30a50a3f",
        },
        CGEOM_TEST_REL: {
            "receipt_expected": "ed96e928e1da80fcd8cd5b57e56975a6ac3b524bab2cd4e534dd7554a3d930a5",
            "current_expected": "9a651df8fed669c10865244ad5fa60f37039c130d68a22c0bf54007f17ba298a",
        },
    },
}

INCREMENTAL_ESTIMATE_BYTES = 1024**3
FREE_RESERVE_BYTES = 768 * 1024**2
MIN_AVAILABLE_BYTES = INCREMENTAL_ESTIMATE_BYTES + FREE_RESERVE_BYTES
MAX_THREADS = 2


class ReplayError(RuntimeError):
    """Fail-closed source, input, or matrix parity error."""


class SharedLockHeld(ReplayError):
    """The existing process-level advisory lock could not be acquired."""


class LocalLockHeld(ReplayError):
    """The CGeom per-run exclusive lock could not be acquired."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return positional.canonical_sha256(value)


def unresolved_global_references(source: str, *, filename: str = "<source>") -> list[dict[str, str]]:
    """Find referenced globals not bound by the module, builtins, or Python's module loader."""
    ast.parse(source, filename=filename)
    root_table = symtable.symtable(source, filename, "exec")
    module_bindings = {
        symbol.get_name()
        for symbol in root_table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    allowed = set(dir(builtins)) | {"__file__", "__package__", "__loader__", "__spec__"}
    findings: set[tuple[str, str, str]] = set()

    def visit(table: symtable.SymbolTable) -> None:
        for symbol in table.get_symbols():
            name = symbol.get_name()
            if (symbol.is_referenced() and symbol.is_global()
                    and name not in module_bindings and name not in allowed):
                findings.add((table.get_name(), table.get_type(), name))
        for child in table.get_children():
            visit(child)

    visit(root_table)
    return [
        {"scope": scope, "scope_type": scope_type, "name": name}
        for scope, scope_type, name in sorted(findings)
    ]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayError(f"Expected JSON object at {path}")
    return value


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True,
        capture_output=True, text=True, encoding="utf-8",
    )
    return result.stdout.strip()


def _git_blob(root: Path, commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative}"],
        check=True, capture_output=True,
    ).stdout


def _ast_node(tree: ast.AST, qualified_name: str) -> ast.AST:
    parts = qualified_name.split(".")
    current: ast.AST = tree
    for index, name in enumerate(parts):
        body = getattr(current, "body", None)
        matches = [
            node for node in body or []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ]
        if len(matches) != 1:
            raise ReplayError(f"AST symbol not uniquely found: {qualified_name}")
        current = matches[0]
        if index < len(parts) - 1 and not isinstance(current, ast.ClassDef):
            raise ReplayError(f"AST path crosses non-class symbol: {qualified_name}")
    return current


def _source_symbol_hash(source: str, qualified_name: str) -> str:
    node = _ast_node(ast.parse(source), qualified_name)
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ReplayError(f"Could not extract source segment: {qualified_name}")
    return sha256_bytes(segment.encode("utf-8"))


def _require_hash_match(label: str, expected: str, actual: str) -> None:
    if expected != actual:
        raise ReplayError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _hash_map(root: Path, entries: Mapping[str, str], label: str) -> dict[str, str]:
    actuals: dict[str, str] = {}
    for relative, expected in entries.items():
        path = root / relative
        if not path.is_file():
            raise ReplayError(f"{label} file is missing: {relative}")
        actual = sha256_file(path)
        _require_hash_match(f"{label} {relative}", expected, actual)
        actuals[relative] = actual
    return actuals


def _verify_receipt_integrity(receipt: dict[str, Any], *, label: str) -> None:
    body = dict(receipt)
    supplied = body.pop("receipt_sha256", None)
    if supplied != canonical_sha256(body):
        raise ReplayError(f"{label} internal receipt digest is invalid")


def _receipt_source_audit(root: Path, key: str, receipt: dict[str, Any]) -> dict[str, Any]:
    declared = receipt.get("source_sha256")
    if not isinstance(declared, dict) or not declared:
        raise ReplayError(f"Source list missing from saved receipt: {key}")
    exceptions = EXPLICIT_SOURCE_VARIANCES.get(key, {})
    if not set(exceptions).issubset(declared):
        raise ReplayError(f"Authorized source-variance entries are absent from receipt: {key}")
    audited: dict[str, Any] = {}
    for relative, expected in declared.items():
        path = root / relative
        if not path.is_file():
            raise ReplayError(f"Receipt source file is missing for {key}: {relative}")
        actual = sha256_file(path)
        exception = exceptions.get(relative)
        if exception is None:
            _require_hash_match(f"{key} frozen source {relative}", expected, actual)
            audited[relative] = {
                "receipt_expected_sha256": expected,
                "current_sha256": actual,
                "authorized_variance": False,
            }
        else:
            _require_hash_match(
                f"{key} receipt's exact historical source {relative}",
                exception["receipt_expected"], expected,
            )
            _require_hash_match(
                f"{key} exact authorized current source {relative}",
                exception["current_expected"], actual,
            )
            audited[relative] = {
                "receipt_expected_sha256": expected,
                "current_sha256": actual,
                "authorized_variance": True,
                "variance_scope": "this unit receipt and this one exact source path only",
            }
    return audited


def _verify_fulltrain_unit(root: Path, key: str) -> dict[str, Any]:
    fit_id, artifact_sha, receipt_sha = UNIT_ASSETS[key]
    stem = f"{UNIT_ROOT_REL}/{fit_id}"
    artifact_rel, receipt_rel = f"{stem}.joblib", f"{stem}.json"
    _hash_map(root, {artifact_rel: artifact_sha, receipt_rel: receipt_sha}, f"saved unit {key}")
    receipt = read_json(root / receipt_rel)
    _verify_receipt_integrity(receipt, label=key)
    if (
        receipt.get("fit_id") != fit_id
        or receipt.get("artifact_sha256") != artifact_sha
        or receipt.get("fit_population_rows") != EXPECTED_ROWS
        or receipt.get("fit_cycle_min_max") != [47, 79]
        or receipt.get("train_sha256") != TRAIN_SHA256
        or receipt.get("synthetic") is not False
        or receipt.get("test_data_used_for_fit") is not False
        or receipt.get("validation_or_test_labels_passed_to_fit") is not False
    ):
        raise ReplayError(f"Saved full-Train unit lineage failed: {key}")
    return {
        "fit_id": fit_id,
        "artifact_path": artifact_rel,
        "artifact_sha256": artifact_sha,
        "receipt_path": receipt_rel,
        "receipt_sha256": receipt_sha,
        "state_sha256": receipt.get("state_sha256"),
        "fit_population_rows": receipt["fit_population_rows"],
        "fit_cycle_min_max": receipt["fit_cycle_min_max"],
        "source_sha256_audit": _receipt_source_audit(root, key, receipt),
    }


def _verify_saved_inputs(root: Path) -> dict[str, Any]:
    immutable: dict[str, str] = {
        TRAIN_REL: TRAIN_SHA256,
        A02_MODEL_REL: A02_MODEL_SHA256,
        A02_RECEIPT_REL: A02_RECEIPT_SHA256,
        CGEOM_PREFLIGHT_REL: CGEOM_PREFLIGHT_SHA256,
        AUTH_PATH: AUTH_SHA256,
        R1_ATTESTATION_REL: R1_ATTESTATION_FILE_SHA256,
        R1_ATTEMPT_REL: R1_ATTEMPT_SHA256,
        R1_RECEIPT_REL: R1_RECEIPT_SHA256,
        R1_RUNNER_REL: R1_RUNNER_SHA256,
        R1_VERIFIER_REL: R1_VERIFIER_SHA256,
        R1_TEST_REL: R1_TEST_SHA256,
        R1_PROTOCOL_REL: R1_PROTOCOL_SHA256,
        R2_ATTESTATION_REL: R2_ATTESTATION_FILE_SHA256,
        R2_ATTEMPT_REL: R2_ATTEMPT_SHA256,
        R2_RECEIPT_REL: R2_RECEIPT_SHA256,
        R2_RUNNER_REL: R2_RUNNER_SHA256,
        R2_VERIFIER_REL: R2_VERIFIER_SHA256,
        R2_TEST_REL: R2_TEST_SHA256,
        R2_PROTOCOL_REL: R2_PROTOCOL_SHA256,
        R2_CORRECTION_NOTE_REL: R2_CORRECTION_NOTE_SHA256,
    }
    units = {key: _verify_fulltrain_unit(root, key) for key in UNIT_ASSETS}
    for unit in units.values():
        immutable[unit["artifact_path"]] = unit["artifact_sha256"]
        immutable[unit["receipt_path"]] = unit["receipt_sha256"]
    immutable_actual = _hash_map(root, immutable, "immutable input")

    r1_attestation = read_json(root / R1_ATTESTATION_REL)
    r1_attestation_body = dict(r1_attestation)
    r1_attestation_digest = r1_attestation_body.pop("attestation_sha256", None)
    r1_receipt = read_json(root / R1_RECEIPT_REL)
    r1_receipt_body = dict(r1_receipt)
    r1_receipt_digest = r1_receipt_body.pop("receipt_sha256", None)
    r1_attempt = read_json(root / R1_ATTEMPT_REL)
    if (
        r1_attestation_digest != "81e9379da76627283672cd0ee62a4d5584ffe29c4cd40857c09c8c628f31a127"
        or r1_attestation_digest != canonical_sha256(r1_attestation_body)
        or r1_receipt.get("status") != "FAILED_FEATURE_ONLY_REPLAY_NO_RETRY"
        or r1_receipt_digest != canonical_sha256(r1_receipt_body)
        or r1_receipt.get("source_commit") != "7766a9e833e54402e534c52240cffd282e4a6b1e"
        or r1_receipt.get("error") != "Feature-only Train read returned unexpected or duplicate columns"
        or r1_receipt.get("target_column_requested_or_loaded") is not False
        or r1_receipt.get("test_or_sample_loaded") is not False
        or r1_receipt.get("new_df_map_created") is not False
        or r1_receipt.get("new_gamma_created") is not False
        or r1_attempt.get("status") != "ONE_REPLAY_ATTEMPT_STARTED"
        or r1_attempt.get("no_retry_if_interrupted") is not True
    ):
        raise ReplayError("Immutable R1 failed-attempt evidence changed or failed its integrity checks")

    r2_attestation = read_json(root / R2_ATTESTATION_REL)
    r2_attestation_body = dict(r2_attestation)
    r2_attestation_digest = r2_attestation_body.pop("attestation_sha256", None)
    r2_attempt = read_json(root / R2_ATTEMPT_REL)
    r2_receipt = read_json(root / R2_RECEIPT_REL)
    r2_receipt_body = dict(r2_receipt)
    r2_receipt_digest = r2_receipt_body.pop("receipt_sha256", None)
    from scripts.verify_optimization_v3_cls_pos_global_strength_e79_v2 import verify_attestation as verify_r2_attestation
    verify_r2_attestation(root)
    if (
        r2_attestation_digest != R2_ATTESTATION_INTERNAL_SHA256
        or r2_attestation_digest != canonical_sha256(r2_attestation_body)
        or r2_attestation.get("status") != "PASS_SOURCE_COMPATIBILITY_ATTESTATION_R2_PENDING_REPLAY"
        or r2_attempt.get("status") != "ONE_REPLAY_ATTEMPT_STARTED"
        or r2_attempt.get("no_retry_if_interrupted") is not True
        or r2_receipt.get("status") != "FAILED_FEATURE_ONLY_REPLAY_NO_RETRY"
        or r2_receipt.get("source_commit") != "71192b65c4ba0b1e31fd9273476d00b28f96b343"
        or r2_receipt.get("error_type") != "NameError"
        or r2_receipt.get("error") != "name 'base_sha' is not defined"
        or r2_receipt_digest != R2_RECEIPT_INTERNAL_SHA256
        or r2_receipt_digest != canonical_sha256(r2_receipt_body)
        or r2_receipt.get("target_column_requested_or_loaded") is not False
        or r2_receipt.get("test_or_sample_loaded") is not False
        or r2_receipt.get("estimator_fit_calls") != 0
        or r2_receipt.get("prediction_or_inference_calls") != 0
        or r2_receipt.get("new_df_map_created") is not False
        or r2_receipt.get("new_gamma_created") is not False
        or r2_receipt.get("csv_created") is not False
        or "matrices" in r2_receipt
    ):
        raise ReplayError("Immutable R2 failed-attempt evidence changed or crossed a forbidden boundary")

    preflight = read_json(root / CGEOM_PREFLIGHT_REL)
    matrix = preflight.get("fulltrain_feature_matrix_audit", {})
    compatibility = preflight.get("a02_vectorizer_compatibility", {})
    reuse = preflight.get("existing_model_reuse_audit", {})
    if (
        matrix.get("status") != "PASS_REAL_PRODUCTION_FULLTRAIN_FEATURE_MATRICES"
        or matrix.get("train_sha256") != TRAIN_SHA256
        or matrix.get("train_rows") != EXPECTED_ROWS
        or matrix.get("target_column_loaded") is not False
        or matrix.get("test_or_sample_loaded") is not False
        or matrix.get("base_shape") != list(BASE_SHAPE)
        or matrix.get("base_nnz") != BASE_NNZ
        or matrix.get("base_matrix_sha256") != BASE_SHA256
        or matrix.get("position_shape") != list(POSITION_SHAPE)
        or matrix.get("position_nnz") != POSITION_NNZ
        or matrix.get("position_matrix_sha256") != POSITION_SHA256
        or matrix.get("ordered_train_ids_sha256") != ORDERED_TRAIN_IDS_SHA256
        or compatibility.get("status") != "PASS_A02_FULLTRAIN_TFIDF_FEATURE_COMPATIBILITY"
        or compatibility.get("target_column_requested_or_loaded") is not False
        or compatibility.get("uses_test_or_sample") is not False
        or compatibility.get("train_sha256") != TRAIN_SHA256
        or compatibility.get("train_rows") != EXPECTED_ROWS
        or reuse.get("a02_model_sha256") != A02_MODEL_SHA256
    ):
        raise ReplayError("Immutable C2_B/A02 feature audit does not match all matrix and source pins")
    if not matrix.get("checks") or not all(matrix["checks"].values()):
        raise ReplayError("Immutable matrix audit has a failed or missing check")

    a02_receipt = read_json(root / A02_RECEIPT_REL)
    if (
        a02_receipt.get("model_sha256") != A02_MODEL_SHA256
        or a02_receipt.get("origin_cycle") != 79
        or a02_receipt.get("text_prefix_rows") != EXPECTED_ROWS
        or a02_receipt.get("text_prefix_row_positions_sha256")
        != compatibility.get("fit_row_positions_sha256")
        or a02_receipt.get("query_labels_passed_to_fit") is not False
        or a02_receipt.get("test_data_used") is not False
        or a02_receipt.get("preprocessing_fit_population")
        != "official Train feature-text prefix cycles 47-79; no target values"
        or a02_receipt.get("state_sha256") != reuse.get("a02_fit_state_sha256")
    ):
        raise ReplayError("Saved A02 full-Train state/feature-text lineage is inconsistent")
    c2b_lr = units["c2b_lr"]
    uniform_pos = units["uniform_pos"]
    if (
        c2b_lr["artifact_sha256"] != "40b9588e1cd2f93350fe09ffc69efa9bf0bcb6927e6b8a2bc0192a4a6235ca80"
        or uniform_pos["artifact_sha256"] != "d741d016f990d8742208edc919f5c58305b7579980d059b202609df4e5b3e650"
    ):
        raise ReplayError("Saved C2_B or uniform POS reference state changed")
    return {
        "immutable_file_sha256": immutable_actual,
        "preserved_r1_attempt": {
            "status": r1_receipt["status"],
            "attempt_marker_sha256": R1_ATTEMPT_SHA256,
            "replay_receipt_sha256": R1_RECEIPT_SHA256,
            "attestation_file_sha256": R1_ATTESTATION_FILE_SHA256,
            "attestation_internal_sha256": r1_attestation_digest,
            "runner_sha256": R1_RUNNER_SHA256,
            "verifier_sha256": R1_VERIFIER_SHA256,
            "test_sha256": R1_TEST_SHA256,
            "protocol_sha256": R1_PROTOCOL_SHA256,
        },
        "preserved_r2_failed_attempt": {
            "status": r2_receipt["status"],
            "attempt_marker_sha256": R2_ATTEMPT_SHA256,
            "replay_receipt_file_sha256": R2_RECEIPT_SHA256,
            "replay_receipt_internal_digest_present": True,
            "replay_receipt_internal_sha256": r2_receipt_digest,
            "attestation_file_sha256": R2_ATTESTATION_FILE_SHA256,
            "attestation_internal_sha256": r2_attestation_digest,
            "runner_sha256": R2_RUNNER_SHA256,
            "verifier_sha256": R2_VERIFIER_SHA256,
            "test_sha256": R2_TEST_SHA256,
            "protocol_sha256": R2_PROTOCOL_SHA256,
            "correction_note_sha256": R2_CORRECTION_NOTE_SHA256,
            "failure": r2_receipt["error"],
            "matrix_digests_persisted": False,
        },
        "saved_units": units,
        "preflight_audit_sha256": CGEOM_PREFLIGHT_SHA256,
        "matrix_reference": {
            "base_shape": list(BASE_SHAPE),
            "base_nnz": BASE_NNZ,
            "base_logical_sha256": BASE_SHA256,
            "position_shape": list(POSITION_SHAPE),
            "position_nnz": POSITION_NNZ,
            "position_logical_sha256": POSITION_SHA256,
            "ordered_train_ids_sha256": ORDERED_TRAIN_IDS_SHA256,
            "feature_order": POSITION_COLUMNS,
        },
        "a02_compatibility_reference": {
            name: compatibility.get(name)
            for name in (
                "status", "vocabulary_sha256", "idf_sha256", "fit_documents_sha256",
                "fit_row_positions_sha256", "target_column_requested_or_loaded", "uses_test_or_sample",
            )
        },
        "a02_reuse_state_sha256": reuse.get("a02_fit_state_sha256"),
    }


def _producer_provenance(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, commit in PRODUCERS.items():
        expected_blobs = EXPECTED_PRODUCER_BLOBS[commit]
        actual_blobs: dict[str, str] = {}
        for relative, expected in expected_blobs.items():
            actual = sha256_bytes(_git_blob(root, commit, relative))
            _require_hash_match(f"producer {commit}:{relative}", expected, actual)
            actual_blobs[relative] = actual
        cgeom_bytes = _git_blob(root, commit, CGEOM_SCRIPT_REL)
        cgeom_source = cgeom_bytes.decode("utf-8")
        body_hash = _source_symbol_hash(cgeom_source, "_make_fit_payload")
        _require_hash_match("historical producer _make_fit_payload body", EXPECTED_FIT_PAYLOAD_BODY_SHA256, body_hash)
        helper_blobs: dict[str, str] = {}
        for relative, expected in HELPER_FILE_PINS.items():
            actual = sha256_bytes(_git_blob(root, commit, relative))
            _require_hash_match(f"producer numerical helper {commit}:{relative}", expected, actual)
            helper_blobs[relative] = actual
        result[role] = {
            "commit": commit,
            "producer_blob_sha256": actual_blobs,
            "make_fit_payload_body_sha256": body_hash,
            "numerical_helper_blob_sha256": helper_blobs,
        }
    return result


def build_source_compatibility_attestation(root: Path = ROOT) -> dict[str, Any]:
    head = _git(root, "rev-parse", "HEAD")
    _require_hash_match("authorization parent HEAD", AUTH_COMMIT, head)
    saved = _verify_saved_inputs(root)
    producers = _producer_provenance(root)

    current_sources: dict[str, str] = {
        "scripts/optimization_v3_cls_pos_global_strength_e79_v3.py": sha256_file(root / "scripts/optimization_v3_cls_pos_global_strength_e79_v3.py"),
        "scripts/verify_optimization_v3_cls_pos_global_strength_e79_v3.py": sha256_file(root / "scripts/verify_optimization_v3_cls_pos_global_strength_e79_v3.py"),
        PROTOCOL_REL: sha256_file(root / PROTOCOL_REL),
        CORRECTION_NOTE_REL: sha256_file(root / CORRECTION_NOTE_REL),
        "tests/test_cls_pos_global_strength_e79_v3.py": sha256_file(root / "tests/test_cls_pos_global_strength_e79_v3.py"),
        CGEOM_SCRIPT_REL: sha256_file(root / CGEOM_SCRIPT_REL),
        CGEOM_TEST_REL: sha256_file(root / CGEOM_TEST_REL),
        **{relative: sha256_file(root / relative) for relative in HELPER_FILE_PINS},
    }
    for relative in (
        "scripts/optimization_v3_cls_pos_global_strength_e79_v3.py",
        "scripts/verify_optimization_v3_cls_pos_global_strength_e79_v3.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        unresolved = unresolved_global_references(source, filename=relative)
        if unresolved:
            raise ReplayError(f"Unresolved global reference in {relative}: {unresolved}")
    for relative, expected in HELPER_FILE_PINS.items():
        _require_hash_match(f"current numerical helper {relative}", expected, current_sources[relative])
    _require_hash_match("current CGeom script source variance", EXPLICIT_SOURCE_VARIANCES["c2b_structured"][CGEOM_SCRIPT_REL]["current_expected"], current_sources[CGEOM_SCRIPT_REL])
    _require_hash_match("current CGeom test source variance", EXPLICIT_SOURCE_VARIANCES["c2b_structured"][CGEOM_TEST_REL]["current_expected"], current_sources[CGEOM_TEST_REL])

    symbol_sources = {
        "scripts/optimization_v3_cls_pos_global_strength_e79_v3.py": [
            "canonical_feature_columns", "validate_feature_frame_columns",
            "read_feature_csv", "assemble_and_validate_matrices",
            "unresolved_global_references", "_execute_feature_only_replay",
            "write_and_readback_receipt",
        ],
        "src/models/optimization_v3_positional_text_pilot_v1.py": [
            "validate_feature_frame", "transform_c2b_structured", "transform_c2b_text",
            "build_position_block", "_c2b_category_tokens", "structured_state_sha256",
            "canonical_sha256", "sha256_array", "sha256_csr",
        ],
        "src/models/classification_models.py": ["_text_values", "ClassificationModel._category_tokens"],
        "src/models/round1_models.py": ["_HGBPreprocessor._raw_matrix", "_HGBPreprocessor.transform"],
        "src/models/positional_text_features.py": ["token_thirds"],
        CGEOM_SCRIPT_REL: ["audit_c2b_vectorizer_compatibility", "_state_summary"],
    }
    symbols: dict[str, str] = {}
    for relative, names in symbol_sources.items():
        source = (root / relative).read_text(encoding="utf-8")
        for name in names:
            actual = _source_symbol_hash(source, name)
            pin = HELPER_FUNCTION_PINS.get(relative, {}).get(name)
            if pin is not None:
                _require_hash_match(f"numerical helper {relative}:{name}", pin, actual)
            symbols[f"{relative}:{name}"] = actual

    protocol = read_json(root / PROTOCOL_REL)
    if protocol.get("experiment_id") != "CLS_POS_GLOBAL_STRENGTH_E79_V3":
        raise ReplayError("E79 protocol identity changed")
    r3_repair = protocol.get("r3_repair", {})
    if (
        r3_repair.get("r1_and_r2_evidence_preserved") is not True
        or r3_repair.get("matrix_digests_must_be_derived_from_returned_matrices") is not True
        or r3_repair.get("synthetic_execute_payload_json_write_readback_and_independent_verifier_required") is not True
        or r3_repair.get("recursive_symtable_global_check_for_runner_and_verifier_required") is not True
        or r3_repair.get("retry_count_after_r3_repair") != 1
    ):
        raise ReplayError("Protocol omits one or more mandatory R3 receipt/QA controls")
    if protocol.get("current_checkpoint", {}).get("test_or_sample_loaded") is not False:
        raise ReplayError("Protocol does not preserve the no-Test boundary")
    if protocol.get("current_checkpoint", {}).get("official_fit_started") is not False:
        raise ReplayError("Protocol unexpectedly records a model fit")

    body: dict[str, Any] = {
        "schema_version": 1,
        "attestation_id": "CLS_POS_GLOBAL_STRENGTH_E79_SOURCE_COMPATIBILITY_V3",
        "status": "PASS_SOURCE_COMPATIBILITY_ATTESTATION_R3_PENDING_REPLAY",
        "authorization": {"path": AUTH_PATH, "sha256": AUTH_SHA256, "decision": "GO_TARGET_FREE_COMPATIBILITY_REPLAY"},
        "prepared_from_commit": AUTH_COMMIT,
        "producer_provenance": producers,
        "saved_inputs_and_fitted_state_receipts": saved,
        "current_source_sha256": current_sources,
        "numerical_helper_closure_symbol_sha256": symbols,
        "source_variances": EXPLICIT_SOURCE_VARIANCES,
        "correction_note": {
            "path": CORRECTION_NOTE_REL,
            "sha256": current_sources[CORRECTION_NOTE_REL],
            "preserves_r1_failure": True,
            "preserves_r2_failed_receipt": True,
            "fix": "Return the measured matrix metadata from the parity helper, use those actual digests in the success payload, and write/readback/self-verify a one-shot receipt before independent verification.",
        },
        "boundaries": {
            "official_train_columns_only": ["record_id", "cycle_code", "CLASSIFICATION_FEATURES"],
            "target_column_read": False,
            "test_or_sample_read": False,
            "estimator_fit_calls": 0,
            "new_vocabulary_or_idf_fit": False,
            "new_df_map": False,
            "new_gamma": False,
            "inference_or_submission": False,
        },
    }
    body["attestation_sha256"] = canonical_sha256(body)
    path = root / ATTESTATION_REL
    if path.exists():
        raise ReplayError(f"Refusing to replace compatibility attestation: {ATTESTATION_REL}")
    _write_once(path, (json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"))
    return body


def _verify_attestation(root: Path = ROOT) -> dict[str, Any]:
    path = root / ATTESTATION_REL
    if not path.is_file():
        raise ReplayError("Committed source-compatibility attestation is missing")
    attestation = read_json(path)
    body = dict(attestation)
    supplied = body.pop("attestation_sha256", None)
    if supplied != canonical_sha256(body):
        raise ReplayError("Source-compatibility attestation digest is invalid")
    if attestation.get("status") != "PASS_SOURCE_COMPATIBILITY_ATTESTATION_R3_PENDING_REPLAY":
        raise ReplayError("Unexpected source-compatibility attestation status")
    _require_hash_match("authorization", AUTH_SHA256, sha256_file(root / AUTH_PATH))
    for relative, expected in attestation.get("current_source_sha256", {}).items():
        _require_hash_match(f"attested source {relative}", expected, sha256_file(root / relative))
    saved = _verify_saved_inputs(root)
    if saved != attestation.get("saved_inputs_and_fitted_state_receipts"):
        raise ReplayError("Saved-state or immutable feature audit differs from the attestation")
    return attestation


def _verify_committed_sources(root: Path, attestation: dict[str, Any]) -> dict[str, str]:
    head = _git(root, "rev-parse", "HEAD")
    parent = _git(root, "rev-parse", "HEAD^")
    _require_hash_match("source commit parent", AUTH_COMMIT, parent)
    committed: dict[str, str] = {}
    tracked_paths = set(attestation["current_source_sha256"]) | {ATTESTATION_REL}
    for relative in sorted(tracked_paths):
        current = sha256_file(root / relative)
        committed_blob = sha256_bytes(_git_blob(root, head, relative))
        _require_hash_match(f"committed/current source {relative}", current, committed_blob)
        if relative in attestation["current_source_sha256"]:
            _require_hash_match(f"attested/current source {relative}", attestation["current_source_sha256"][relative], current)
        committed[relative] = committed_blob
    return {"head": head, "parent": parent, "committed_path_sha256": committed}


def _resource_snapshot(stage: str) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    process_info = psutil.Process().memory_info()
    peak = getattr(process_info, "peak_wset", None)
    if peak is None:
        peak = getattr(process_info, "peak_rss", None)
    return {
        "stage": stage,
        "available_ram_bytes": int(vm.available),
        "process_rss_bytes": int(process_info.rss),
        "process_peak_rss_bytes": int(peak) if peak is not None else None,
        "incremental_memory_estimate_bytes": INCREMENTAL_ESTIMATE_BYTES,
        "required_free_reserve_bytes": FREE_RESERVE_BYTES,
        "projected_reserve_bytes": int(vm.available) - INCREMENTAL_ESTIMATE_BYTES,
        "gate_pass": int(vm.available) >= MIN_AVAILABLE_BYTES,
        "estimate_is_measured_peak": False,
    }


@contextmanager
def _shared_job_lock() -> Iterator[dict[str, Any]]:
    import tempfile

    lock_path = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    acquired = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise SharedLockHeld("Windows nonblocking shared advisory lock is held") from exc
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise SharedLockHeld("Nonblocking shared advisory lock is held") from exc
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        acquired = True
        yield {"path": str(lock_path), "mechanism": "OS nonblocking advisory lock", "acquired": True}
    finally:
        if acquired:
            try:
                unlock()
            except OSError:
                pass
        stream.close()


@contextmanager
def _cgeom_local_lock(root: Path) -> Iterator[dict[str, Any]]:
    lock_path = root / CGEOM_RUN_ROOT_REL / ".training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner = json.dumps(
        {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "E79 feature-only replay"},
        sort_keys=True,
    ).encode("utf-8")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise LocalLockHeld(f"CGeom local lock is held: {lock_path}") from exc
    try:
        os.write(descriptor, owner)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield {"path": str(lock_path), "mechanism": "exclusive owner-file lock", "acquired": True}
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            if lock_path.read_bytes() == owner:
                lock_path.unlink()
        except FileNotFoundError:
            pass


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ReplayError(f"Refusing to replace an existing one-shot output: {path}")
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    if temporary.exists():
        raise ReplayError(f"Stale partial output exists; refusing overwrite: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ReplayError(f"Immutable output was concurrently created: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_and_readback_receipt(path: Path, body: dict[str, Any]) -> dict[str, Any]:
    """Persist a self-digested one-shot receipt and verify its serialized readback."""
    if "receipt_sha256" in body:
        raise ReplayError("Receipt body must not supply its own digest")
    payload = dict(body)
    payload["receipt_sha256"] = canonical_sha256(body)
    encoded = (json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    _write_once(path, encoded)
    persisted = read_json(path)
    persisted_body = dict(persisted)
    supplied = persisted_body.pop("receipt_sha256", None)
    if supplied != canonical_sha256(persisted_body):
        raise ReplayError("Persisted replay receipt failed its internal digest on readback")
    if canonical_sha256(persisted_body) != canonical_sha256(body):
        raise ReplayError("Persisted replay receipt differs from the returned payload")
    if persisted.get("status") == "PASS_SAVED_STATE_BASE_AND_POSITION_MATRIX_PARITY":
        matrices = persisted.get("matrices", {})
        if (not matrices.get("checks") or not all(matrices["checks"].values())
                or not matrices.get("base", {}).get("logical_sha256")
                or not matrices.get("position", {}).get("logical_sha256")):
            raise ReplayError("Successful replay receipt is missing actual matrix parity evidence")
    return persisted


def _future_df_map_cost() -> dict[str, Any]:
    per_token = VOCABULARY_SIZE * 8
    position_penalty = POSITION_SHAPE[1] * 8
    return {
        "not_computed_in_this_checkpoint": True,
        "retained_unigram_vocabulary_size": VOCABULARY_SIZE,
        "source_position_matrix_shape": list(POSITION_SHAPE),
        "source_position_matrix_nnz": POSITION_NNZ,
        "operation": "count each token once per Train row if nonzero in any of three positional blocks",
        "exact_array_payload_bytes": {
            "document_frequency_int64": per_token,
            "raw_clipped_weights_float64": per_token,
            "aligned_weights_float64": per_token,
            "permuted_weights_float64": per_token,
            "permutation_int64": per_token,
            "repeated_global_penalty_float64": position_penalty,
            "scalar_float64": 8,
            "all_transient_arrays_total": 5 * per_token + position_penalty + 8,
            "persisted_npz_arrays_total_uncompressed": 4 * per_token + position_penalty + 8,
        },
        "array_payload_is_not_peak_memory_or_runtime": True,
        "no_df_or_penalty_array_created": True,
    }


def canonical_feature_columns(feature_columns: Any) -> list[str]:
    """Return ordered unique feature columns and fail before any CSV read on labels."""
    if not isinstance(feature_columns, (list, tuple)):
        raise ReplayError("Feature column declaration must be a list or tuple")
    raw = ["record_id", "cycle_code", *feature_columns]
    if any(not isinstance(name, str) or not name for name in raw):
        raise ReplayError("Feature column declaration contains a non-string or empty name")
    forbidden = {"artifact_signal", "threat_rank"}
    requested_targets = forbidden & set(raw)
    if requested_targets:
        raise ReplayError(f"Target columns are forbidden in a feature-only read: {sorted(requested_targets)}")
    columns = list(dict.fromkeys(raw))
    if len(columns) != len(set(columns)):
        raise ReplayError("Canonical feature request is not unique")
    return columns


def validate_feature_frame_columns(frame: pd.DataFrame, expected_columns: list[str]) -> None:
    """Require the exact unique feature-only columns returned by a reader."""
    actual_columns = list(frame.columns)
    if len(actual_columns) != len(set(actual_columns)):
        raise ReplayError("Feature-only frame contains duplicate column names")
    forbidden = {"artifact_signal", "threat_rank"}
    if forbidden & set(actual_columns):
        raise ReplayError("Feature-only frame unexpectedly contains a target column")
    if set(actual_columns) != set(expected_columns) or len(actual_columns) != len(expected_columns):
        missing = sorted(set(expected_columns) - set(actual_columns))
        extra = sorted(set(actual_columns) - set(expected_columns))
        raise ReplayError(f"Feature-only frame columns differ from request: missing={missing}, extra={extra}")


def read_feature_csv(path: Path, feature_columns: Any) -> tuple[pd.DataFrame, list[str]]:
    """Read an exact, deduplicated feature-only column set from a CSV."""
    columns = canonical_feature_columns(feature_columns)
    try:
        frame = pd.read_csv(path, usecols=columns, dtype={"record_id": str}, low_memory=False)
    except ValueError as exc:
        raise ReplayError(f"Feature-only CSV is missing or cannot supply a requested column: {exc}") from exc
    validate_feature_frame_columns(frame, columns)
    return frame, columns


def assemble_and_validate_matrices(
    structured: sparse.spmatrix,
    text: sparse.spmatrix,
    missing: Any,
    position: sparse.spmatrix,
    record_ids: Any,
    expected: Mapping[str, Any],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, str, dict[str, bool], dict[str, dict[str, Any]]]:
    """High-level C2_B base assembly and exact matrix/order parity gate."""
    structured_csr = sparse.csr_matrix(structured, dtype=np.float64, copy=False)
    text_csr = sparse.csr_matrix(text, dtype=np.float64, copy=False)
    position_csr = sparse.csr_matrix(position, dtype=np.float64, copy=False)
    missing_mask = np.asarray(missing, dtype=bool).reshape(-1)
    ids = np.asarray(record_ids).astype(str).reshape(-1)
    row_counts = {structured_csr.shape[0], text_csr.shape[0], position_csr.shape[0], len(missing_mask), len(ids)}
    if len(row_counts) != 1:
        raise ReplayError("Synthetic/replayed matrices and metadata have different row counts")
    base = sparse.hstack(
        (structured_csr, text_csr, sparse.csr_matrix(missing_mask.astype(np.float64).reshape(-1, 1))),
        format="csr", dtype=np.float64,
    )
    ids_sha = positional.sha256_array(ids)
    base_sha = positional.sha256_csr(base)
    position_sha = positional.sha256_csr(position_csr)
    checks = {
        "base_shape": list(base.shape) == expected["base_shape"],
        "base_nnz": int(base.nnz) == expected["base_nnz"],
        "base_logical_sha256": base_sha == expected["base_logical_sha256"],
        "position_shape": list(position_csr.shape) == expected["position_shape"],
        "position_nnz": int(position_csr.nnz) == expected["position_nnz"],
        "position_logical_sha256": position_sha == expected["position_logical_sha256"],
        "ordered_train_ids_sha256": ids_sha == expected["ordered_train_ids_sha256"],
        "finite_base": bool(np.isfinite(base.data).all()),
        "finite_position": bool(np.isfinite(position_csr.data).all()),
    }
    if not all(checks.values()):
        raise ReplayError(f"Replayed matrix failed expected shape/nnz/hash/order parity: {checks}")
    actual = {
        "base": {"shape": list(base.shape), "nnz": int(base.nnz), "logical_sha256": base_sha},
        "position": {"shape": list(position_csr.shape), "nnz": int(position_csr.nnz), "logical_sha256": position_sha},
    }
    return base, position_csr, ids_sha, checks, actual


def _execute_feature_only_replay(root: Path, attestation: dict[str, Any], commit_info: dict[str, Any],
                                 locks: list[dict[str, Any]], resource_before: dict[str, Any]) -> dict[str, Any]:
    from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom

    started_at_utc = datetime.now(timezone.utc).isoformat()
    train_path = root / TRAIN_REL
    train, train_columns = read_feature_csv(train_path, cgeom.CLASSIFICATION_FEATURES)
    if len(train) != EXPECTED_ROWS or train.record_id.isna().any() or train.record_id.duplicated().any():
        raise ReplayError("Official Train row or record_id support differs from the immutable population")
    cycles = pd.to_numeric(train.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if tuple(sorted(np.unique(cycles).tolist())) != EXPECTED_CYCLES:
        raise ReplayError("Official Train cycle population differs from cycles 47-79")
    compatibility = cgeom.audit_c2b_vectorizer_compatibility(
        frame=train.loc[:, ["record_id", "cycle_code", "signal_stream"]].copy(),
        model=joblib.load(root / A02_MODEL_REL),
    )
    expected_compatibility = read_json(root / CGEOM_PREFLIGHT_REL)["a02_vectorizer_compatibility"]
    for name in (
        "status", "train_sha256", "train_rows", "vocabulary_sha256", "idf_sha256",
        "fit_documents_sha256", "fit_row_positions_sha256", "target_column_requested_or_loaded", "uses_test_or_sample",
    ):
        _require_hash_match(f"A02 feature compatibility {name}", str(expected_compatibility.get(name)), str(compatibility.get(name)))
    if compatibility.get("target_column_requested_or_loaded") is not False or compatibility.get("uses_test_or_sample") is not False:
        raise ReplayError("A02 compatibility audit crossed the target or Test boundary")

    model = joblib.load(root / A02_MODEL_REL)
    vectorizer = model.text_space.vectorizer
    unit_objects: dict[str, Any] = {}
    unit_state_hashes: dict[str, str] = {}
    for key in ("c2b_structured", "c2b_scaler", "c2b_ohe"):
        record = attestation["saved_inputs_and_fitted_state_receipts"]["saved_units"][key]
        state = joblib.load(root / record["artifact_path"])
        summary = cgeom._state_summary(state)
        state_sha = canonical_sha256(summary)
        _require_hash_match(f"loaded saved-state summary {key}", record["state_sha256"], state_sha)
        unit_objects[key] = state
        unit_state_hashes[key] = state_sha

    features = train.loc[:, list(cgeom.CLASSIFICATION_FEATURES)].copy()
    structured = positional.transform_c2b_structured(
        unit_objects["c2b_structured"], unit_objects["c2b_scaler"], unit_objects["c2b_ohe"], features,
    )
    text, missing = positional.transform_c2b_text(vectorizer, features)
    documents, _missing_check = classification_models._text_values(features["signal_stream"])
    position, position_build_audit = positional.build_position_block(documents, vectorizer, shuffled=False)
    expected_matrix = {
        "base_shape": list(BASE_SHAPE),
        "base_nnz": BASE_NNZ,
        "base_logical_sha256": BASE_SHA256,
        "position_shape": list(POSITION_SHAPE),
        "position_nnz": POSITION_NNZ,
        "position_logical_sha256": POSITION_SHA256,
        "ordered_train_ids_sha256": ORDERED_TRAIN_IDS_SHA256,
    }
    base, position, ids_sha, checks, actual_matrices = assemble_and_validate_matrices(
        structured, text, missing, position, train.record_id.astype(str).to_numpy(dtype=str), expected_matrix,
    )
    _require_hash_match("ordered official Train IDs", ORDERED_TRAIN_IDS_SHA256, ids_sha)

    resource_after = _resource_snapshot("after feature-only replay; matrices still resident")
    if resource_after["available_ram_bytes"] < FREE_RESERVE_BYTES:
        raise ReplayError("Actual available RAM fell below the required 0.75 GiB reserve during replay")
    return {
        "schema_version": 1,
        "replay_id": "CLS_POS_GLOBAL_STRENGTH_E79_TARGET_FREE_FEATURE_REPLAY_V3",
        "status": "PASS_SAVED_STATE_BASE_AND_POSITION_MATRIX_PARITY",
        "started_at_utc": started_at_utc,
        "source_commit": commit_info["head"],
        "source_commit_parent": commit_info["parent"],
        "source_compatibility_attestation_sha256": attestation["attestation_sha256"],
        "source_sha256": attestation["current_source_sha256"],
        "input_sha256": attestation["saved_inputs_and_fitted_state_receipts"]["immutable_file_sha256"],
        "locks": locks,
        "resource_before": resource_before,
        "resource_after": resource_after,
        "train_path": TRAIN_REL,
        "train_sha256": TRAIN_SHA256,
        "train_rows": len(train),
        "train_columns_requested": train_columns,
        "target_column_requested_or_loaded": False,
        "test_or_sample_loaded": False,
        "estimator_fit_calls": 0,
        "prediction_or_inference_calls": 0,
        "new_df_map_created": False,
        "new_gamma_created": False,
        "csv_created": False,
        "a02_compatibility": compatibility,
        "saved_state_summary_sha256": unit_state_hashes,
        "ordered_train_ids_sha256": ids_sha,
        "matrices": {
            "base": actual_matrices["base"],
            "position": {**actual_matrices["position"],
                         "columns": POSITION_COLUMNS, "build_audit": position_build_audit},
            "checks": checks,
        },
        "future_df_map_cost": _future_df_map_cost(),
    }


def run_feature_replay(root: Path = ROOT) -> dict[str, Any]:
    attestation = _verify_attestation(root)
    commit_info = _verify_committed_sources(root, attestation)
    run_root = root / RUN_ROOT_REL
    attempt_path = root / ATTEMPT_REL
    receipt_path = root / REPLAY_RECEIPT_REL
    if attempt_path.exists() or receipt_path.exists():
        raise ReplayError("A prior replay attempt marker/receipt exists; no retry is permitted")
    lock_audit: list[dict[str, Any]] = []
    with ExitStack() as stack:
        shared = stack.enter_context(_shared_job_lock())
        lock_audit.append(shared)
        local = stack.enter_context(_cgeom_local_lock(root))
        lock_audit.append(local)
        resource_before = _resource_snapshot("immediately before one feature-only replay")
        base = {
            "schema_version": 1,
            "replay_id": "CLS_POS_GLOBAL_STRENGTH_E79_TARGET_FREE_FEATURE_REPLAY_V3",
            "source_commit": commit_info["head"],
            "source_commit_parent": commit_info["parent"],
            "source_compatibility_attestation_sha256": attestation["attestation_sha256"],
            "source_sha256": attestation["current_source_sha256"],
            "input_sha256": attestation["saved_inputs_and_fitted_state_receipts"]["immutable_file_sha256"],
            "locks": lock_audit,
            "resource_before": resource_before,
            "target_column_requested_or_loaded": False,
            "test_or_sample_loaded": False,
            "estimator_fit_calls": 0,
            "prediction_or_inference_calls": 0,
            "new_df_map_created": False,
            "new_gamma_created": False,
            "csv_created": False,
        }
        if not resource_before["gate_pass"]:
            body = {
                **base,
                "status": "BLOCKED_RESOURCE_GATE",
                "note": "Feature-only replay not started; actual available RAM did not leave the required projected reserve.",
            }
            return write_and_readback_receipt(receipt_path, body)

        marker = {
            "schema_version": 1,
            "status": "ONE_REPLAY_ATTEMPT_STARTED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": commit_info["head"],
            "attestation_sha256": attestation["attestation_sha256"],
            "locks": lock_audit,
            "resource_before": resource_before,
            "no_retry_if_interrupted": True,
        }
        _write_once(attempt_path, (json.dumps(marker, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"))
        started = time.perf_counter()
        try:
            with threadpool_limits(limits=MAX_THREADS):
                body = _execute_feature_only_replay(root, attestation, commit_info, lock_audit, resource_before)
        except Exception as exc:
            body = {
                **base,
                "status": "FAILED_FEATURE_ONLY_REPLAY_NO_RETRY",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": float(time.perf_counter() - started),
                "resource_after_failure": _resource_snapshot("after failed feature-only replay"),
            }
        body["elapsed_seconds"] = float(time.perf_counter() - started)
        return write_and_readback_receipt(receipt_path, body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-attestation", action="store_true")
    mode.add_argument("--verify-attestation", action="store_true")
    mode.add_argument("--feature-replay", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.build_attestation:
            result = build_source_compatibility_attestation()
        elif args.verify_attestation:
            result = _verify_attestation()
        else:
            result = run_feature_replay()
    except SharedLockHeld as exc:
        print(json.dumps({"status": "BLOCKED_SHARED_ADVISORY_LOCK_HELD", "error": str(exc)}, indent=2))
        return 2
    except LocalLockHeld as exc:
        print(json.dumps({"status": "BLOCKED_CGEOM_LOCAL_LOCK_HELD", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    return 0 if str(result.get("status", "")).startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
