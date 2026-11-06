from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/simplification_v1/CLS_POST_BAG2_NEXT_V1"
FREEZE_PATH = REPORT_ROOT / "FREEZE.json"
PRELABEL_PATH = REPORT_ROOT / "PRELABEL_VERIFICATION.json"
SCORE_SCRIPT = ROOT / "scripts/score_cls_a02_position_kernel_v1.py"
MANIFEST_PATH = REPORT_ROOT / "SCORE_RUNTIME_ADAPTER_01.json"
SCORE_ATTEMPT_PATH = REPORT_ROOT / "SCORING_ATTEMPT.json"
RESULTS_PATH = REPORT_ROOT / "RESULTS.json"
RESULTS_MD_PATH = REPORT_ROOT / "RESULTS.md"
AMENDMENT_ID = "CLS_A02_POSITION_KERNEL_V1_SCORE_RUNTIME_ADAPTER_01"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def committed_clean(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                             text=True, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT,
                           text=True, capture_output=True).returncode == 0
    return tracked and clean


def frozen_source_hash(freeze: dict[str, Any], path: Path) -> str | None:
    target = path.relative_to(ROOT).as_posix()
    return next((value for name, value in freeze.get("source_hashes", {}).items()
                 if name.replace("\\", "/") == target), None)


def _score_artifacts_absent() -> bool:
    return not any(path.exists() for path in (SCORE_ATTEMPT_PATH, RESULTS_PATH, RESULTS_MD_PATH))


def _validate_inputs(*, require_adapter_manifest: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (FREEZE_PATH, PRELABEL_PATH, SCORE_SCRIPT):
        if not path.is_file() or not committed_clean(path):
            raise RuntimeError(f"Required committed file is missing or modified: {path}")
    freeze = load_json(FREEZE_PATH)
    prelabel = load_json(PRELABEL_PATH)
    if (prelabel.get("status") != "PASS_AMENDED_PRELABEL_VERIFICATION"
            or prelabel.get("freeze_sha256") != sha_file(FREEZE_PATH)
            or prelabel.get("amendment_id") != "CLS_A02_POSITION_KERNEL_V1_PRELABEL_AMENDMENT_01"
            or prelabel.get("original_prelabel_boundary_compliant") is not False
            or prelabel.get("scientific_qualification") is not False
            or prelabel.get("model_construction_training_boundary_verified") is not True):
        raise RuntimeError("Committed amended prelabel verification is not an acceptable score boundary")
    if frozen_source_hash(freeze, SCORE_SCRIPT) != sha_file(SCORE_SCRIPT):
        raise RuntimeError("Frozen scorer source hash changed; adapter only supports the exact frozen scorer")
    if freeze.get("original_prelabel_boundary_compliant") is not False or freeze.get("scientific_qualification") is not False:
        raise RuntimeError("Historical protocol-deviation flags changed")
    if not _score_artifacts_absent():
        raise RuntimeError("A score attempt or result exists; the one-score rule forbids another run")
    if require_adapter_manifest and (not MANIFEST_PATH.is_file() or not committed_clean(MANIFEST_PATH)):
        raise RuntimeError("Runtime compatibility manifest must be committed and unchanged")
    return freeze, prelabel


def _source_has_missing_subprocess_import() -> bool:
    import ast

    tree = ast.parse(SCORE_SCRIPT.read_text(encoding="utf-8"))
    module_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_imports.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_imports.update(alias.asname or alias.name for alias in node.names)
    return "subprocess" in unresolved_standard_module_names() and "subprocess" not in module_imports


def unresolved_standard_module_names() -> set[str]:
    import ast

    known_stdlib_modules = {
        "argparse", "collections", "csv", "datetime", "functools", "hashlib", "importlib",
        "itertools", "json", "math", "os", "pathlib", "re", "statistics", "subprocess",
        "sys", "tempfile", "threading", "time", "typing", "warnings",
    }
    tree = ast.parse(SCORE_SCRIPT.read_text(encoding="utf-8"))

    def imports(nodes) -> set[str]:
        found = set()
        for node in nodes:
            if isinstance(node, ast.Import):
                found.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    found.add(node.module.split(".")[0])
                found.update(alias.asname or alias.name for alias in node.names)
        return found

    global_bound = imports(tree.body)
    unresolved = set()
    for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        nodes = []
        stack = list(function.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            nodes.append(node)
            stack.extend(ast.iter_child_nodes(node))
        local_bound = global_bound | imports(nodes)
        used = {node.id for node in nodes if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load) and node.id in known_stdlib_modules}
        unresolved.update(used - local_bound)
    return unresolved


def inject_subprocess_compatibility(scorer: Any) -> None:
    if hasattr(scorer, "subprocess"):
        raise RuntimeError("The frozen scorer already has a subprocess global; refusing adapter injection")
    scorer.subprocess = subprocess


def prepare_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        raise RuntimeError("Refusing to overwrite the immutable scorer adapter manifest")
    if git("status", "--porcelain"):
        raise RuntimeError("Commit the adapter code with a clean tree before creating its manifest")
    freeze, prelabel = _validate_inputs(require_adapter_manifest=False)
    if not committed_clean(Path(__file__)):
        raise RuntimeError("Adapter code must be committed and unchanged before manifest creation")
    if not _source_has_missing_subprocess_import():
        raise RuntimeError("The frozen scorer no longer has the verified missing-subprocess compatibility issue")
    if unresolved_standard_module_names() != {"subprocess"}:
        raise RuntimeError(f"Unexpected unresolved standard-library module globals: {sorted(unresolved_standard_module_names())}")
    if prelabel.get("old_verifier_invoked") is not False or prelabel.get("query_labels_accessed_by_this_verifier") is not False:
        raise RuntimeError("The amended target-free verification boundary has changed")

    manifest = {
        "amendment_id": AMENDMENT_ID,
        "status": "APPROVED_SCORER_RUNTIME_COMPATIBILITY_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": freeze["source_commit"],
        "implementation_commit": git("rev-parse", "HEAD"),
        "freeze_sha256": sha_file(FREEZE_PATH),
        "prelabel_verification_sha256": sha_file(PRELABEL_PATH),
        "prelabel_status": prelabel["status"],
        "frozen_score_script_path": SCORE_SCRIPT.relative_to(ROOT).as_posix(),
        "frozen_score_script_sha256": sha_file(SCORE_SCRIPT),
        "adapter_script_path": Path(__file__).relative_to(ROOT).as_posix(),
        "adapter_script_sha256": sha_file(Path(__file__)),
        "compatibility_change": "Inject only the Python standard-library subprocess module as a missing global on the already imported frozen scorer module, then call its unchanged score_once() function.",
        "frozen_scorer_modified": False,
        "scoring_logic_or_metric_changed": False,
        "previous_runtime_failure": {
            "exception": "NameError: name 'subprocess' is not defined",
            "failure_phase": "before SCORING_ATTEMPT creation and before artifact_signal read",
            "score_attempt_file_created": False,
            "target_labels_accessed": False,
            "test_or_sample_accessed": False,
            "kaggle_accessed": False,
        },
        "one_score_run_allowed_after_this_manifest": True,
        "scientific_qualification": False,
        "original_prelabel_boundary_compliant": False,
    }
    with MANIFEST_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
                      "adapter_sha256": manifest["adapter_script_sha256"],
                      "frozen_scorer_sha256": manifest["frozen_score_script_sha256"],
                      "prelabel_verification_sha256": manifest["prelabel_verification_sha256"]}, sort_keys=True))
    return manifest


def run_once() -> dict[str, Any]:
    freeze, prelabel = _validate_inputs(require_adapter_manifest=True)
    manifest = load_json(MANIFEST_PATH)
    adapter_sha = sha_file(Path(__file__))
    if (manifest.get("amendment_id") != AMENDMENT_ID
            or manifest.get("status") != "APPROVED_SCORER_RUNTIME_COMPATIBILITY_ONLY"
            or manifest.get("source_commit") != freeze.get("source_commit")
            or manifest.get("freeze_sha256") != sha_file(FREEZE_PATH)
            or manifest.get("prelabel_verification_sha256") != sha_file(PRELABEL_PATH)
            or manifest.get("frozen_score_script_sha256") != sha_file(SCORE_SCRIPT)
            or manifest.get("adapter_script_sha256") != adapter_sha
            or manifest.get("frozen_scorer_modified") is not False
            or manifest.get("scoring_logic_or_metric_changed") is not False
            or manifest.get("one_score_run_allowed_after_this_manifest") is not True):
        raise RuntimeError("Committed score adapter manifest does not bind the verified frozen scorer")
    if not _source_has_missing_subprocess_import():
        raise RuntimeError("Frozen scorer compatibility issue differs from the manifest")
    if unresolved_standard_module_names() != {"subprocess"}:
        raise RuntimeError(f"Unexpected unresolved standard-library module globals: {sorted(unresolved_standard_module_names())}")

    scorer = importlib.import_module("scripts.score_cls_a02_position_kernel_v1")
    if scorer.__file__ is None or Path(scorer.__file__).resolve() != SCORE_SCRIPT.resolve():
        raise RuntimeError("Imported scorer module is not the frozen score script")
    inject_subprocess_compatibility(scorer)
    result = scorer.score_once()
    summary = {
        "status": result.get("status"),
        "amendment_id": AMENDMENT_ID,
        "adapter_manifest_sha256": sha_file(MANIFEST_PATH),
        "adapter_script_sha256": adapter_sha,
        "frozen_score_script_sha256": sha_file(SCORE_SCRIPT),
        "prelabel_verification_sha256": sha_file(PRELABEL_PATH),
        "results_sha256": sha_file(RESULTS_PATH),
        "scoring_attempt_sha256": sha_file(SCORE_ATTEMPT_PATH),
        "scientific_qualification": result.get("scientific_qualification"),
        "original_prelabel_boundary_compliant": result.get("original_prelabel_boundary_compliant"),
    }
    print(json.dumps(summary, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Use the frozen A02 scorer with its one missing stdlib module global supplied.")
    parser.add_argument("--prepare-manifest", action="store_true", help="Create the immutable pre-score runtime compatibility manifest.")
    args = parser.parse_args()
    if args.prepare_manifest:
        prepare_manifest()
    else:
        run_once()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
