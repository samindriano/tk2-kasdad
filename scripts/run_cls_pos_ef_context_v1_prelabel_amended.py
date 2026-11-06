"""Run the frozen independent prelabel verifier with its two missing hash aliases bound."""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib
import importlib.util
from pathlib import Path
import subprocess
import symtable
import sys
from typing import Any

AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1"
AMENDMENT_PATH = REPORT_ROOT / "PRELABEL_COMPAT_AMENDMENT.json"
WRAPPER_REL = "scripts/run_cls_pos_ef_context_v1_prelabel_amended.py"
FROZEN_VERIFIER_REL = "scripts/verify_cls_pos_ef_context_v1.py"
ALLOWED_MISSING_HELPERS = {"sha256_array", "sha256_csr"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_normalized_text(path: Path) -> str:
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args]).decode("utf-8").strip()


def unresolved_global_names(path: Path) -> set[str]:
    tree = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
    module_names = {symbol.get_name() for symbol in tree.get_symbols()
                    if symbol.is_assigned() or symbol.is_imported()}
    walk = lambda table: [table] + [child for nested in table.get_children() for child in walk(nested)]
    missing = set()
    for table in walk(tree):
        for symbol in table.get_symbols():
            name = symbol.get_name()
            if (symbol.is_global() and name not in module_names and name not in vars(builtins)
                    and name not in {"__name__", "__file__", "__package__", "__spec__"}):
                missing.add(name)
    return missing


def load_runner():
    if str(AUDIT_ROOT) in sys.path:
        sys.path.remove(str(AUDIT_ROOT))
    sys.path.insert(0, str(AUDIT_ROOT))
    path = AUDIT_ROOT / "scripts/run_cls_pos_ef_context_v1_amended.py"
    spec = importlib.util.spec_from_file_location("cls_pos_ef_context_v1_fit_amendment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load committed frozen-fit adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verified_verifier_module(runner, pilot, freeze: dict[str, Any]):
    verifier_path = AUDIT_ROOT / FROZEN_VERIFIER_REL
    expected = freeze["source_hashes"].get(FROZEN_VERIFIER_REL)
    pilot.require(expected is not None and sha256_file(verifier_path) == expected,
                  "Frozen independent verifier source hash differs")
    unresolved = unresolved_global_names(verifier_path)
    pilot.require(unresolved == ALLOWED_MISSING_HELPERS,
                  f"Frozen verifier unresolved-global inventory changed: {sorted(unresolved)}")

    verifier = importlib.import_module("scripts.verify_cls_pos_ef_context_v1")
    pilot.require(Path(verifier.__file__).resolve() == verifier_path.resolve(),
                  "Independent verifier was imported from a different repository")
    pilot.require(not any(hasattr(verifier, name) for name in ALLOWED_MISSING_HELPERS),
                  "Frozen verifier unexpectedly defines a helper alias")
    verifier.sha256_array = pilot.sha256_array
    verifier.sha256_csr = pilot.sha256_csr
    pilot.require(verifier.sha256_array is pilot.sha256_array and verifier.sha256_csr is pilot.sha256_csr,
                  "Verifier helper binding does not match the frozen producer helpers")
    return verifier, unresolved


def verify_committed_amendment(runner, pilot, freeze: dict[str, Any]) -> tuple[dict[str, Any], str]:
    amendment = pilot.read_json(AMENDMENT_PATH)
    pilot.require(amendment.get("status") == "VERIFIER_COMPAT_ONLY"
                  and amendment.get("candidate_id") == freeze["candidate_id"]
                  and amendment.get("parent_freeze_signature") == freeze["freeze_signature"]
                  and amendment.get("parent_source_commit") == freeze["source_commit"]
                  and amendment.get("frozen_verifier_sha256") == freeze["source_hashes"][FROZEN_VERIFIER_REL]
                  and amendment.get("bound_helper_names") == sorted(ALLOWED_MISSING_HELPERS),
                  "Prelabel compatibility amendment does not bind to the frozen source")
    wrapper_hash = sha256_file(AUDIT_ROOT / WRAPPER_REL)
    pilot.require(wrapper_hash == amendment.get("wrapper_sha256"), "Prelabel compatibility wrapper hash mismatch")
    head = git(AUDIT_ROOT, "rev-parse", "HEAD")
    wrapper_blob = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"HEAD:{WRAPPER_REL}"])
    pilot.require(sha256_bytes(wrapper_blob) == wrapper_hash,
                  "Prelabel compatibility wrapper must be committed before verifier execution")
    manifest_rel = AMENDMENT_PATH.relative_to(AUDIT_ROOT).as_posix()
    manifest_blob = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"HEAD:{manifest_rel}"])
    pilot.require(sha256_bytes(manifest_blob) == sha256_normalized_text(AMENDMENT_PATH),
                  "Prelabel compatibility manifest must be committed before verifier execution")
    dirty = git(AUDIT_ROOT, "status", "--porcelain", WRAPPER_REL, manifest_rel)
    pilot.require(not dirty, "Prelabel compatibility wrapper/manifest has uncommitted changes")
    return amendment, head


def run(official_root: Path) -> dict[str, Any]:
    runner = load_runner()
    pilot = runner.load_frozen_driver()
    root = official_root.resolve()
    freeze = pilot.verify_freeze(root)
    runner.load_audit_core(pilot, freeze)
    amendment, head = verify_committed_amendment(runner, pilot, freeze)
    verifier, unresolved = verified_verifier_module(runner, pilot, freeze)

    result = verifier.verify_prelabel_outputs(root, write=False)
    pilot.require(result.get("status") == "PASS_TARGET_FREE_PRELABEL"
                  and result.get("query_labels_read") is False
                  and result.get("test_or_sample_access") is False,
                  "Independent prelabel verifier did not pass target-free boundaries")
    result.pop("verification_signature", None)
    result["verification_compatibility"] = {
        "amendment_id": amendment["amendment_id"],
        "execution_commit": head,
        "frozen_verifier_sha256": freeze["source_hashes"][FROZEN_VERIFIER_REL],
        "bound_helper_names": sorted(unresolved),
        "helper_source": "hash-pinned frozen producer module functions",
        "verifier_source_modified": False,
    }
    result["verification_signature"] = pilot.canonical_sha256(result)
    pilot.write_once(pilot.PRELABEL_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True)
    args = parser.parse_args()
    result = run(Path(args.official_root))
    print(__import__("json").dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
