"""Fail-closed Kaggle gate for approved regression research candidates.

Audit mode is local and read-only. ``--execute`` is deliberately separate and
requires an explicitly approved manifest. Each candidate may be uploaded at
most once by this tool; a reserved/ambiguous candidate is reconciled against
Kaggle history and is never retried automatically.

Example (audit only)::

    python scripts/submit_regression_research.py --csv submissions/regression_research/<candidate>.csv \
        --manifest reports/regression_research/submissions/<candidate>.json

Example (only after the main agent approves that exact manifest)::

    python scripts/submit_regression_research.py --csv submissions/regression_research/<candidate>.csv \
        --manifest reports/regression_research/submissions/<candidate>.json --execute

Kaggle submissions and scores are stored only in the separate research ledger;
CLI output is sanitized and never forwards raw CLI output or credentials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import task_files  # noqa: E402


COMPETITION = "tp2-regression-cyber-vulnerability-kasdd-gasal-2026-2027"
MAX_ADDITIONAL_SUBMISSIONS = 6
INITIAL_SUBMISSION_REF = "57808231"
INITIAL_SUBMISSION_COUNTS = False
EXPECTED_TRAIN_ROWS = 50_608
EXPECTED_TEST_ROWS = 29_421
CSV_HEADER = ("record_id", "threat_rank")
DEFAULT_KAGGLE_CLI = Path(
    os.environ.get(
        "KAGGLE_CLI_PATH",
        r"C:\Users\Sam\AppData\Local\Temp\tp2-kaggle-cli-venv\Scripts\kaggle.exe",
    )
)
HEX_40 = re.compile(r"^[0-9a-fA-F]{40}$")
HEX_64 = re.compile(r"^[0-9a-fA-F]{64}$")
CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
ZERO_HASH = "0" * 64


class CandidateError(ValueError):
    """Expected fail-closed candidate validation failure."""


class KaggleCLIError(RuntimeError):
    """Kaggle request failed; raw CLI output is intentionally not retained."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(ids: Iterable[str]) -> str:
    """Hash ordered IDs as UTF-8, one ID per line, with a final LF."""
    return _sha256_bytes(("\n".join(ids) + "\n").encode("utf-8"))


def _resolve_owned_file(path: str | Path, root: Path, allowed_dir: str, suffix: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        root_resolved = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        allowed = (root / allowed_dir).resolve(strict=True)
        allowed.relative_to(root_resolved)
        resolved.relative_to(allowed)
    except (OSError, ValueError) as exc:
        raise CandidateError(f"Path must resolve inside {allowed_dir}.") from exc
    if not resolved.is_file() or resolved.suffix.lower() != suffix:
        raise CandidateError(f"Expected a {suffix} file inside {allowed_dir}.")
    return resolved


def _resolve_official_files(project_root: Path) -> dict[str, Path]:
    configured = os.environ.get("TP2_DATA_DIR")
    data_root = Path(configured).expanduser().resolve() if configured else (project_root / "data" / "raw").resolve()
    files = task_files("regression", root=data_root)
    missing = [role for role, path in files.items() if path is None]
    if missing:
        raise CandidateError("Official regression Train/Test/SampleSubmission files are unavailable.")
    return {role: Path(path).resolve(strict=True) for role, path in files.items()}


def _read_csv_ids(path: Path, expected_header: tuple[str, ...] | None = None) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header or len(header) != len(set(header)):
                raise CandidateError("CSV header is empty or contains duplicate columns.")
            if expected_header is not None and tuple(header) != expected_header:
                raise CandidateError("Submission CSV header/order must be exactly record_id, threat_rank.")
            if "record_id" not in header:
                raise CandidateError("CSV must contain record_id.")
            id_index = header.index("record_id")
            ids: list[str] = []
            for row in reader:
                if len(row) != len(header):
                    raise CandidateError("CSV has a malformed row width.")
                value = row[id_index]
                if not value:
                    raise CandidateError("CSV contains a missing record_id.")
                ids.append(value)
            return ids
    except UnicodeDecodeError as exc:
        raise CandidateError("CSV is not valid UTF-8.") from exc
    except OSError as exc:
        raise CandidateError("CSV could not be read.") from exc


def _read_train_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header or "threat_rank" not in header:
                raise CandidateError("Official Train does not contain the regression target.")
            return sum(1 for _ in reader)
    except (OSError, UnicodeDecodeError) as exc:
        raise CandidateError("Official Train could not be read.") from exc


def _validate_submission_values(path: Path) -> tuple[list[str], int]:
    ids: list[str] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if tuple(header or ()) != CSV_HEADER:
                raise CandidateError("Submission CSV header/order must be exactly record_id, threat_rank.")
            for row in reader:
                if len(row) != 2:
                    raise CandidateError("Submission CSV has a malformed row width.")
                record_id, raw_value = row
                if not record_id or record_id in seen:
                    raise CandidateError("Submission IDs contain a missing value or duplicate.")
                seen.add(record_id)
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise CandidateError("Submission threat_rank values must be numeric.") from exc
                if not math.isfinite(value) or value < 0.0 or value > 100.0:
                    raise CandidateError("Submission threat_rank values must be finite and in [0, 100].")
                ids.append(record_id)
    except UnicodeDecodeError as exc:
        raise CandidateError("Submission CSV is not valid UTF-8.") from exc
    except OSError as exc:
        raise CandidateError("Submission CSV could not be read.") from exc
    return ids, len(ids)


def audit_candidate(
    csv_path: str | Path,
    manifest_path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate an approved candidate and its official-input identity locally."""
    root = Path(project_root).resolve()
    csv_file = _resolve_owned_file(csv_path, root, "submissions/regression_research", ".csv")
    manifest_file = _resolve_owned_file(
        manifest_path, root, "reports/regression_research/submissions", ".json"
    )
    if csv_file.stem != manifest_file.stem:
        raise CandidateError("CSV and manifest basenames must match.")

    try:
        with manifest_file.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError("Manifest is unreadable or invalid JSON.") from exc
    if not isinstance(manifest, dict):
        raise CandidateError("Manifest must be a JSON object.")

    candidate_id = manifest.get("candidate_id")
    if not isinstance(candidate_id, str) or not CANDIDATE_ID.fullmatch(candidate_id):
        raise CandidateError("Manifest candidate_id is missing or unsafe.")
    if csv_file.stem != candidate_id:
        raise CandidateError("Candidate ID must match the CSV and manifest basename.")
    if manifest.get("status") != "KAGGLE_CANDIDATE_APPROVED":
        raise CandidateError("Manifest is not marked KAGGLE_CANDIDATE_APPROVED.")
    if manifest.get("task") != "regression" or manifest.get("competition") != COMPETITION:
        raise CandidateError("Manifest task or competition does not match the regression target.")
    approval = manifest.get("main_approval")
    if not isinstance(approval, dict) or approval.get("approved") is not True:
        raise CandidateError("Main approval for this exact candidate is missing.")
    if not isinstance(manifest.get("source_commit"), str) or not HEX_40.fullmatch(manifest["source_commit"]):
        raise CandidateError("Manifest source_commit must be a full 40-character Git commit hash.")
    if not isinstance(manifest.get("protocol_sha256"), str) or not HEX_64.fullmatch(manifest["protocol_sha256"]):
        raise CandidateError("Manifest protocol_sha256 must be a 64-character SHA-256 hash.")
    recipe = manifest.get("recipe")
    if not ((isinstance(recipe, dict) and recipe) or (isinstance(recipe, str) and recipe.strip())):
        raise CandidateError("Manifest recipe must be a nonempty object or string.")
    if type(manifest.get("fit_rows")) is not int or manifest["fit_rows"] != EXPECTED_TRAIN_ROWS:
        raise CandidateError("Manifest fit_rows must equal the official full Train row count.")
    if type(manifest.get("test_rows")) is not int or manifest["test_rows"] != EXPECTED_TEST_ROWS:
        raise CandidateError("Manifest test_rows must equal the official Test row count.")
    if manifest.get("test_labels_used") is not False:
        raise CandidateError("Manifest must attest that Test labels were not used.")

    input_hashes = manifest.get("input_sha256")
    if not isinstance(input_hashes, dict) or any(
        not isinstance(input_hashes.get(key), str) or not HEX_64.fullmatch(input_hashes[key])
        for key in ("train", "test", "sample")
    ):
        raise CandidateError("Manifest input_sha256 must contain Train, Test, and SampleSubmission SHA-256 values.")
    csv_hash = manifest.get("csv_sha256")
    if not isinstance(csv_hash, str) or not HEX_64.fullmatch(csv_hash):
        raise CandidateError("Manifest csv_sha256 must be a 64-character SHA-256 hash.")
    expected_ids_hash = manifest.get("test_ids_sha256")
    if not isinstance(expected_ids_hash, str) or not HEX_64.fullmatch(expected_ids_hash):
        raise CandidateError("Manifest test_ids_sha256 is required and must be a 64-character SHA-256 hash.")

    files = _resolve_official_files(root)
    freeze_file = root / "experiments" / "round1" / "freeze.json"
    try:
        with freeze_file.open("r", encoding="utf-8") as handle:
            freeze = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError("Frozen regression input receipt is unavailable or invalid.") from exc
    frozen_hashes = freeze.get("identity", {}).get("input_sha256", {}) if isinstance(freeze, dict) else {}
    if not isinstance(frozen_hashes, dict) or any(not HEX_64.fullmatch(str(frozen_hashes.get(key, ""))) for key in ("train", "test", "sample")):
        raise CandidateError("Frozen regression input hashes are incomplete.")

    actual_input_hashes = {role: sha256_file(path) for role, path in files.items()}
    for role in ("train", "test", "sample"):
        if actual_input_hashes[role] != frozen_hashes[role] or input_hashes[role] != frozen_hashes[role]:
            raise CandidateError(f"Official {role} input hash does not match the frozen receipt and manifest.")
    if _read_train_rows(files["train"]) != EXPECTED_TRAIN_ROWS:
        raise CandidateError("Official Train row count differs from the frozen submission protocol.")

    test_ids = _read_csv_ids(files["test"])
    sample_ids = _read_csv_ids(files["sample"], expected_header=CSV_HEADER)
    if len(test_ids) != EXPECTED_TEST_ROWS or len(sample_ids) != EXPECTED_TEST_ROWS:
        raise CandidateError("Official Test or SampleSubmission row count is not 29,421.")
    if len(set(test_ids)) != EXPECTED_TEST_ROWS or len(set(sample_ids)) != EXPECTED_TEST_ROWS:
        raise CandidateError("Official Test or SampleSubmission contains duplicate IDs.")
    if test_ids != sample_ids:
        raise CandidateError("Official Test IDs do not exactly match SampleSubmission IDs and order.")
    actual_ids_hash = ordered_ids_sha256(test_ids)
    if actual_ids_hash != expected_ids_hash:
        raise CandidateError("Manifest test_ids_sha256 does not match official ordered Test IDs.")

    actual_csv_hash = sha256_file(csv_file)
    if actual_csv_hash != csv_hash:
        raise CandidateError("Submission CSV SHA-256 does not match its manifest.")
    submission_ids, submission_rows = _validate_submission_values(csv_file)
    if submission_rows != EXPECTED_TEST_ROWS:
        raise CandidateError("Submission CSV must contain exactly 29,421 data rows.")
    if len(set(submission_ids)) != EXPECTED_TEST_ROWS:
        raise CandidateError("Submission CSV contains duplicate IDs.")
    if submission_ids != test_ids:
        raise CandidateError("Submission IDs must match all official Test IDs in exact order.")

    return {
        "status": "AUDIT_PASS",
        "candidate_id": candidate_id,
        "competition": COMPETITION,
        "csv_sha256": actual_csv_hash,
        "manifest_sha256": sha256_file(manifest_file),
        "source_commit": manifest["source_commit"].lower(),
        "protocol_sha256": manifest["protocol_sha256"].lower(),
        "input_sha256": actual_input_hashes,
        "fit_rows": EXPECTED_TRAIN_ROWS,
        "test_rows": submission_rows,
        "test_ids_sha256": actual_ids_hash,
        "test_labels_used": False,
    }


def _parse_json_output(raw: str, label: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KaggleCLIError(f"Kaggle {label} response was not valid JSON.") from exc


def _run_cli(cli_path: str | Path, args: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    executable = Path(cli_path).expanduser()
    if not executable.is_file():
        raise KaggleCLIError("Kaggle CLI executable is unavailable.")
    try:
        return subprocess.run(
            [str(executable), *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise KaggleCLIError("Kaggle CLI operation timed out; outcome may be ambiguous.") from exc
    except OSError as exc:
        raise KaggleCLIError("Kaggle CLI could not be started.") from exc


def _history_rows(cli_path: str | Path) -> list[dict[str, Any]]:
    proc = _run_cli(
        cli_path,
        ["competitions", "submissions", COMPETITION, "--format", "json", "--page-size", "200"],
    )
    raw = (proc.stdout or "").strip()
    if proc.returncode != 0:
        combined = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
        if "no submissions found" in combined or "0 submissions" in combined:
            return []
        raise KaggleCLIError("Could not verify Kaggle submission history.")
    if not raw:
        return []
    parsed = _parse_json_output(raw, "submission history")
    if not isinstance(parsed, list):
        raise KaggleCLIError("Kaggle submission history had an unexpected structure.")
    rows: list[dict[str, Any]] = []
    for row in parsed:
        if not isinstance(row, dict):
            raise KaggleCLIError("Kaggle submission history had an unexpected row.")
        rows.append(
            {
                "ref": str(row.get("ref", "")),
                "file_name": str(row.get("fileName", "")),
                "message": str(row.get("description", "")),
                "status": _normal_status(row.get("status", "UNKNOWN")),
                "date": str(row.get("date", "")),
                "public_score": _safe_score(row.get("publicScore")),
            }
        )
    return rows


def _safe_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _normal_status(value: Any) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1].upper() or "UNKNOWN"


def _submission_limits(cli_path: str | Path) -> dict[str, int]:
    proc = _run_cli(cli_path, ["competitions", "submission-limits", COMPETITION, "--json"])
    if proc.returncode != 0:
        raise KaggleCLIError("Could not verify current Kaggle submission allowance.")
    parsed = _parse_json_output((proc.stdout or "").strip(), "submission allowance")
    if not isinstance(parsed, dict):
        raise KaggleCLIError("Kaggle submission allowance had an unexpected structure.")
    try:
        values = {key: int(parsed[key]) for key in ("numToday", "numTotal", "numAllowedNow")}
    except (KeyError, TypeError, ValueError) as exc:
        raise KaggleCLIError("Kaggle submission allowance omitted required counters.") from exc
    if any(value < 0 for value in values.values()):
        raise KaggleCLIError("Kaggle submission allowance returned invalid counters.")
    return values


def _ledger_path(project_root: Path) -> Path:
    base = project_root / "reports" / "regression_research"
    base.mkdir(parents=True, exist_ok=True)
    ledger = base / "kaggle_log.json"
    try:
        ledger.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise CandidateError("Research ledger path escaped the repository.") from exc
    return ledger


def _initial_ledger() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "competition": COMPETITION,
        "initial_submission": {
            "ref": INITIAL_SUBMISSION_REF,
            "counts_toward_additional_limit": INITIAL_SUBMISSION_COUNTS,
        },
        "events": [],
        "head_sha256": ZERO_HASH,
    }


def _validate_ledger(ledger: Any) -> dict[str, Any]:
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 1 or ledger.get("competition") != COMPETITION:
        raise CandidateError("Research ledger has an unsupported schema or competition.")
    initial = ledger.get("initial_submission")
    if not isinstance(initial, dict) or str(initial.get("ref")) != INITIAL_SUBMISSION_REF or initial.get("counts_toward_additional_limit") is not False:
        raise CandidateError("Research ledger initial submission identity is invalid.")
    events = ledger.get("events")
    if not isinstance(events, list):
        raise CandidateError("Research ledger events are invalid.")
    prev = ZERO_HASH
    for expected_seq, event in enumerate(events, start=1):
        if not isinstance(event, dict) or event.get("sequence") != expected_seq:
            raise CandidateError("Research ledger event sequence is invalid.")
        if event.get("previous_event_sha256") != prev:
            raise CandidateError("Research ledger hash chain is invalid.")
        stored = event.get("event_sha256")
        payload = {key: value for key, value in event.items() if key != "event_sha256"}
        calculated = _sha256_bytes(_canonical_bytes(payload))
        if stored != calculated:
            raise CandidateError("Research ledger event checksum is invalid.")
        prev = stored
    if ledger.get("head_sha256") != prev:
        raise CandidateError("Research ledger head checksum is invalid.")
    return ledger


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _initial_ledger()
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError("Research ledger is unreadable or invalid JSON.") from exc
    return _validate_ledger(value)


def _save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    content = json.dumps(ledger, ensure_ascii=False, indent=2) + "\n"
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _append_event(ledger: dict[str, Any], kind: str, candidate_id: str | None, **fields: Any) -> None:
    event: dict[str, Any] = {
        "sequence": len(ledger["events"]) + 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": kind,
        "candidate_id": candidate_id,
        "previous_event_sha256": ledger["head_sha256"],
        **fields,
    }
    event["event_sha256"] = _sha256_bytes(_canonical_bytes(event))
    ledger["events"].append(event)
    ledger["head_sha256"] = event["event_sha256"]


class _ExclusiveLock:
    def __init__(self, path: Path):
        self.path = path
        self.token = uuid.uuid4().hex
        self.acquired = False

    def __enter__(self) -> "_ExclusiveLock":
        payload = json.dumps({"token": self.token, "pid": os.getpid(), "created_utc": datetime.now(timezone.utc).isoformat()})
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise CandidateError("A Kaggle submission lock already exists; no upload was attempted.") from exc
        try:
            os.write(descriptor, payload.encode("utf-8"))
        finally:
            os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.acquired:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            # Keep an unreadable lock as a fail-closed signal for manual review.
            return


def _candidate_events(ledger: dict[str, Any], candidate_id: str) -> list[dict[str, Any]]:
    return [event for event in ledger["events"] if event.get("candidate_id") == candidate_id]


def _count_additional_candidates(ledger: dict[str, Any], history_rows: list[dict[str, Any]] | None = None) -> int:
    """Count this phase's submissions across ledger and Kaggle history once."""
    history_rows = history_rows or []
    keys: set[str] = set()
    history_by_candidate: dict[str, list[dict[str, Any]]] = {}
    prefix = "Samuel REG Research "
    for row in history_rows:
        if row.get("ref") == INITIAL_SUBMISSION_REF:
            continue
        message = str(row.get("message", ""))
        if not message.startswith(prefix):
            continue
        candidate_id = message[len(prefix):]
        history_by_candidate.setdefault(candidate_id, []).append(row)
        ref = str(row.get("ref", ""))
        keys.add(f"ref:{ref}" if ref else f"candidate:{candidate_id}")

    reserved_ids = {
        str(event.get("candidate_id"))
        for event in ledger["events"]
        if event.get("candidate_id")
        and event.get("event") in {"candidate_reserved", "candidate_reserved_from_history"}
    }
    for candidate_id in reserved_ids:
        associated = history_by_candidate.get(candidate_id, [])
        if associated:
            # History refs are already counted above; don't double-count the
            # durable local reservation for that same candidate.
            continue
        recorded_refs = {
            str(event.get("submission_ref"))
            for event in _candidate_events(ledger, candidate_id)
            if event.get("submission_ref")
        }
        if recorded_refs:
            keys.update(f"ref:{ref}" for ref in recorded_refs)
        else:
            keys.add(f"candidate:{candidate_id}")
    return len(keys)


def _history_match(rows: list[dict[str, Any]], candidate_id: str, csv_name: str) -> dict[str, Any] | None:
    expected_message = f"Samuel REG Research {candidate_id}"
    message_matches = [row for row in rows if row["message"] == expected_message]
    file_matches = [row for row in rows if row["file_name"] == csv_name]
    if any(row["file_name"] != csv_name for row in message_matches):
        raise CandidateError("Kaggle history has the candidate message attached to a different filename.")
    if any(row["message"] != expected_message for row in file_matches):
        raise CandidateError("Kaggle history already contains the candidate filename under a different message.")
    matches = [row for row in message_matches if row["file_name"] == csv_name]
    if len(matches) > 1:
        raise CandidateError("Kaggle history contains duplicate matching candidate submissions.")
    return matches[0] if matches else None


def _append_history_status(
    ledger: dict[str, Any],
    candidate_id: str,
    csv_name: str,
    csv_hash: str,
    row: dict[str, Any],
) -> None:
    if not _candidate_events(ledger, candidate_id):
        _append_event(
            ledger,
            "candidate_reserved_from_history",
            candidate_id,
            file_name=csv_name,
            csv_sha256=csv_hash,
        )
    latest = [
        event
        for event in _candidate_events(ledger, candidate_id)
        if event.get("event") in {"candidate_status", "candidate_outcome_unknown"}
    ]
    status = row["status"]
    ref = row["ref"]
    score = row.get("public_score")
    if latest and latest[-1].get("submission_ref") == ref and latest[-1].get("status") == status and latest[-1].get("public_score") == score:
        return
    _append_event(
        ledger,
        "candidate_status",
        candidate_id,
        file_name=csv_name,
        csv_sha256=csv_hash,
        submission_ref=ref,
        status=status,
        public_score=score,
        kaggle_date=row.get("date", ""),
    )


def _latest_candidate_status(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("event") in {"candidate_status", "candidate_outcome_unknown"}:
            return str(event.get("status", event.get("reason", "UNKNOWN")))
    return None


def execute_candidate(
    audit: dict[str, Any],
    csv_path: str | Path,
    *,
    kaggle_cli: str | Path = DEFAULT_KAGGLE_CLI,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Upload one approved candidate at most once; ambiguous outcomes stay held."""
    root = Path(project_root).resolve()
    csv_file = _resolve_owned_file(csv_path, root, "submissions/regression_research", ".csv")
    candidate_id = audit.get("candidate_id")
    if not isinstance(candidate_id, str) or not CANDIDATE_ID.fullmatch(candidate_id):
        raise CandidateError("Validated audit is missing a safe candidate_id.")
    if audit.get("status") != "AUDIT_PASS" or audit.get("competition") != COMPETITION:
        raise CandidateError("Only a successful audit for the fixed regression competition can be submitted.")
    if audit.get("csv_sha256") != sha256_file(csv_file) or csv_file.stem != candidate_id:
        raise CandidateError("CSV changed after validation; no submission was attempted.")

    ledger_path = _ledger_path(root)
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    with _ExclusiveLock(lock_path):
        ledger = _load_ledger(ledger_path)
        rows = _history_rows(kaggle_cli)
        history_match = _history_match(rows, candidate_id, csv_file.name)
        existing_events = _candidate_events(ledger, candidate_id)

        if history_match is not None:
            if not existing_events and _count_additional_candidates(ledger, rows) >= MAX_ADDITIONAL_SUBMISSIONS:
                raise CandidateError("Six additional research candidates are already recorded; candidate cannot be admitted.")
            _append_history_status(ledger, candidate_id, csv_file.name, audit["csv_sha256"], history_match)
            _save_ledger(ledger_path, ledger)
            return {
                "status": "HISTORY_RECONCILED",
                "candidate_id": candidate_id,
                "submission_ref": history_match["ref"],
                "kaggle_status": history_match["status"],
                "public_score_recorded": history_match.get("public_score") is not None,
                "uploaded_now": False,
            }

        if existing_events:
            status = _latest_candidate_status(existing_events) or "PENDING_OR_UNKNOWN"
            raise CandidateError(
                f"Candidate already has a ledger reservation ({status}) but no matching Kaggle history; no retry was attempted."
            )
        if _count_additional_candidates(ledger, rows) >= MAX_ADDITIONAL_SUBMISSIONS:
            raise CandidateError("Six additional research candidates are already reserved.")

        limits = _submission_limits(kaggle_cli)
        if limits["numAllowedNow"] <= 0:
            raise CandidateError("Kaggle reports no submission allowance remaining now.")

        message = f"Samuel REG Research {candidate_id}"
        _append_event(
            ledger,
            "history_preflight",
            None,
            existing_refs=[
                {
                    "ref": row["ref"],
                    "file_name": row["file_name"],
                    "message": row["message"],
                    "status": row["status"],
                    "date": row["date"],
                }
                for row in rows
            ],
            limits=limits,
        )
        _append_event(
            ledger,
            "candidate_reserved",
            candidate_id,
            file_name=csv_file.name,
            csv_sha256=audit["csv_sha256"],
            manifest_sha256=audit["manifest_sha256"],
            source_commit=audit["source_commit"],
            protocol_sha256=audit["protocol_sha256"],
            message=message,
            limits_before=limits,
        )
        # Persist a durable reservation before the external side effect. If the
        # process stops during upload, the next run must first reconcile history.
        _save_ledger(ledger_path, ledger)

        try:
            proc = _run_cli(
                kaggle_cli,
                ["competitions", "submit", COMPETITION, "-f", str(csv_file), "-m", message, "-q"],
                timeout=240,
            )
            if proc.returncode != 0:
                raise KaggleCLIError("Kaggle upload returned a nonzero exit status.")
        except KaggleCLIError:
            # A failed/timed-out client response may follow a successful server
            # upload. Inspect history once now; never repeat the upload here.
            try:
                failed_rows = _history_rows(kaggle_cli)
                failed_match = _history_match(failed_rows, candidate_id, csv_file.name)
            except (KaggleCLIError, CandidateError):
                failed_match = None
            if failed_match is not None:
                _append_history_status(ledger, candidate_id, csv_file.name, audit["csv_sha256"], failed_match)
                _save_ledger(ledger_path, ledger)
                return {
                    "status": failed_match["status"],
                    "candidate_id": candidate_id,
                    "uploaded_now": "UNKNOWN",
                    "submission_ref": failed_match["ref"],
                    "public_score_recorded": failed_match.get("public_score") is not None,
                }
            _append_event(
                ledger,
                "candidate_outcome_unknown",
                candidate_id,
                file_name=csv_file.name,
                csv_sha256=audit["csv_sha256"],
                reason="CLI_FAILED_OR_TIMED_OUT_HISTORY_RECONCILIATION_REQUIRED",
            )
            _save_ledger(ledger_path, ledger)
            return {
                "status": "OUTCOME_UNKNOWN",
                "candidate_id": candidate_id,
                "uploaded_now": "UNKNOWN",
                "next_action": "Run --refresh to reconcile Kaggle history; this candidate will not be retried automatically.",
            }

        try:
            after_rows = _history_rows(kaggle_cli)
            after_match = _history_match(after_rows, candidate_id, csv_file.name)
        except (KaggleCLIError, CandidateError):
            after_match = None
        if after_match is None:
            _append_event(
                ledger,
                "candidate_status",
                candidate_id,
                file_name=csv_file.name,
                csv_sha256=audit["csv_sha256"],
                status="CLI_ACCEPTED_PENDING_HISTORY",
                public_score=None,
            )
            _save_ledger(ledger_path, ledger)
            return {
                "status": "CLI_ACCEPTED_PENDING_HISTORY",
                "candidate_id": candidate_id,
                "uploaded_now": "CLI_ACCEPTED_UNCONFIRMED",
                "submission_ref": None,
                "public_score_recorded": False,
            }

        _append_history_status(ledger, candidate_id, csv_file.name, audit["csv_sha256"], after_match)
        _save_ledger(ledger_path, ledger)
        return {
            "status": after_match["status"],
            "candidate_id": candidate_id,
            "uploaded_now": True,
            "submission_ref": after_match["ref"],
            "public_score_recorded": after_match.get("public_score") is not None,
        }


def refresh_ledger(*, kaggle_cli: str | Path = DEFAULT_KAGGLE_CLI, project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Refresh unresolved candidate statuses without submitting anything."""
    root = Path(project_root).resolve()
    ledger_path = _ledger_path(root)
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    with _ExclusiveLock(lock_path):
        ledger = _load_ledger(ledger_path)
        rows = _history_rows(kaggle_cli)
        changed = 0
        candidates = {
            event.get("candidate_id")
            for event in ledger["events"]
            if event.get("candidate_id") and event.get("event") in {"candidate_reserved", "candidate_reserved_from_history"}
        }
        for candidate_id in sorted(candidates):
            events = _candidate_events(ledger, candidate_id)
            reservation = next(
                event for event in events if event.get("event") in {"candidate_reserved", "candidate_reserved_from_history"}
            )
            match = _history_match(rows, candidate_id, str(reservation.get("file_name", "")))
            if match is None:
                continue
            before = len(ledger["events"])
            _append_history_status(
                ledger,
                candidate_id,
                str(reservation.get("file_name", "")),
                str(reservation.get("csv_sha256", "")),
                match,
            )
            changed += len(ledger["events"]) - before
        if changed:
            _save_ledger(ledger_path, ledger)
        current = {}
        for candidate_id in sorted(candidates):
            events = _candidate_events(ledger, candidate_id)
            current[candidate_id] = _latest_candidate_status(events) or "PENDING_OR_UNKNOWN"
        return {
            "status": "REFRESH_COMPLETE",
            "competition": COMPETITION,
            "candidates": current,
            "new_events": changed,
            "uploaded": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Candidate CSV under submissions/regression_research/")
    parser.add_argument("--manifest", help="Approved manifest under reports/regression_research/submissions/")
    parser.add_argument("--execute", action="store_true", help="Upload only after exact main-agent manifest approval.")
    parser.add_argument("--refresh", action="store_true", help="Refresh existing candidate statuses; never upload.")
    parser.add_argument("--kaggle-cli", default=str(DEFAULT_KAGGLE_CLI), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.refresh:
        if args.csv or args.manifest or args.execute:
            parser.error("--refresh cannot be combined with candidate paths or --execute.")
        try:
            result = refresh_ledger(kaggle_cli=args.kaggle_cli)
        except (CandidateError, KaggleCLIError) as exc:
            print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0

    if not args.csv or not args.manifest:
        parser.error("--csv and --manifest are required unless --refresh is used.")
    try:
        audit = audit_candidate(args.csv, args.manifest)
        result = (
            execute_candidate(audit, args.csv, kaggle_cli=args.kaggle_cli)
            if args.execute
            else audit
        )
    except (CandidateError, KaggleCLIError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
