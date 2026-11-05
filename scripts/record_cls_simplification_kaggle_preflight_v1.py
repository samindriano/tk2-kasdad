"""Capture a one-time authenticated Kaggle preflight; never uploads anything."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/optimization_v3/cls_simplification_benchmark_v1"
OUTPUT = REPORT_DIR / "KAGGLE_PREUPLOAD_PREFLIGHT.json"
RUN_MANIFEST = REPORT_DIR / "RUN_MANIFEST.json"
CSV = ROOT / "submissions/optimization_v3/cls_simplification_benchmark_v1/classification_no_pos_e79.csv"
COMPETITION = "tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027"
TEAM_NAME = "kasdud lovers"
SUBMISSION_MESSAGE = "Samuel CLS Simplified A02 Only Frozen Benchmark"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kaggle_json(*args: str) -> Any:
    result = subprocess.run(
        ["kaggle", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    csv_sha = sha256_file(CSV)
    if csv_sha != manifest["csv"]["sha256"]:
        raise ValueError("Candidate CSV changed after the committed run manifest")

    slug = COMPETITION
    listing = kaggle_json("competitions", "list", "--search", slug, "--format", "json")
    exact = [row for row in listing if row.get("ref", "").rstrip("/").endswith("/" + slug)]
    if len(exact) != 1:
        raise ValueError("Could not identify exactly one live competition entry")
    competition = exact[0]
    rules_pages = kaggle_json(
        "competitions", "pages", "list", "--content", "--page-name", "rules", "--format", "json", slug
    )
    rules = next((page["content"] for page in rules_pages if page.get("name") == "rules"), None)
    if not isinstance(rules, str) or not rules.strip():
        raise ValueError("Live competition rules page was not returned")
    if "melalui tim" not in rules.lower():
        raise ValueError("Live rules do not confirm team-only submission")
    if "data tambahan di luar dari data yang diberikan" not in rules.lower():
        raise ValueError("Live rules data-source restriction changed or is missing")

    limits = kaggle_json("competitions", "submission-limits", slug, "--json")
    history = kaggle_json("competitions", "submissions", slug, "--format", "json")
    leaderboard = kaggle_json("competitions", "leaderboard", "--show", "--format", "json", slug)
    team = next((row for row in leaderboard if row.get("teamName", "").casefold() == TEAM_NAME), None)
    if team is None:
        raise ValueError("Expected official Kaggle team is absent from current leaderboard")
    team_id = int(team["teamId"])
    team_submissions = kaggle_json(
        "competitions", "team-submissions", str(team_id), "--format", "json"
    )

    deadline = competition.get("deadline")
    has_entered = competition.get("userHasEntered") is True
    quota_value = int(limits.get("numAllowedNow", 0))
    filename = CSV.name
    filename_already_present = any(row.get("fileName") == filename for row in history)
    message_already_present = any(row.get("description") == SUBMISSION_MESSAGE for row in history)
    checks = {
        "authenticated_api_replies": True,
        "user_has_entered_exact_competition": has_entered,
        "deadline_returned": bool(deadline),
        "live_rules_require_team_submission": "melalui tim" in rules.lower(),
        "team_found_on_leaderboard": team is not None,
        "candidate_filename_absent_from_history": not filename_already_present,
        "submission_message_absent_from_history": not message_already_present,
        "at_least_one_submission_allowed_now": quota_value >= 1,
        "candidate_csv_hash_matches_run_manifest": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Kaggle preflight failed: {checks}")

    evidence = {
        "schema_version": 1,
        "status": "PREFLIGHT_PASS_READY_FOR_SINGLE_UPLOAD",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": manifest["source_commit"],
        "benchmark_id": manifest["benchmark_id"],
        "competition": competition,
        "rules": {"name": "rules", "content": rules, "sha256_utf8": hashlib.sha256(rules.encode("utf-8")).hexdigest()},
        "submission_limits": limits,
        "team": {
            "id": team_id,
            "name": team.get("teamName"),
            "current_leaderboard_entry": team,
            "team_submissions": team_submissions,
        },
        "personal_or_team_submission_history_returned_by_cli": history,
        "idempotency": {
            "candidate_filename": filename,
            "candidate_filename_absent_from_history": not filename_already_present,
            "submission_message": SUBMISSION_MESSAGE,
            "message_absent_from_history": not message_already_present,
            "local_duplicate_vectors": manifest["csv"]["duplicate_local_vectors"],
            "changed_rows_vs_current_public_champion": manifest["csv"]["required_comparisons"]["current_public_champion"]["changed_rows"],
            "changed_rows_vs_historical_a02": manifest["csv"]["required_comparisons"]["historical_a02"]["changed_rows"],
        },
        "candidate": {
            "path": manifest["csv"]["path"],
            "sha256": csv_sha,
            "label_vector_sha256": manifest["csv"]["label_vector_sha256"],
            "rows": manifest["csv"]["rows"],
            "submission_message": SUBMISSION_MESSAGE,
        },
        "checks": checks,
        "upload_performed": False,
        "note": "One authorized Classification upload maximum. Kaggle API/CLI only; no Chrome. Recheck team quota and idempotency immediately before submit.",
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
