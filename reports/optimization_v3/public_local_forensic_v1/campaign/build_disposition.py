"""Build the 45-card disposition table from the authoritative compact registry.

Read-only source inputs; writes only CARD_DISPOSITION.md beside this script.
No model, labels, prediction cache, Kaggle service, or raw event log is opened.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "CARD_DISPOSITION.md"
REGISTRY = ROOT / "reports/optimization_v3/batch1/batch_ledger.json"
CARD_REVIEW = ROOT / "reports/optimization_v3/postscreen_review/CARD_REVIEW.md"

# Short descriptions are keyed to the fixed card IDs. They characterize only
# the frozen recipe actually tested and do not expand its verdict to a family.
DESCRIPTIONS = {
    "C01": ("Missing-stream route replaced by structured HGB", "Cached full-six recipe lost; missing W1 classification/recall also fell."),
    "C02": ("Fit the hybrid LR only on stream-present rows", "Primary slightly negative; excluding missing rows did not help this recipe."),
    "C03": ("Reallocate sparse vocabulary between unigram and bigram", "Small negative primary and H12; this allocation did not help."),
    "C04": ("Widen sparse vocabulary to 60k", "Primary negative; small H12 increase did not offset primary failure."),
    "C05": ("Lower min_df to 2 with 60k vocabulary", "Both aggregate views declined."),
    "C06": ("Strict-prior-cycle category novelty", "Primary negative; chronological construction passed but gain did not appear."),
    "C07": ("Category frequency/rarity features", "Near-zero mixed effects across views."),
    "C08": ("Smoothed chronological category target rates", "Tiny primary gain with H12 harm near the allowed boundary."),
    "C09": ("TF-IDF by entry-friction LR interaction", "Tiny primary gain, negative H12; final comparison was row-matched after scope reconciliation."),
    "C10": ("Add trigram block", "Positive but below the frozen primary margin (+0.000343 vs +0.0005 required)."),
    "C11": ("Word plus char sparse boosting", "Thin pilot advance; H12 slightly negative and late gain much smaller than early."),
    "C12": ("Char-only sparse boosting", "Primary positive, but H12 harm exceeded its limit; late view also declined."),
    "C13": ("Structured features plus SVD32 with HGB", "Negative primary and H12; does not isolate whether compression or learner caused it."),
    "C14": ("Random feature partitions plus LR", "Broad negative aggregate/window/subgroup evidence."),
    "C15": ("Two-score prequential meta-head (late profile)", "H12_L positive, but W4+W6 primary negative and subgroup policy failed."),
    "C16": ("Sparse-model capacity/regularization variant", "Pilot advanced; late gain was smaller. Later continuation failed qualification; that is a separate stage."),
    "C17": ("Second sparse-capacity variant", "Pilot advanced but overlapped C16 family and showed near-zero late gain."),
    "C18": ("Random-threshold sparse boosting", "Near-zero aggregates plus W1 text-load Q3 Macro-F1 harm 0.010471 (n=772); registry gate scope was narrower than diagnostics."),
    "C19": ("Dense XGBoost auxiliary blend", "Positive H12, but primary below margin and supported text-load Q4 subgroup guard failed."),
    "C20": ("Q80-volume route (cached full-six)", "Negative primary and H12; route-specific result, not evidence against all availability routing."),
    "R01": ("Sparse LinearSVR L1", "Positive aggregate, but target-high subgroup harm and late reversal failed risk policy."),
    "R02": ("Widen sparse vocabulary", "Positive but below the +0.03 MAE materiality gate."),
    "R03": ("Reallocate unigram/bigram vocabulary", "Positive but below materiality; does not show every bigram design fails."),
    "R04": ("Feature hashing", "Positive but below materiality; one technical attempt failed, then valid result was saved and verified."),
    "R05": ("Binary token presence without IDF", "Positive but below materiality."),
    "R06": ("Char sparse L1 LightGBM", "Pilot advance with strong gains; later evaluation found a localized missing-stream guard failure."),
    "R07": ("Replace text geometry with NMF32", "Large negative aggregate result for this replacement."),
    "R08": ("Add text-derived scalar features", "Strong negative aggregate and tail evidence for this recipe."),
    "R09": ("LightGBM Huber objective", "Negative aggregate and tail result; not a universal rejection of robust losses (R22 differs)."),
    "R10": ("Fair-loss variant conditional on R09", "Prerequisite was not met; zero fits, unmeasured rather than a candidate failure."),
    "R11": ("Nine ordinal heads plus conditional median", "Pilot advanced but used 18 predictor fits and had smaller H12 gain than R22/R06."),
    "R12": ("LambdaRank within-cycle percentile blend", "Strong aggregate, but missing-stream harm failed guards; later present-only route was post hoc."),
    "R13": ("CatBoost ranker conditional on R12", "Composite prerequisite not met; zero fits, unmeasured. Missing Spearman evidence alone is not the full reason."),
    "R14": ("Entry-friction specialist", "Positive aggregate but H12_L target>=95 harm 0.555867 on n=1,057 exceeded 0.25."),
    "R15": ("10% structured missing-stream expert", "Small positive late aggregate below margin; does not test all fallback policies."),
    "R16": ("Half-life-8 recency weighting", "Positive but below the primary margin; early/late effects diverged."),
    "R17": ("Affine OOF L1 calibration", "Positive late aggregates but low-tail risk guard failed."),
    "R18": ("Past-four-cycle median residual offset", "Small positive late aggregate but low-tail harm; zero estimator fits by design."),
    "R19": ("Past-eight-cycle EF-specific median offset", "Small positive late aggregate but tail guard failed; deterministic offset, no new estimator fit."),
    "R20": ("Supervised selection of 128 tokens", "Strong negative aggregate and tail result for this selection/compression recipe."),
    "R21": ("XGBoost absolute-error auxiliary", "Pilot advanced; one failed attempt was retained in accounting; missing-stream risk remained diagnostic."),
    "R22": ("XGBoost pseudo-Huber slope-5 auxiliary", "Strong pilot advance; full-six continuation later qualified internally but public score did not beat R06."),
    "R23": ("ExtraTrees L1", "Large positive aggregate, but missing/tail-high subgroup risk; post hoc route reduced late gain."),
    "R24": ("TF-IDF by entry-friction Ridge interaction", "Positive but far below the +0.03 MAE materiality gate."),
    "R25": ("Unnormalized token counts", "Positive but below materiality."),
}


def gain(cell: dict | None, metric: str) -> str:
    if not cell or cell.get("gain_positive_is_better") is None:
        return "—"
    rows = cell.get("rows")
    value = float(cell["gain_positive_is_better"])
    return f"{rows:,}; {value:+.6f} {metric}"


def evidence_links(card: dict) -> str:
    artifacts = card.get("artifacts") or {}
    links = []
    for label in ("result", "verification", "gate"):
        path = artifacts.get(label)
        if not path:
            continue
        rel = Path(path).relative_to("reports/optimization_v3").as_posix()
        links.append(f"[{label}](../../{rel})")
    return "; ".join(links) if links else "UNKNOWN"


def disposition(card: dict) -> tuple[str, str]:
    cid = card["card_id"]
    if card["pilot_status"] == "SKIPPED":
        if cid == "R13":
            return "Prerequisite skip; unmeasured", "Composite prerequisite required R12 to miss aggregate MAE gates and improve within-cycle Spearman at both origins; R12 passed aggregate and the saved late Spearman comparison did not improve. Zero fits."
        return "Prerequisite skip; unmeasured", str(card.get("skip_reason") or "Skip reason unavailable")
    if card["pilot_status"] == "ADVANCE":
        return "Pilot advance; not promotion", "The pilot gate allowed the next stage; this is not a final qualification or champion decision."
    if card.get("pilot_blockers", {}).get("supported_subgroup_failure_count", 0) > 0:
        bucket = "Valid aggregate/risk-gate failure"
    elif cid in {"R01", "R12", "R14", "R17", "R18", "R19", "R23", "C12", "C15", "C18", "C19"}:
        bucket = "Positive/mixed signal; risk or horizon guard"
    elif card.get("primary") and card["primary"].get("gain_positive_is_better", 0) > 0:
        bucket = "Positive but below frozen margin"
    else:
        bucket = "Negative/near-null for frozen recipe"
    reason = DESCRIPTIONS[cid][1]
    return bucket, reason


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry_sha = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    review_sha = hashlib.sha256(CARD_REVIEW.read_bytes()).hexdigest()
    cards = registry.get("cards", [])
    ids = [c.get("card_id") for c in cards]
    expected = {f"C{i:02}" for i in range(1, 21)} | {f"R{i:02}" for i in range(1, 26)}
    if len(cards) != 45 or set(ids) != expected or len(set(ids)) != 45:
        raise SystemExit("STOP: registry does not contain exactly the expected unique 45 cards")
    if set(DESCRIPTIONS) != expected:
        raise SystemExit("STOP: curated mechanism/failure descriptions do not cover exactly 45 card IDs")

    lines = [
        "# V3 screening: 45-card retrospective disposition",
        "",
        "Generated from `reports/optimization_v3/batch1/batch_ledger.json` (the compact authoritative registry); mechanism notes are checked against `postscreen_review/CARD_REVIEW.md`. This is a new retrospective label only; historical verdicts are copied unchanged.",
        f"Source SHA-256: registry `{registry_sha}`; card review `{review_sha}`.",
        "Failure explanations use the reviewed numeric gate/result and actual supported-group evidence; generic recursively collected `False` flags are not treated as failed gates. Recipe/state flags such as an OOF target-cache-opened marker are provenance annotations, not performance failures.",
        "",
        "**Gain convention:** Classification Macro-F1 gain = candidate − baseline; Regression MAE gain = baseline − candidate. Standard pilot primary/H12 are W1+W4 (n=9,064)/H12 (n=38,588); LATE uses W4+W6 (n=13,441)/H12_L (n=20,903); CACHE_SIX uses saved six-origin rows. Do not rank different profiles, horizons, or task metrics together. Primary and H12 can reuse overlapping labels.",
        "",
        "| ID | Profile | Mechanism actually tested | Local primary; H12 gain | Historical status / gate | Retrospective bucket and failure mode | Exact saved evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cards:
        cid = c["card_id"]
        task = c["task"].lower()
        metric = "Macro-F1" if task == "classification" else "MAE"
        mechanism, _ = DESCRIPTIONS[cid]
        bucket, reason = disposition(c)
        p = c.get("primary")
        h = c.get("h12")
        p_label = c.get("primary_view_label") or "primary"
        h_label = "H12_L" if c.get("evaluation_profile") == "LATE" else "H12"
        if c.get("evaluation_profile") == "CACHE_SIX":
            h_label = "same cached six-origin view"
        metric_pair = f"{p_label}: {gain(p, metric)}<br>{h_label}: {gain(h, metric)}"
        lines.append(
            f"| {cid} | {task.title()} / {c.get('evaluation_profile')} | {mechanism} | {metric_pair} | {c.get('pilot_status')} / {c.get('gate_status')} | **{bucket}.** {reason} | {evidence_links(c)} |"
        )
    lines += [
        "",
        "## Later families and public outcomes",
        "",
        "Public status below comes from the read-only official CLI snapshots in `../KAGGLE_CLASSIFICATION_SNAPSHOT_20261104.json` (SHA-256 `9ddbd945733af3c689f2024bf4598ee9a06fccf7dd372567ddd004f6b1643daa`) and `../KAGGLE_REGRESSION_SNAPSHOT_20261104.json` (SHA-256 `ca7d66aa2567b99460c11b0f47e7653a0d07490ecce34e29e6ad445446cdae36`). They contain 19 Classification records (18 COMPLETE, one ERROR) and 18 Regression records (18 COMPLETE). They show current public results, not private outcomes or public-subset denominators. Exact local-to-upload identity is only claimed where a local manifest/receipt supplies it; otherwise it is marked UNKNOWN.",
        "",
        "| Later family / recipe | Local evidence and original disposition | Public record in the Nov 4 snapshot | Retrospective reading / exact-recipe status |",
        "|---|---|---|---|",
        "| C16/C11/C17 sparse and char variants | C16/C17 advanced at pilot; C11 advanced thinly. C16 continuation later failed its primary/late/text-load Q4 guards; C11 was not the winning continuation. | C16 ref 57871989 .82844; C11 ref 57874815 .82739; C17 ref 57874808 .82825. | Publicly measured family outputs; none beats current MIX50 .83144. Do not treat pilot advance as final qualification. |",
        "| C19 XGB/SVD32 | Continuation primary full-six .829944→.830611 (+.000667, n=38,588); H12 pooled .802966→.807117 (+.004152, n=38,588). W6 −.002599; W2/Q3 n=1,450 F1 harm .010461 and recall harm .017857. | C19 full-Train benchmark ref 57907572 .82760. | Same named XGB/SVD32 family was publicly benchmarked; exact full-six temporal continuation recipe was not the uploaded model. Public is −.00148 vs Sparse25 .82908 and −.00384 vs MIX50 .83144. |",
        "| A02 continuation / positional offset / MIX50 | A02 passed its continuation scientific gates; local full-six gain +.002101 vs Sparse25. POS_BAG_OFFSET remained diagnostic-only: E67 H12 +.001151 vs bag, below +.003 required; E75 W6 +.001059. | A02 ref 57909983 .82930; POSOFFSET ref 57955753 .83123; MIX50 ref 57956714 .83144. | POSOFFSET beat the earlier A02 public score by .00193, but sits .00021 below current MIX50. This shows baseline choice/order matters; no automatic replacement. |",
        "| CGEOM / joint positional / POS_MOMENTS / bigram / REAL contrast | Later experimental families with their own local gates; historical failures remain unchanged. | CGEOM ref 57955748 .82884; JointPos ref 57955761 .82911; Moments ref 57976581 .82979; Bigram ref 57981852 .83064; Contrast ref 57987123 .82935. | All below MIX50 .83144 by .00020–.00260. They are real public observations; no public-label significance or private-score inference. |",
        "| R06 / R22 / R21 / R11 | R22 qualified internal continuation vs champion: primary gain .039699, H12 .124873. R06 has strong local aggregate but later missing-W1 subgroup harm .395556 on n=392 exceeded .25; R22/R06 primary and H12 rankings vary. | R06 ref 57874821 MAE 11.16620; R22 ref 57871993 11.30535; R21 ref 57874835 11.29314; R11 ref 57874829 11.39469. | R06 is best among these public scores; R22's internal qualification does not imply public superiority. The guard encodes a different risk objective from pooled public MAE. |",
        "| Research Next N04_DIAGNOSTIC density weighting | Pooled primary gain +.026578 vs R22 and +.013904 vs shuffled control; H12 +.012444/+ .002635. Across six windows, gains were positive in 4/6 vs R22 and 3/6 vs shuffle; H12 missing-stream harm vs R22 +.038067. | No public submission in snapshot. | Diagnostic-only: positive pooled signal with mixed windows/subgroup risk; evidence against equating gate failure with zero predictive value, but not qualified for continuation. |",
        "| R12 current route / RNBR / positional L1 / direct position | R12 posthoc present-only route has local gains +.285422 primary and +.306170 H12 vs legacy after exact missing fallback; this was selected after outcomes. RNBR continuation passed its bounded local gates but had an inherited H12_L loss vs R06. Direct-position pilot included E67 H12 and E75 W6; the E75 W6 gain was negative and E71 W5 was not executed. | R12 current-R06-route ref 57987865 11.18673; RNBR ref 57926493 11.19115; positional L1 ref 57975981 11.24513; direct-position ref 57988146 11.14400. | These are public observations, not common validation denominators. Direct-position is publicly positive vs R06; local recipe/baseline equivalence is under the separate REG audit. R12 upload-to-posthoc-policy identity is UNKNOWN in this lane. |",
        "",
        "**Later research checkpoint:** Research Next N00–N06 found no candidate eligible for the original continuation queue; N04_DIAGNOSTIC showed positive pooled primary/H12 average gains but mixed windows and missing-stream harm, and remained diagnostic-only. This supports neither blanket rejection nor automatic reopening. See `../research_next_v1/CAMPAIGN_SUMMARY.md`.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}; cards={len(cards)}; unique={len(set(ids))}")


if __name__ == "__main__":
    main()
