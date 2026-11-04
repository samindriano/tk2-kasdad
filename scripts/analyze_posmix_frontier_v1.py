"""Descriptive error audit of already frozen POS/A02/MIX50 predictions; zero fits.

No policy selection, Test access, threshold search, or production ledger writes.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
OUT = ROOT / "reports/optimization_v3/posmix_followup_v1"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(y, prob):
    labels = prob >= .5
    return dict(n=int(len(y)), positives=int(y.sum()), predicted_positive=int(labels.sum()),
                errors=int(np.sum(labels != y)), macro_f1=float(f1_score(y, labels, average="macro")),
                auc=float(roc_auc_score(y, prob)), logloss=float(log_loss(y, prob, labels=[0, 1])))


def main():
    before = sha(LEDGER)
    expected = json.loads((ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/POLICY_METRICS.json").read_text())
    assert sha(SOURCE) == expected["prediction_npz_sha256"]
    result = dict(adaptive_descriptive=True, overlapping_origins=True, new_fits=0,
                  test_accessed=False, policy_search=False, source_sha256=sha(SOURCE), origins={})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    with np.load(SOURCE, allow_pickle=False) as z:
        for origin, ax in zip((67, 75), axes):
            key = f"E{origin}_"
            y = z[key + "y"].astype(int)
            pp, pa, pm = (z[key + n] for n in ("pP", "pA", "MIX50"))
            miss = z[key + "missing"].astype(bool)
            cycles = z[key + "cycles"]
            assert np.array_equal(pm, np.where(miss, pp, .5 * pp + .5 * pa))
            assert np.array_equal(pm[miss], pp[miss])
            lp, la, lm = (v >= .5 for v in (pp, pa, pm))
            present = ~miss
            agree = present & (lp == la)
            disagree = present & (lp != la)
            summary = dict(metrics={n: metric(y, p) for n, p in (("POS", pp), ("A02", pa), ("MIX50", pm))},
                           present=int(present.sum()), missing=int(miss.sum()),
                           agree_present=int(agree.sum()), jointly_wrong_present=int(np.sum(agree & (lp != y))),
                           disagree_present=int(disagree.sum()),
                           POS_correct_in_disagreement=int(np.sum(disagree & (lp == y))),
                           A02_correct_in_disagreement=int(np.sum(disagree & (la == y))),
                           MIX_correct_in_disagreement=int(np.sum(disagree & (lm == y))),
                           fixes_POS=int(np.sum((lp != y) & (lm == y))),
                           breaks_POS=int(np.sum((lp == y) & (lm != y))),
                           fixes_A02=int(np.sum((la != y) & (lm == y))),
                           breaks_A02=int(np.sum((la == y) & (lm != y))), per_cycle=[], text_groups={}, availability={})
            for group, mask in (("present", present), ("missing", miss)):
                summary["availability"][group] = {n: metric(y[mask], p[mask]) for n, p in (("POS", pp), ("A02", pa), ("MIX50", pm))}
            for cycle in np.unique(cycles):
                m = cycles == cycle
                cell = dict(cycle=int(cycle), **{n: metric(y[m], p[m]) for n, p in (("POS", pp), ("A02", pa), ("MIX50", pm))})
                summary["per_cycle"].append(cell)
            for group in np.unique(z[key + "group_text_load"]):
                m = z[key + "group_text_load"] == group
                summary["text_groups"][str(group)] = {n: metric(y[m], p[m]) for n, p in (("POS", pp), ("A02", pa), ("MIX50", pm))}
            rows = summary["per_cycle"]
            for comparator, color in (("POS", "#2563eb"), ("A02", "#d97706")):
                ax.plot([r["cycle"] for r in rows], [r["MIX50"]["macro_f1"] - r[comparator]["macro_f1"] for r in rows], marker="o", label=f"MIX50 minus {comparator}", color=color)
            ax.axhline(0, color="gray", linewidth=.8)
            ax.set(title=f"Origin E{origin}: overlapping sensitivity views", xlabel="cycle_code", ylabel="Macro F1 difference")
            ax.legend(fontsize=8)
            result["origins"][str(origin)] = summary
    # Label-free decomposition of the existing positional head. This does not
    # score a dropped component or define a new deployment policy.
    result["original_position_head_decomposition"] = []
    pilot = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
    offset = ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
    for origin in (67, 75):
        manifest = json.loads((pilot / f"MATRIX_MANIFEST_E{origin}.json").read_text())
        base_width = manifest["matrices"]["score_base"]["shape"][1]
        for arm, matrix_name in (("REAL", "score_position"), ("SHUFFLED", "score_shuffled_position")):
            item = manifest["matrices"][matrix_name]
            path = ROOT / item["path"]
            assert sha(path) == item["sha256"]
            x = sparse.load_npz(path)[:, base_width:].tocsr()
            receipt = json.loads((offset / "receipts" / f"POS_BAG_OFFSET_V1_E{origin}_{arm}.json").read_text())
            gamma_path = ROOT / receipt["artifact_path"]
            assert sha(gamma_path) == receipt["artifact_sha256"]
            with np.load(gamma_path, allow_pickle=False) as z:
                g = z["gamma"]
            assert x.shape[1] == len(g) and len(g) % 3 == 0
            width = len(g) // 3
            blocks = [x[:, j*width:(j+1)*width] for j in range(3)]
            gs = np.split(g, 3)
            gmean = (gs[0] + gs[1] + gs[2]) / 3
            common = np.asarray((blocks[0] + blocks[1] + blocks[2]) @ gmean).ravel()
            contrast = sum(np.asarray(b @ (v-gmean)).ravel() for b, v in zip(blocks, gs))
            total = np.asarray(x @ g).ravel()
            assert np.allclose(common + contrast, total, atol=1e-12, rtol=0)
            energy = np.asarray(x.multiply(x).sum(axis=1)).ravel()
            ce = np.asarray((blocks[0]+blocks[1]+blocks[2]).power(2).sum(axis=1)).ravel() / 3
            valid = energy > 0
            result["original_position_head_decomposition"].append(dict(
                origin=origin, arm=arm, scored_rows=x.shape[0], matrix_sha256=item["sha256"],
                gamma_sha256=receipt["artifact_sha256"],
                common_mode_coefficient_energy_fraction=float(3 * np.dot(gmean, gmean) / np.dot(g, g)),
                common_mode_feature_energy_fraction_mean=float(np.mean(ce[valid] / energy[valid])),
                mean_abs_common_logit_present=float(np.mean(np.abs(common[valid]))),
                mean_abs_contrast_logit_present=float(np.mean(np.abs(contrast[valid]))),
                reconstruction_max_abs_error=float(np.max(np.abs(common+contrast-total))))
            )
    assert sha(LEDGER) == before
    result["ledger_sha256_before_after"] = before
    result["source_script_sha256"] = sha(Path(__file__))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ERROR_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    fig.savefig(OUT / "cycle_complementarity.png", dpi=160)
    plt.close(fig)
    print(json.dumps({o: {k: v for k, v in s.items() if k not in ("per_cycle", "text_groups")} for o, s in result["origins"].items()}, indent=2))


if __name__ == "__main__":
    main()
