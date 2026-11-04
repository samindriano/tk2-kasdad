"""Zero-fit Regression readiness/error audit using explicit read-only source root.

No Test, Classification targets, estimators, indexes, or shared-ledger writes.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

WORKTREE = Path(__file__).resolve().parents[1]
OUT = WORKTREE / "reports/optimization_v3/regression_position_frontier_v1"
RUN = "experiments/runs/optimization_v3_research_next_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(y, p):
    residual = y-p
    return dict(n=int(len(y)), mae=float(np.mean(np.abs(residual))),
                median_y_minus_prediction=float(np.median(residual)),
                mean_y_minus_prediction=float(np.mean(residual)),
                spearman=float(spearmanr(y, p).statistic))


def main(source):
    trainpath = source / "data/raw/regression/Train.csv"
    assert sha(trainpath) == "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
    frame = pd.read_csv(trainpath, usecols=["record_id", "cycle_code", "signal_stream", "threat_rank"], dtype={"record_id": str})
    # Explicitly exclude the Classification target at the CSV reader boundary.
    classframe = pd.read_csv(source / "data/raw/classification/Train.csv", usecols=["record_id", "cycle_code", "signal_stream"], dtype={"record_id": str})
    assert np.array_equal(frame.record_id, classframe.record_id)
    assert np.array_equal(frame.cycle_code, classframe.cycle_code)
    assert np.array_equal(frame.signal_stream.fillna(""), classframe.signal_stream.fillna(""))
    result = dict(status="ZERO_FIT_READINESS_PASS", classification_targets_read=False,
                  test_accessed=False, new_fits=0, shared_ledger_written=False,
                  adaptive_development=True, origins_overlap=True,
                  regression_train_sha256=sha(trainpath), origins={})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for origin, ax in zip((67, 75), axes):
        folder = "RNBR_PILOT_OFFICIAL_V1/recovered_bundles" if origin == 67 else "RNBR_PILOT_OFFICIAL_V1/bundles"
        name = f"{RUN}/{folder}/RNBR_PILOT_E{origin}_OBSERVED/arrays.npz"
        path = source/name
        prior = json.loads((source/"reports/optimization_v3/rnbr_postpublic_v1/audit.json").read_text())
        expected = {k.replace("\\", "/"):v for k,v in prior["sources_sha256"].items()}
        assert sha(path) == expected[name]
        with np.load(path, allow_pickle=False) as z:
            rows=z["score_rows"].astype(int); ids=z["score_record_ids"].astype(str)
            cycles=z["score_cycles"].astype(int); pred=z["baseline_prediction"].astype(float)
            donors=z["donor_rows"].astype(int); donorcycles=z["donor_cycles"].astype(int)
            donorids=z["donor_record_ids"].astype(str)
            donorcut=z["donor_source_cutoffs"].astype(int)
            donorbase=z["donor_r06_predictions"].astype(float)
            donorres=z["donor_residuals"].astype(float)
        assert np.array_equal(frame.record_id.iloc[rows].to_numpy(), ids)
        assert np.array_equal(frame.cycle_code.iloc[rows].to_numpy(), cycles)
        assert np.array_equal(frame.record_id.iloc[donors].to_numpy(), donorids)
        assert np.array_equal(frame.cycle_code.iloc[donors].to_numpy(), donorcycles)
        assert np.all(donorcut < donorcycles) and donorcycles.max() <= origin
        assert np.all(cycles > origin)
        assert np.allclose(frame.threat_rank.iloc[donors].to_numpy()-donorbase, donorres, rtol=0, atol=1e-12)
        y=frame.threat_rank.iloc[rows].to_numpy()
        missing=frame.signal_stream.iloc[rows].fillna("").str.strip().eq("").to_numpy()
        cells={"full":np.ones(len(rows),bool), "present":~missing, "missing":missing,
               "target_le5":y<=5, "target_ge95":y>=95}
        stats={k:summary(y[m],pred[m]) for k,m in cells.items()}
        cycle_metrics=[]
        for cycle in np.unique(cycles):
            mask=cycles==cycle
            cycle_metrics.append(dict(cycle=int(cycle),**summary(y[mask],pred[mask])))
        manifest_path=source/f"{RUN}/POSITIONAL_TEXT_PILOT_V1/MATRIX_MANIFEST_E{origin}.json"
        manifest=json.loads(manifest_path.read_text())
        width=manifest["matrices"]["fit_base"]["shape"][1]
        fitrows=np.flatnonzero(frame.cycle_code.to_numpy()<=origin)
        identity_path=source/f"{RUN}/POSITIONAL_TEXT_PILOT_V1/E{origin}/predictions_E{origin}.npz"
        with np.load(identity_path,allow_pickle=False) as z:
            # Read identity members only, never Classification probabilities.
            assert np.array_equal(z["fit_rows"].astype(int),fitrows)
            assert np.array_equal(z["score_rows"].astype(int),rows)
            assert np.array_equal(z["score_record_ids"].astype(str),ids)
            assert np.array_equal(z["score_cycles"].astype(int),cycles)
            assert z["matrix_manifest_sha256"].astype(str)[0]==sha(manifest_path)
        matrix_audit={}
        for split in ("fit","score"):
            identityrows=fitrows if split=="fit" else rows
            for arm in ("position","shuffled_position"):
                item=manifest["matrices"][f"{split}_{arm}"]
                matrix_path=source/item["path"]
                assert sha(matrix_path)==item["sha256"]
                matrix=sparse.load_npz(matrix_path)[:,width:].tocsr()
                assert matrix.shape[0]==len(identityrows) and matrix.shape[1]%3==0
                assert np.isfinite(matrix.data).all()
                m=frame.signal_stream.iloc[identityrows].fillna("").str.strip().eq("").to_numpy()
                assert matrix[m].nnz==0
                matrix_audit[f"{split}_{arm}"]=dict(shape=list(matrix.shape),nnz=int(matrix.nnz),
                    csr_bytes=int(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes),
                    sha256=item["sha256"],path=item["path"])
        replay=source/f"{RUN}/RNBR_IMPACT_MATCH_V2_1/text_space_replays/E{origin}"
        receipt=json.loads((replay/"receipt.json").read_text())
        assert sha(replay/"vectorizer.joblib")==receipt["vectorizer_sha256"]
        assert receipt["fit_population"]["cycle_bounds"]==[47,origin]
        result["origins"][str(origin)]=dict(query_rows=len(rows), query_stream_present=int((~missing).sum()),
            prefix_rows=len(fitrows), oof_donor_rows=len(donors),donor_cycles=[int(donorcycles.min()),int(donorcycles.max())],
            oof_rows_strictly_past_cutoff_verified=True, reference_R06=stats,per_cycle=cycle_metrics,
            baseline_bundle_path=name,baseline_bundle_sha256=sha(path), matrices=matrix_audit,
            positional_manifest_sha256=sha(manifest_path),
            positional_identity_bundle_sha256=sha(identity_path),
            standalone_vectorizer_path=(replay/"vectorizer.joblib").relative_to(source).as_posix(),
            standalone_vectorizer_sha256=receipt["vectorizer_sha256"],
            donor_signed_residual_quantiles=np.quantile(donorres,[0,.1,.5,.9,1]).tolist())
        ax.bar([v["cycle"] for v in cycle_metrics],[v["mean_y_minus_prediction"] for v in cycle_metrics],color="#2563eb")
        ax.axhline(0,color="gray",linewidth=.8)
        ax.set(title=f"R06, origin {origin}",xlabel="cycle_code",ylabel="Mean residual (target - prediction)")
    result["source_script_sha256"]=sha(Path(__file__))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"READINESS_AND_ERROR_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
    fig.savefig(OUT/"r06_cycle_residuals.png",dpi=160);plt.close(fig)
    print(json.dumps({o:{k:v for k,v in d.items() if k in ("prefix_rows","query_rows","query_stream_present","oof_donor_rows","reference_R06")} for o,d in result["origins"].items()},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-root",type=Path,required=True)
    main(parser.parse_args().source_root.resolve())
