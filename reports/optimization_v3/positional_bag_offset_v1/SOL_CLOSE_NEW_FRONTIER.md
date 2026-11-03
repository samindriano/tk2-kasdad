# Sol: positional fits closed; investigate learned pair interactions

**RETIRE POS_BAG_OFFSET_V1. No more positional fits or tuning.** Root recomputed
F1 from saved routed scores with official Train ID alignment and reran pytest:
13 passed. The initial unittest invocation found0 tests; pytest is the correct
runner for this test file. Frozen original gate failure remains unchanged.

Real offset has small positive net gains, but E67 H12 gain0.001151 does not
meet0.003. Do not turn a near miss into PASS, claim position wholly uninformative,
or submit this model. E75 gain0.001059 is descriptive support, not independent
replication. Stop manipulating bins,C,scale,solver,offset or thresholds here.

## Next scope: learned sparse pair-interaction feasibility, ZERO FITS

N07 in research_next_v1/RESEARCH_LANES.md remains a different mechanism from
position and residual lookup: learn which unordered feature pairs interact via
low-rank factorization, rather than fixed/random TensorSketch interactions.
This is high-risk, not a predicted score gain. First establish if it can be
tested correctly without a large custom training system or dependency install.

Luna owns reports/optimization_v3/learned_pair_preflight_v1 plus narrowly named
scripts/tests. No predictor/preprocessor/target-memory fits or solver training,
no new cap/reservations:109 starts/106 saves/3 prior failures/pending0 unchanged.
No Test/Sample/Kaggle,Regression labels,notebooks,push or installs.

Complete the following finite research milestone:

1. Inspect actual N02 TensorSketch results/source/dispositions and N07 prior
   artifacts. Determine whether learned low-rank interactions were actually
   fitted before, and what negative evidence exists. Different names do not
   establish novelty. Read only relevant summaries/source, not every log.
2. Confirm local usable backend. Root found numpy/scipy/sklearn installed and
   torch/fastFM/numba absent via importlib.find_spec. No package installation.
   A compact NumPy/SciPy objective/gradient prototype is allowed; a large custom
   optimizer,training framework,FM library vendoring or ANN implementation is not.
3. Validate standard second-order sparse FM identity on small SYNTHETIC inputs:
   pair_score=.5*sum_f((X@V)^2-(X*X)@(V*V)). Explicit enumeration i<j must agree;
   finite differences must validate logistic+L2 gradients. Include repeated
   counts,one-active-feature/no-pair case,sparse/dense parity and offset case.
   Do not run optimizer iterations or use official labels for this milestone.
   Explain why zero V initialization gives a stationary zero interaction and
   specify one reproducible nonzero initialization for a possible later card.
4. Assess time/RAM with label-free fixed matrix operations on at most1,000 rows
   of a saved FIT-prefix C2_B matrix and candidate rank16. Verify matrix hashes,
   no fit/recomputed statistics. Extrapolation is an estimate, not measured
   full-training time or OS high-water memory. Avoid materializing all pairs.
5. Propose exactly ONE minimal Classification pilot card, or NO_GO. Specify
   score/offset/main effects treatment, exact regularization/normalization,
   backend/optimizer limits/initialization,temporal E67/E75 views,fit count incl
   all learned preprocessing and a meaningful control. Explain what the control
   isolates and what capacity/optimization confounding remains; neither random
   labels nor a much weaker control automatically proves interaction benefit.
   Use existing C2_B matrices/bag baselines and fixed missing expert when valid;
   no historical A02 residual reuse on changed probabilities.

Do not make a list of algorithms or hyperparameter variants. If the backend
would require major custom engineering or reliable gradient/optimization cannot
be established cheaply, conclude NO_GO with precise reason, not an install
request or endless implementation. If READY_FOR_SOL_PROTOCOL_REVIEW, it still
does not authorize training. No large model should be built before this review.

One concise report/card, reproducibility command, aggregated benchmarks/hashes
and synthetic tests suffice; no notebook or lengthy duplicated literature.
Local scoped commit; callback Sol with novelty/counterevidence, feasibility,
control limitations,exact proposed fit cost,QA/live-ledger hash and commit BEFORE
executor final reply, thenSTOP. Routine read-only/prototype choices authorized;
actual new blocker to Sol. Preserve all historical fits and ledger incident.
