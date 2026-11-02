# Sol decision: six-fit R-NBR continuation, no final refit or submission

**Scope:** six new target-memory fits at origins59/63/71, observed and matched
shuffle. Global cap70->76, continuation31->37; pilot38, control_replay0,
technical_retry1 unchanged. This is a prospective incremental diagnostic, not
automatic qualification or promotion. Record a new append-only amendment.

Authorization: Sam explicitly approved execution and delegated further measured
decisions to Sol with "jangan nunggu apa2 lagi. setuju semua", following the
request to keep exploring while the account allowance remains. Sol selects this
bounded tranche; there is no unlimited fit allowance or reset of historical use.

## Evidence reviewed directly

Sol read the pilot result and independent verification, matched all four bundle
NPZ hashes against the verified report, joined official Regression Train targets
by saved row/cycle, checked missing equality, and independently recomputed:

| View | R06 MAE | R-NBR MAE | Shuffled MAE |
|---|---:|---:|---:|
| E67 W4 + E75 W6, n13,441 | 10.464680 | 10.392689 | 10.481887 |
| E67 H12-L, n20,903 | 11.529709 | 11.577980 | 11.582025 |

The gain is not stable: E67 W4 harms0.112556, E75 W6 improves0.179576.
E67 on the same W6 outcomes harms0.017145. Changing history/model origin thus
matters, but these comparisons do not isolate a causal effect of history length.
The H12 harm0.048271 only narrowly passes the fixed0.05 tolerance; that is a
material concern, not evidence of long-horizon benefit. Keep all negative results.

Six additional paired fits complete the missing primary origins and distinguish
a broad effect from one favorable origin. This is more informative than changing
scale/cap after the pilot. No pilot fit is repeated and no recipe is changed.

## Fixed recipe and views

Same R06 final mapped baseline, saved target-free top32 word-TFIDF geometry,
strictly past R06 OOF residual donors, cosine-squared weighted median,
delta=clip(.25*median,-5,+5), finalclip[0,100], no rerank. Missing/<5neighbor
fallback exactR06. Same fixed within-cycle/prediction-decile shuffle seed42.

New banks: E59 cycles56-59, E63 cycles56-63, E71 cycles56-71. Donor OOF cutoff
must precede its cycle; bank ends at query origin. Query baseline always uses
that origin's R06 forecast, never a per-cycle donor-source mapping.
New queries: E59->60-63 n6493; E63->64-67 n7078; E71->72-75 n7462.
Their pooled new-origin denominator is21,033, not cycles68-79/n20,903.

Full primary38,588 = exactR06 W1 fallback + learned origins59/63/67/71/75.
Late W5+W6 n15,953. H12-L remains exact inherited E67 forecast68-79 n20,903,
not a composite using E71/E75. It is already-seen evidence and must not change.

## Prospectively fixed continuation checks

- Full-primary MAE gain versusR06>=0.03 and versusmatchedshuffle>=0.01.
- New-origin aggregate gain versusR06>=0 and versusshuffle>=0.
- At least3/5 learned primary windows nonnegative gain versusR06; W1 fallback
  is disclosed, not counted as an independent learned win.
- Late W5+W6 MAE gain versusR06>0.
- Inherited E67 H12-L harm versusR06<=0.05, unchanged from pilot.
- Each learned primary window harm versusR06<0.25.
- Missing and actual-target tails<=5/>=95, n>=100, harm<=0.25 versusR06 across
  declared windows/aggregates/H12; missing incremental change exactly zero.
- All integrity, no-future donor, source/provenance/view/geometry checks PASS.

Report verified R22/legacy contrasts and original R06 guard failures separately.
Passing an incremental check does not clear inherited R06 legacy risk. Unsupported
groups remain explicit. No gate relaxation, new reranking, scale or cap rescue.

## Recovery, accounting and stop

Use safe Unicode identifiers; never allow_pickle to read an object-ID bundle.
Preflight actual payload-builder/readback/resume before each source version.
Retain the recovered E67 pilot and original failed receipt. The ledger's three
failure events include one recovered output: starts70, FIT_SAVED events67,
one recovery, two unrelated unresolved failures. Count fit IDs and valid outputs,
not a simplistic starts-saves difference. No refund, duplicate reservation or
unapproved retry. Reserve six unique continuation IDs before construction.

No new text/neighbor geometry fitting, base-expert refits, other-task targets,
Test/Sample/Kaggle access, install, notebook edits or push. Use fresh continuation
wrappers rather than editing frozen pilot sources. One job, two threads,
6GiB ceiling and2GiB free reserve. New outputs durable before reports.

After six fits and independent evaluation, commit only scoped files, send exact
result/counterevidence/ledger/commit to Sol, then yield for the next decision.
This tranche includes no origin79 final fit or submission. Sam need not be woken
for routine decisions; genuinely new execution scope comes from Sol explicitly.
