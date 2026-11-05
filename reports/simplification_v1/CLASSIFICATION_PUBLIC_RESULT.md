# Classification NO_POS benchmark — scored

Root independently checked the exact CSV SHA256 and authenticated Kaggle CLI submission history after Luna1's completion. Ref58024217 is COMPLETE, public MacroF1 **0.82951**. Champion ref58001282 remains0.83225; public loss **0.00274**. Historical A02 ref57909983 scored0.82930, so the new missing-threshold policy gains0.00021 versus that older architecture. Private performance is unknown.

CSV SHA2560884d518072274ad5ae6e35c9b37f275c0b503870bf6ab23772df250f4e40c67 matches the manifest and independent QA. Output has29,421 binary-labelled rows. Worker independent QA reports exact official ID/order, route replay,18 changed rows versus historical A02 and377 versus current champion. No new fits or model inference, and one upload only. Main worktree result commit debdb554e1a515d3e6bc9598d8d1c53664fb7cfc.

Decision: keep the smaller three-predictor A02 architecture as an efficiency fallback, **do not replace the public champion**. Its local retention PASS is unchanged; the tested public result fails to retain the champion's observed score. This is evidence of temporal/public disagreement, not proof of private superiority or a reason to tune the threshold against public scores. Retain the full POS/A02 mixture for the score-chasing path pending other evidence.

Detailed worker receipts: MAIN reports/optimization_v3/cls_simplification_benchmark_v1/{RUN_MANIFEST.json,INDEPENDENT_QA.json,KAGGLE_RESULT.json}. REG NO_POS/NO_CHAR are separate authorized benchmarks and remain in progress; this Classification completion does not cancel them. No final Kaggle selections changed.
