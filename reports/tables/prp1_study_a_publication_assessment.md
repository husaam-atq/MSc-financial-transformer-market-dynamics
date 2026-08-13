# PRP-1 Study A External Replication Assessment

- Evidence class: external geographic/market/calendar replication using the same provider family; status `partial_replication`.
- Eligible assets: 21 of 21.
- Test pooled Transformer ROC-AUC: 0.5431.
- Test static asset-prior ROC-AUC: 0.5749.
- Test pair-weighted within-asset Transformer ROC-AUC: 0.5032.
- Validation-selected calibration/threshold: `isotonic`/0.1000; test positive prediction rate=0.7346.
- Raw 0.5-threshold positive prediction rate=1.0000; raw threshold decisions are degenerate and are not positive evidence. Raw scores are retained only for ranking metrics.
- Data-quality rule: 1 asset fell back wholesale to raw close after invalid adjusted values; nonpositive OHLC cells were set missing without filling.

## Frozen Gates

- `pooled_model_auc`: failed.
- `static_asset_prior_auc`: failed.
- `pooled_minus_within`: failed.
- `within_near_chance`: passed.
- `order_invariance`: passed.

Low pooled discrimination is inconclusive rather than evidence of genuine temporal skill. This is not provider-independent replication because Yahoo supplied both historical and external-market OHLCV.

## Independent simulation replication

The independently implemented stylised DGP completed all 1,040 registered runs with no failures. Static-prior AUC increased by 0.4000 between the frozen low- and high-heterogeneity conditions. Core no-signal within-asset AUC was 0.5001; injected strong ordered signal raised it to 0.7840. Strong-signal reversal and deterministic-permutation AUC drops were 0.3810 and 0.1023. All five frozen point-estimate gates passed. Seed-clustered ordinary and Bonferroni simultaneous intervals are reported separately and support the same qualitative mechanisms.

This is code-path replication and a mechanism-sufficiency result. It is not independent market evidence, external confirmation, a phase transition, or evidence that the simulated mechanism has a particular prevalence in financial markets.

## Distinct-provider ECB FX replication

R09 used official ECB close-only reference rates and 28 currencies passing the frozen coverage gate. The original shortcut pattern did not replicate: the conditioned LSTM reached pooled ROC-AUC 0.7598 and pair-weighted within-currency ROC-AUC 0.7660, while the train-only currency prior reached 0.5596. Registered and endpoint-preserving perturbations support ordered-history dependence.

The positive interpretation is limited by a post-hoc target-mechanics audit. Relative-volatility spikes produced 96.36% of positive test labels, and inverse trailing 20-session volatility alone reached within-currency ROC-AUC 0.7603. The LSTM's paired incremental AUC was 0.0057 with interval [-0.0244, 0.0273]. R09 therefore shows that temporal-order sensitivity can coexist with no robust incremental lift over a mechanical endpoint baseline. It is distinct-provider external evidence, but not confirmation of genuine market dynamics and not sufficient for publication readiness.

## Adaptive Bank of Canada direction replication

R10 acquired all 27 registered official Bank of Canada series; 21 passed the outcome-blind coverage gate. The conditioned LSTM reached pooled ROC-AUC 0.5766 and pair-weighted within-currency ROC-AUC 0.5360. The train-only currency prior reached pooled ROC-AUC 0.5739. Only two of five shortcut gates passed, and the composite temporal-skill gate failed.

The mandatory endpoint logistic reached the frozen 1,000-iteration cap in all seeds. Its within-currency AUC of 0.5254 and the LSTM increment of 0.0106 with paired interval [-0.0091, 0.0306] are therefore descriptive sensitivity evidence, not an estimable promotion comparison. Full-window perturbations were mixed and confound order with endpoint-state changes. R10 is completed but comparator-inconclusive. It supplies no positive temporal-skill evidence and does not make Study A publication-ready by itself.
