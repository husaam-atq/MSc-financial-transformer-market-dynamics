# IFDDRP Static-Prior + Dynamic-Residual Verdict

Evidence class: historical held-out but adaptive. The validation period and historical test have been opened in prior phases; this result cannot provide independent confirmation.

The validation-selected residual arm was `static_dynamic__mlp__family_prior_dynamic_residual` with pair-weighted within-asset ROC-AUC `0.5588` and per-asset macro ROC-AUC `0.5796`.
Its strongest matched dynamic comparator was `static_dynamic__flattened_logistic_no_id` at `0.5165`. The paired date-block lift interval was `[-0.0327, 0.1062]`.

- within_auc: **pass**
- within_auc_lower_bound: **fail**
- macro_auc: **pass**
- dynamic_lift: **fail**
- proper_scores: **fail**
- order_sensitivity: **fail**
- shuffled_residual: **fail**
- seed_stability: **pass**
- zero_residual_identity: **pass**

Overall promotion gate: **failed**.

A fixed additive prior cannot itself alter ranking within one asset. Any within-asset difference is attributable to the fitted residual and training interaction. Pooled AUC is not used as evidence of timing skill.