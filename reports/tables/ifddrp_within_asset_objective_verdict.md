# IFDDRP Within-Asset Objective Verdict

Evidence class: historical held-out but adaptive.

The frozen pair registries contain `323,584` training pairs and `230,037` validation pairs. These are weighting support, not independent observations.

The validation-selected aligned objective was `within_objective__transformer_encoder__bce_plus_within_asset_pairwise` with within-asset ROC-AUC `0.6012` versus `0.5745` for its matched pooled-BCE control.
The paired date-block lift interval over the matched control was `[-0.0143, 0.0708]`.

- within_auc: **pass**
- within_auc_lower_bound: **pass**
- lift_over_pooled_bce: **fail**
- order_sensitivity: **fail**
- seed_stability: **pass**

Overall promotion gate: **failed**.