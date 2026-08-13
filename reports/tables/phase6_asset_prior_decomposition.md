# Phase 6 Asset-Prior Decomposition

## Exact-endpoint comparison

On the corrected 21,514-row test set, the Transformer scored ROC-AUC 0.7891, PR-AUC 0.4029 and Brier 0.1109. A training-only static asset-prevalence score achieved 0.8239, 0.4531 and 0.1033. The family prior also exceeded the Transformer on all three proper metrics (0.8165, 0.4149 and 0.1049).

The Transformer therefore adds no demonstrated aggregate lift beyond the static asset prior. Its deltas are -0.0348 ROC-AUC, -0.0502 PR-AUC and +0.0077 Brier loss.

## Prior-neutral views

Within-asset centring and residual scores are diagnostics, not calibrated probabilities. On the corrected endpoints, the Transformer pair-weighted within-asset ROC-AUC was 0.4982 across 74 eligible assets. Its per-asset macro ROC-AUC was 0.5466, but macro averaging gives tiny and sparse assets the same weight and does not contradict the pair-weighted chance result. Residual score ROC-AUCs were 0.4349 after asset-prior subtraction and 0.4926 after family-prior subtraction.

No-asset-ID training reduced pooled ROC-AUC to 0.7159 and pair-weighted within-asset ROC-AUC to 0.4783. This demonstrates dependence on identity, not hidden asset-invariant forecasting skill.

## Interpretation

The central pooled result is primarily a between-asset prevalence-ranking result. Some representations contain decodable state information, but the trained classification head does not convert it into reliable within-asset ordering. This is a primary falsification and methodological contribution, not a positive forecasting result.
