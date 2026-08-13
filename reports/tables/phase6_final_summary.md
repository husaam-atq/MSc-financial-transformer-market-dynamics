# Phase 6 Final Summary

## Answer to the central question

The tested pooled Transformer does **not** demonstrate genuine time-varying market-dynamics learning beyond static asset priors. On corrected endpoints, the Transformer achieved ROC-AUC 0.7891, PR-AUC 0.4029 and Brier 0.1109; a training-only static asset prior achieved 0.8239, 0.4531 and 0.1033. Pair-weighted within-asset ROC-AUC was 0.4982. Reversing, deterministically permuting or circularly shifting sequence order left pooled ROC-AUC between 0.7886 and 0.7913.

## Remediation and model evidence

- The pure purge change from 10 to 18 dates left ROC-AUC almost unchanged (0.8009 to 0.8011) but reduced thresholded F1 from 0.5618 to 0.5323.
- Correct observed-session construction restored 20,687 valid labels and increased the represented corrected universe to 79 assets.
- The no-ID model fell to pooled ROC-AUC 0.7159 and within-asset ROC-AUC 0.4783.
- The no-macro model reached pooled ROC-AUC 0.8068 but within-asset ROC-AUC 0.5005 and still lost to the static prior on proper metrics.
- Identity swaps changed probability by 0.0900 on average and reduced ROC-AUC to 0.6829.
- Full-model representations encoded asset identity at 0.1669 accuracy versus 0.0127 chance. A stress probe reached within-asset ROC-AUC 0.5230, showing modest decodability that the trained head did not use reliably.
- The corrected test has 698 contiguous positive-label episodes. Transformer onset recall was 0.5702 with 3,372 false-positive windows and 26.60% alert exposure.

## Target and dynamics

The operational stress target is not economically or statistically comparable across families. Validation prevalence ranged from 1.74% for bonds to 46.80% for crypto. A training-defined asset-relative maximum-loss target was constructed and audited on validation only; it was not scored on the opened test.

Of four existing market-dynamics associations, only momentum-dispersion association intervals excluded zero across all tested block lengths in both training and validation. Equity lead-lag and equity-bond effects did not survive the full dependence sensitivity. All remain observational; asynchronous closes and common shocks prevent causal language.

## Evidence classification

- **Primary falsification:** pooled Transformer ranking is weaker than a static asset prior and does not translate to within-asset ranking.
- **Primary methodological finding:** union-calendar placeholders corrupted eligibility and historical coverage; corrected observed-session processing is now enforced.
- **Robustness evidence:** pure purge retraining, common-date weighting, order destruction, ID removal/swap, no-macro sensitivity and event controls.
- **Secondary exploratory:** modest latent stress decodability and momentum-dispersion association.
- **Negative:** no established temporal-order dependence, no advanced-architecture justification, no independent fresh confirmation.
- **Excluded:** Transformer superiority, universal stress prediction, attention-as-explanation, causal lead-lag discovery and positive claims from alternative-target validation.

All 23 v4 weaknesses were reviewed: 3 are fully resolved, 11 materially mitigated, 5 partially mitigated, 2 unchanged, 1 requires new data as its primary disposition and 1 requires dissertation wording only. Eight rows carry a new-data flag even where Phase 6 partially mitigated the immediate risk. No v4 weakness was classified as worsened. Nine additional weaknesses were found: 1 critical, 4 major and 4 moderate. The largest was union-calendar placeholder contamination of feature/target eligibility.

The official title is now **overstated**. The strongest evidence-aligned alternative is: *Do Pooled Financial Transformers Learn Temporal Market Dynamics? An Adversarial Multi-Asset Study*.

## Stopping decision

Phase 6 meets the scientific stopping rule. Additional architectures would be test-set mining. The next work should be dissertation consolidation and collection of new preregistered outcomes, not more modelling.
