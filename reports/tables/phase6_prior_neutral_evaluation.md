# Phase 6 Prior-Neutral Evaluation

No single adjustment is treated as definitive. The analysis triangulates exact static priors, within-asset pair weighting, per-asset macro averaging, equal asset/date/family weighting, common-date support, centred scores, residual scores and label-episode detection.

The corrected Transformer has pooled ROC-AUC 0.7891 but pair-weighted within-asset ROC-AUC 0.4982. Equal-asset and common-date ROC-AUCs remain high (0.7708 and 0.7698) because both retain between-asset ranking. They must not be described as prior-neutral temporal skill. Per-asset macro ROC-AUC is 0.5466 across 74 assets, but its unweighted treatment of sparse groups makes it secondary.

Static asset prevalence is stronger than the Transformer: ROC-AUC 0.8239, PR-AUC 0.4531 and Brier 0.1033. Residual transformations do not establish incremental skill and are not proper probabilities. The fairest central answer is therefore the pair-weighted within-asset result alongside the exact static-prior comparison.

There is no defensible positive incremental lift beyond static priors on the opened historical test. Any renewed superiority claim requires a preregistered comparable target and genuinely new outcomes.
