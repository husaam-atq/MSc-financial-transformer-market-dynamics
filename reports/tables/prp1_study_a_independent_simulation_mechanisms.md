# PRP-1 Study A Independent Simulation Mechanisms

Completed 1040 registered runs; failures: 0.

## Frozen estimates

- `static_prior_auc_increase`: 0.4000.
- `no_signal_within_auc`: 0.5001.
- `strong_signal_within_auc`: 0.7840.
- `strong_signal_reversal_drop`: 0.3810.
- `strong_signal_permutation_drop`: 0.1023.

## Frozen gates

- `static_prior_inflation`: passed.
- `no_signal_within_near_chance`: passed.
- `strong_signal_within_recovery`: passed.
- `strong_signal_reversal_sensitivity`: passed.
- `strong_signal_permutation_sensitivity`: passed.

## Registered-gate scope sensitivity

The coded primary strong-signal gate uses persistence 0.7. The frozen prose did not state that qualifier, so the all-persistence estimates are reported as a non-selective sensitivity:

- `strong_signal_all_persistence_within_auc`: 0.7605.
- `strong_signal_all_persistence_reversal_drop`: 0.3352.
- `strong_signal_all_persistence_permutation_drop`: 0.1233.

The preregistered gates above use the frozen point-estimate rules. A separate seed-clustered table reports ordinary 95% and post-hoc Bonferroni simultaneous intervals; those intervals strengthen inference but do not retroactively redefine the gates.

This independent stylised DGP establishes mechanism sufficiency, not prevalence in real markets or a phase transition. Robustness anchors cannot rescue failed core gates.
