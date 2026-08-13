# IFDDRP Continuous Downside-Risk Verdict

Evidence class: historical held-out but adaptive. The outcome is maximum origin-to-path loss, not peak-to-trough drawdown.

Validation selected `continuous_downside__transformer_encoder__no_id_temporal`. On the historical test its equal-asset MAE was `0.037816` (95% date-block interval `[0.035374, 0.040008]`) versus `0.020896` for `continuous_downside__training_asset_mean`, a relative reduction of `-80.98%`. Test Spearman was `0.2556` with interval `[0.1805, 0.3271]`.

- mae_reduction: **fail**
- spearman: **pass**
- order_sensitivity: **fail**
- seed_improvement: **fail**

Overall promotion gate: **failed**.

Practical monitoring remains descriptive and cannot be interpreted as a trading strategy.