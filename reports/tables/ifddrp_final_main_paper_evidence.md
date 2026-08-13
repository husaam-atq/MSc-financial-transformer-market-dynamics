# IFDDRP Final Main-Paper Evidence

## Contribution 1: corrected Transformer pipeline

Report the exact observed-session pipeline, 60-session lookback, ten-session stress target, 34 inputs plus asset embedding, corrected purge/embargo, three-seed training and raw/calibrated metric distinction. The pooled ROC-AUC establishes why the model initially looked successful; it is not a superiority claim.

## Contribution 2: pooled performance versus genuine timing

Make the static asset/family priors, pair-weighted within-asset ROC-AUC, no-ID ablation and three temporal-order controls the central result. The clean claim is that the tested pooled metric mainly measured cross-sectional structure and did not establish useful within-asset chronology.

## Contribution 3: mechanism and cross-model recurrence

Use the registered simulation to demonstrate that heterogeneous label priors are sufficient to inflate pooled AUC and that within-asset/order-sensitive metrics recover planted dynamic signal. Pair it with the bounded five-model panel, where no strict temporal-skill gate passed. This is stronger than a Transformer-only negative.

Recommended compact presentation: one pipeline/target diagram, one pooled-versus-within/static table, one order-control figure, and one simulation mechanism figure. Experiments A-C belong in a compact robustness table.
