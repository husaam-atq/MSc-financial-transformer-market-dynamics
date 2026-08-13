# Final Transformer Interpretability Verdict

Evidence class: post-hoc robustness on an opened historical test.

The largest mean zero-occlusion response was `macro_context` (0.1607 mean absolute probability change). The largest registered lag response was `distant_days_41_60` (0.1413). These are sensitivity diagnostics, not causal importance.

Prediction reconstruction passed for all checkpoints. `4` sanity/control rows failed their desired dynamic-explanation condition. Historical order controls remain decisive: sequence destruction barely changed pooled ranking.

Asset identity remains the strongest scientific interpretation. The model encodes state-sensitive inputs, but the trained head did not establish useful within-asset timing. Macro/context sensitivity is provenance-limited and is not promoted.

Attention pooling weights are retained only as supplementary diagnostics. They are not treated as explanations.