# IFDDRP Final Evidence Freeze

Frozen on 2026-08-08 from evidence-producing source commit `e66470a6de35019cf8239eb4f8e6b8aab38e0bbb`. This freeze supersedes reporting ambiguities but does not overwrite historical files.

## Three principal contributions

1. **A real corrected Transformer forecasting pipeline.** A 272,449-parameter Transformer used 60 observed sessions, 34 features and a 12-channel asset embedding to forecast ten-session stress across a broad daily panel. Its raw pooled ROC-AUC was `0.789814`.
2. **An adversarial shortcut diagnosis.** The training-only asset prior (`0.823906`) and family prior (`0.816549`) exceeded the Transformer; the Transformer's within-asset AUC was `0.491638`; identity was decodable; removing identity reduced pooled ranking; and registered order destruction barely changed pooled ranking.
3. **Mechanism and recurrence evidence.** A 1,040-run controlled simulation showed how target-prior heterogeneity can inflate pooled AUC while within-asset evaluation remains at chance in no-signal settings and recovers planted dynamics. Across logistic, MLP, LSTM, TCN and Transformer models, no strict empirical temporal-skill gate passed.

## Evidence classes

- Main paper: the three contributions above.
- Major supporting: identity ablation/swap/probes and bounded Experiments A-C as valid negatives.
- One-line robustness: zero of 14 regimes passed; latent-state analysis remained locked.
- Viva/supplement: exact model specification, target/purge history, endpoint manifests, feature and lag diagnostics, external checks and protocol deviations.
- Excluded/superseded: Transformer superiority, literal discovery of predictive emergent dynamics, attention as explanation, causal macro attribution, trading/monitoring claims, non-selected test winners and calibrated-score AUC as ranking authority.

## Scientific stopping rule

No historical model fairly beat the static asset prior, no model passed the chronology gate, and none of the three final bounded recovery experiments passed. Additional historical model search would increase adaptivity without supplying independent evidence. The empirical modelling programme is frozen; the next action is dissertation writing from this registry, followed only by independently preregistered replication if the research continues.
