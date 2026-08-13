# Authoritative Transformer Specification

This specification is reconstructed from the committed Phase 6 config, code and verified checkpoints.

- Target: `target_stress_10d` over the next 10 observed sessions.
- Lookback: `60` observed sessions, ending at market close on origin day t.
- Inputs: `34` train-scaled features plus a `12`-channel learned asset embedding repeated over time.
- Hidden width: `128`; encoder layers: `2`; heads: `4`; feed-forward width: `256`.
- Position: fixed sinusoidal encoding with maximum length `1024`.
- Pooling: `temporal_attention`; head: LayerNorm, dropout `0.3`, scalar linear logit.
- Parameters: `272,449`.
- Training: `soft_f1` loss, AdamW at `0.0003`, weight decay `0.0001`, batch size `1024`, at most `12` epochs, three deterministic seeds.
- Split: fold `3`, purge `18` global dates, embargo `1` date, train-only per-asset scaling.
- Raw OHLCV fields are not direct model inputs. Technical inputs are engineered at or before t. Seven macro/context inputs are current-vintage/provenance-limited and cannot support strong positive attribution claims.
- There is no explicit family or missingness-indicator input.