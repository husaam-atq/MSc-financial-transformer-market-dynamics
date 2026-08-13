# Phase 6 Target Comparability And Data-Path Audit

## Deterministic Repair

- Observed OHLC rows: 306,174.
- Frozen stress labels: 284,687.
- Correct observed-session stress labels: 305,374.
- Restored valid labels: 20,687.
- Label disagreements on shared non-missing endpoints: 0.
- Corrected split: purge 18; train through 2023-11-24, validation through 2025-02-23, test through 2026-06-13.
- Historical processed data and freezes were not overwritten.

## Original Operational Target

- Validation family prevalence ranges from 0.0174 to 0.4680.
- The target remains heterogeneous in rarity and severity and combines adverse returns, minimum future price and volatility spikes.
- Contiguous label runs are reported as label episodes, not independent economic events.

## Fixed Training-Defined Alternative

- The only alternative audited is the per-asset training 90th percentile of maximum loss over ten observed sessions.
- Validation family prevalence ranges from 0.0404 to 0.0901.
- It standardises within-asset rarity, not economic loss. Historical test and fresh labels were not scored.

## Macro Availability

- Maximum training missingness across macro columns is 0.9693.
- Current-vintage and release-timing limitations remain. The no-macro model is a sensitivity analysis, not a repaired macro claim.
