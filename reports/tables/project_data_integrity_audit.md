# Data Integrity Audit

## Verified Scale And Structure

- Historical processed daily panel: 358,160 rows.
- Rows with observed close: 306,174.
- Calendar-alignment placeholders with missing close: 51,986.
- Duplicate asset-date rows found in the checked panel: zero.
- Missing-close rows carrying a non-null stress target: zero.

The model-ready target rows are not contaminated by the placeholders. However, the final
inclusion manifest counted placeholders and used their dates, producing incorrect row counts
and artificial 2010 start dates for later-listed instruments. Filtering to observed closes
fixes the metadata. All 80 assets still exceed the inclusion threshold, so the historical
universe and metrics do not change.

## Material Data Risks

1. **Current-vintage macro data.** FRED observations can be revised. A uniform one-business-
   day shift does not encode each series' release calendar or vintage available at time t.
2. **Asynchronous closes.** US, international and crypto observations aligned by calendar
   date do not share one information cutoff. Lead-lag results are especially exposed.
3. **Survivorship and availability.** The current instrument universe excludes historical
   delistings and is not a historical constituent set.
4. **Provider dependence.** Adjusted closes and corporate-action histories can be revised by
   the provider. Raw OHLC-based intraday proxies and adjusted-close returns measure different
   price concepts.
5. **Crypto daily aggregation.** UTC-based daily bars are not economically synchronized with
   exchange-traded asset closes.

## Findings Not Established

No evidence was found that ETF weekends were forward-filled into target-bearing rows, that
duplicate target rows drove results, or that placeholder missingness itself created labels.
No provider-vintage audit can prove historical point-in-time macro availability from the
current cache.

## Required Claims Boundary

Call the panel provider-reproducible and timestamp-disciplined at the row level, not fully
point-in-time vintage safe. Limit generalisation to the selected surviving instruments.
Treat cross-market lead-lag timing as exploratory unless exchange-close alignment is added.
