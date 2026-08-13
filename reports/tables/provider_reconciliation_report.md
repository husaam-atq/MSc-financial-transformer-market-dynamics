# Provider Reconciliation Report

Yahoo is the training source. Stooq is an independent check only and is not merged into the training panel.

```csv
ticker,stooq_symbol,status,reason

SPY,spy.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/spy.us.csv.

QQQ,qqq.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/qqq.us.csv.

IWM,iwm.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/iwm.us.csv.

DIA,dia.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/dia.us.csv.

VTI,vti.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/vti.us.csv.

VT,vt.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/vt.us.csv.

XLK,xlk.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xlk.us.csv.

XLF,xlf.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xlf.us.csv.

XLV,xlv.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xlv.us.csv.

XLE,xle.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xle.us.csv.

XLY,xly.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xly.us.csv.

XLP,xlp.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xlp.us.csv.

XLI,xli.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xli.us.csv.

XLB,xlb.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xlb.us.csv.

XLU,xlu.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xlu.us.csv.

XLRE,xlre.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xlre.us.csv.

XLC,xlc.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/xlc.us.csv.

MTUM,mtum.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/mtum.us.csv.

VLUE,vlue.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/vlue.us.csv.

QUAL,qual.us,failed,Automated Stooq access failed (RemoteDataError); direct CSV access is unavailable or challenged (Stooq direct endpoint did not provide a CSV (HTTP 404; content type text/html).). Download the daily CSV manually from Stooq and place it at data/external/stooq_manual/qual.us.csv.
```

## Binance Vision and CCXT

CCXT is an independent recent spot-API check. Binance Vision remains the historical training source; duplicate candles are not merged.

- Successful pairs: 20 / 20
- Maximum absolute close difference: 0.0
- Maximum mean absolute volume difference: 0.0

```csv
symbol,status,overlap_rows,overlap_start,overlap_end,max_abs_close_difference,mean_abs_close_difference,mean_abs_volume_difference,retrieval_timestamp_utc
BTC/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:22:56.249844+00:00
ETH/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:22:58.855774+00:00
BNB/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:02.537860+00:00
XRP/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:05.298821+00:00
ADA/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:07.898703+00:00
SOL/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:10.462984+00:00
DOGE/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:13.070439+00:00
TRX/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:15.710031+00:00
LTC/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:18.342415+00:00
BCH/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:20.992065+00:00
LINK/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:23.513688+00:00
AVAX/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:26.133639+00:00
DOT/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:29.153095+00:00
POL/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:31.776162+00:00
ATOM/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:34.588286+00:00
UNI/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:37.188107+00:00
XLM/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:39.761411+00:00
ETC/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:42.390680+00:00
FIL/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:44.982914+00:00
AAVE/USDT,success,192,2026-05-24,2026-05-31 23:00:00,0.0,0.0,0.0,2026-06-23T17:23:47.549846+00:00
```
