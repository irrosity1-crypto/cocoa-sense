# Data sources

All sources are free and openly licensed. Exact files, retrieval times and SHA-256 hashes are in
`data/raw/manifest.json`; the raw files themselves are committed under `data/raw/`.

| Key | What | Unit | Coverage (as downloaded 2026-11-04) | Role |
|---|---|---|---|---|
| `worldbank_pinksheet` | World Bank Commodity Price Data ("Pink Sheet"), monthly, cocoa column | US$/kg → converted to US$/tonne (× 1,000) | 1960-01 – 2026-10, 802 months | **Target series** |
| `fred_pcocousdm` | IMF Global price of Cocoa, via FRED | US$/tonne | 1992-01 – 2026-09, 417 months | Cross-check only |
| `imf_er_ghs_usd` | IMF Exchange Rates, Ghana, national currency per US$, period average | GHS per US$ (new cedis) | 1990-01 – 2026-01, 433 months | FX experiment (sub-question 2) |
| live rate (D21) | fawazahmed0/exchange-api ("currency-api"), CC0, no key; `https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date or latest}/v1/currencies/usd.min.json` | GHS per US$, daily | sampled from 2026-02, updated on every run | Cedi values in the dashboard only; no model uses it |

## Details and terms

**World Bank Pink Sheet.** Landing page: <https://www.worldbank.org/en/research/commodity-markets>.
File URL in `configs/data.yaml` (the document ID changes with each monthly update; update the URL
and run `cocoasense ingest --refresh` to refresh, which archives the old file). The World Bank
describes the cocoa series as the International Cocoa Organization daily price, average of the
first three positions on the London and New York terminal markets. Licence: CC BY 4.0.

**FRED PCOCOUSDM.** <https://fred.stlouisfed.org/series/PCOCOUSDM>. Source: IMF Primary Commodity
Prices. Used only to confirm the primary series (mean absolute difference 0.43 %).

**IMF Exchange Rates (ER).** SDMX 2.1 API, dataflow `IMF.STA,ER`, key `GHA.XDC_USD.PA_RT.M`.
Access level `PUBLIC_OPEN`. `XDC_USD` means domestic currency per US dollar, so the series is
already `ghs_per_usd`. Values are in new cedis throughout (no 2007 redenomination break).

## Supplied files not used as model inputs

Kept in the author's project folder for the background chapter; see `docs/decisions.md` D10.

| File | Content | Why not a model input |
|---|---|---|
| FAOSTAT Ghana / World cocoa bean exports | Annual tonnes, 1988–2016 | Annual; ends before the test period |
| World Bank WDI exchange rate, GDP growth | Annual, to 2017 | Annual; ends before the test period |
| "Data on variables used in the study" (Table 9) | Annual table from an earlier study | Unidentified source; annual |

## Manual fallback

If a download fails (proxy, firewall, changed URL), save the file with the same name in
`data/raw/manual/` and rerun `cocoasense ingest`. The manifest then records `origin: manual` and
the file's hash.
