# Decision log

Each entry: date, options considered, choice, reason. Newest last.

## D1 · 2026-11-04 · Primary price series
- **Options:** (a) IMF via FRED `PCOCOUSDM` (Jan 1992 –, as in the scope document); (b) World
  Bank Pink Sheet monthly cocoa (Jan 1960 –), supplied by the author and freely downloadable.
- **Choice:** (b) World Bank Pink Sheet as the target series; FRED as an independent cross-check.
- **Reason:** About twice the history (802 vs 417 months), already provided by the author, and the
  two agree closely (417 overlapping months, mean absolute difference 0.43 %, correlation 0.9998;
  `artifacts/data_quality.md`). The official file downloaded on 2026-11-04 ("Updated on November
  03, 2026") was used. Its cocoa column is identical to the author's October copy for every
  common month. The "Mismatch Details" sheet inside the workbook is part of the World Bank's own
  file and does not affect cocoa.

## D2 · 2026-11-04 · Training cut-off and frozen-origin evaluation
- **Requirement (author):** models must not be trained on data after 2020 so that predictions can
  be compared with actual prices as a percentage error.
- **Options:** (a) cut-off December 2019, test from January 2020; (b) cut-off December 2020, test
  from January 2021; and, for the test period, (i) expanding-window refits, (ii) parameters frozen
  at the cut-off.
- **Choice:** (a) + (ii). Final models are fitted once on targets up to 2019-12 and frozen. Each
  test month is forecast from the real history up to its decision month without refitting.
- **Reason:** (a) matches the scope document's split (train ≤ 2015, validation 2016–2019, test
  2020 →), satisfies "not trained on data above 2020", and keeps COVID-2020 out-of-sample.
  (ii) is the only reading that strictly satisfies the requirement; refitting (i) would train on
  test data. Frozen parameters may lose accuracy as time passes; that is reported, not hidden.

## D3 · 2026-11-04 · Test sub-periods
- **Choice:** by target month: pre-surge 2020-01 – 2022-12, surge 2023-01 – 2025-01 (to the
  January 2025 peak of US$10,750/t), post-peak 2025-02 onward.
- **Reason:** The scope asks for results with and without the 2023 surge. The data now also
  include the 2025–26 fall, so a third period separates rising and falling regimes. Boundaries are
  set from the public price path, not from any model result.

## D4 · 2026-11-04 · Exchange-rate source
- **Options:** Statistics Ghana StatsBank (manual export), Bank of Ghana (daily), IMF Exchange
  Rates dataset via the open SDMX API, World Bank WDI (annual only, ends 2017 in the supplied CSV).
- **Choice:** IMF ER dataset, `GHA.XDC_USD.PA_RT.M` (cedis per US dollar, monthly average).
- **Reason:** Free, machine-readable, reproducible by URL, monthly. Quote direction is explicit
  (`XDC_USD` = domestic currency per USD). The series is expressed in new cedis throughout: no
  break around the July 2007 redenomination (May–Sep 2007 values 0.926–0.935), checked
  automatically in `validation/checks.py`. Coverage 1990-01 – 2026-01.

## D5 · 2026-11-04 · Exchange rate in the main models
- **Choice:** Main frozen models use price-only features. Sub-question 2 is answered by a paired
  experiment (Random Forest and XGBoost, with and without FX features, same training start
  1991-01, same months) on both validation and test periods.
- **Reason:** The FX series ends January 2026, so FX-dependent models could not forecast the last
  months of the test period. Keeping main models price-only gives every model the same test
  months; the paired design isolates the effect of FX.

## D6 · 2026-11-04 · Environment and repository location
- **Choice:** Python 3.12 virtual environment managed by `uv` (lockfile `uv.lock`); repository at
  `C:\Users\helin\cocoa-sense` (outside OneDrive), pushed to GitHub.
- **Reason:** The system Python is 3.14, which some scientific packages do not yet support.
  OneDrive syncing can lock files inside `.git` and `.venv`.

## D7 · 2026-11-04 · Training start month
- **Choice:** For ARIMA, Random Forest and XGBoost, compare training starts 1960-01, 1980-01 and
  1991-01 on the validation period and keep the one with the lowest mean validation MAE.
- **Reason:** Longer history gives more examples but includes very different regimes (1970s
  boom). Choosing on validation avoids an arbitrary choice and never uses test data.

## D8 · 2026-11-04 · Prediction intervals (80 %)
- ARIMA: native Gaussian intervals on log price, transformed to price.
- Random Forest / XGBoost: split-conformal intervals from out-of-fold absolute residuals of
  time-series CV inside the training window (finite-sample rank ⌈(n+1)·0.8⌉).
- Naive: empirical 10th / 90th percentiles of historical h-month log returns (training window).
- **Reason:** Each uses training data only; coverage is checked on validation and test.

## D9 · 2026-11-04 · Simulation choices
- Forecast-driven rule `w = clip(0.5 + a·z, 0, 1)`, `z = (F_t − forecast) / half-width`.
- `a` and the driving model are chosen together on the validation period by the highest mean
  revenue per tonne (ties → smaller `a`), then frozen in `configs/frozen.yaml`.
- Financing cost valued at the decision-date forward price: `C = r · h/12 · (1 − w) · Q · F_t`,
  default `r` = 10 % a year (sensitivity 0–20 %). Full list in `docs/simulation_assumptions.md`.

## D10 · 2026-11-04 · Supplied annual datasets
- FAOSTAT exports (1988–2016), World Bank exchange rate and GDP growth (to 2017) and the
  "Table 9" document are annual and stop before the test period, so they cannot be monthly model
  inputs. They are kept outside the repository for the background chapter only. The "Table 9"
  document is from an unidentified earlier study and is not committed; cite its source before
  using it in the dissertation.

## D11 · 2026-11-04 · ICCO free tables
- **Choice:** not implemented. The free daily table appears to start in October 2024 and adds
  nothing the monthly series lacks. Can be revisited in Phase 8.

## D12 · 2026-11-04 · Data anomalies kept, not edited
- January 1973 monthly log change +0.57 and February 2001 World Bank vs FRED difference of 23 %
  are listed in the data-quality report and left unchanged (raw data is never edited). 1973 is
  before every candidate training start except 1960; February 2001 is inside training data only.

## D13 · 2026-11-04 · Layout deviations from the build prompt
- `ingestion/sources.py` + `validation/parsers.py` replace the per-source `fred.py`,
  `statsbank.py`, `icco_free.py`, `manual.py` and `validation/schemas.py` files: one parser per
  source and one generic obtain/manual-override routine are simpler and fully tested.
- Stress scenarios live in `simulation/engine.py`; tree models share `models/tree.py`.
- `cocoasense train` = development stage (fit to 2015, forecast validation); `freeze` writes the
  frozen configuration; `evaluate --final` runs the test period once.
- A Makefile is provided for CI/Linux; on Windows the same commands run via `uv run`.

## D14 · 2026-11-04 · Forecast-driven rule chosen at the edge of its grid
- **Observation:** on validation, mean revenue of the forecast-driven rule rose monotonically
  with `a` for the XGBoost driver, so the frozen choice is `a = 2.0`, the largest value in the
  grid fixed before training (`configs/simulation.yaml`). The gain over Fixed 50/50 on validation
  is small (`artifacts/metrics/validation_simulation_a_grid.csv`) and the worst-10 % outcome is
  slightly lower.
- **Options:** (a) keep the pre-declared grid; (b) widen the grid after seeing validation results.
- **Choice:** (a).
- **Reason:** widening would push towards an almost all-or-nothing rule tuned on only 48
  validation months, i.e. overfitting the validation period. The test-period sensitivity table
  reports results for every `a` in the grid, clearly labelled as not used for selection.

## D15 · 2026-11-04 · Random Forest single-threaded for reproducibility
- **Observation:** leakage test 4 failed once in a full test run and passed on rerun. Cause:
  with `n_jobs=-1`, scikit-learn sums per-tree predictions in thread-dependent order, so identical
  refits differ at ~1e-16 (verified: 6.7e-16 max difference across six refits; 0.0 with
  `n_jobs=1`).
- **Choice:** `RandomForestRegressor(n_jobs=1)`; XGBoost was already single-threaded. The
  development stage and freeze were rerun after this change, before any test-period run.
- **Reason:** the build requirement is bit-for-bit identical metrics on rerun, and leakage test 4
  compares fitted settings exactly.

## D16 · 2026-11-05 · LSTM stretch model added after the main final run
- **Context:** the author asked for the LSTM stretch goal after the four main models had been
  frozen and evaluated on the test period (`artifacts/final_runs.log`, first entry).
- **Options:** (a) add the LSTM to `frozen.yaml` and rerun the main final run; (b) give the LSTM
  its own development → freeze → final cycle and leave the original results untouched.
- **Choice:** (b). `configs/lstm.yaml` (architecture, grid, training settings) and the code were
  committed before the LSTM was run on any real data. The development stage (`cocoasense lstm
  train`) fits on targets to 2015-12 and forecasts 2016–2019; `cocoasense lstm freeze` picks the
  training start from validation MAE only and writes `configs/frozen_lstm.yaml`, which is
  committed before the one logged `cocoasense lstm evaluate --final` run. The final LSTM is fitted
  on targets up to 2019-12 and never refitted. `test_metrics.csv` and the simulation's frozen rule
  are unchanged; the LSTM appears in `test_metrics_with_lstm.csv` and, as a driver, only in the
  reported-only sensitivity table.
- **Design (fixed in advance):** same task as the tree models (h-month log return, split-conformal
  80 % intervals from out-of-fold residuals, `TimeSeriesSplit` with gap = h inside the training
  rows). One LSTM layer with a linear head; grid over input set {returns only, returns + log
  price}, lookback {12, 24} months and hidden size {8, 16}; 200 full-batch Adam epochs, learning
  rate 0.01, weight decay 1e-4, no early stopping; 3 seeded networks averaged; single-threaded,
  so refits are bit-for-bit identical (leakage test 4 includes the LSTM).
- **Caveat (stated in limitations):** unlike the original four, the LSTM was designed by someone
  who had already seen the other models' test results, so its test is not fully blind. To limit
  this, both input sets (with and without the price level) are in the grid and the choice is left
  to cross-validation on training data, and the LSTM is not changed after its test run.

## D17 · 2026-11-05 · Cedi revenue view
- **Context:** the author asked for the cedi revenue view stretch item. COCOBOD sells cocoa in US
  dollars but its costs and producer prices are set in cedis, so revenue in cedis is the figure
  that matters locally.
- **Options:** (a) convert at the decision-month rate; (b) convert at the delivery-month rate;
  (c) build a cedi-aware selling rule.
- **Choice:** (b), as a reporting view only. Both the forward and the spot share are paid in US$
  at delivery, so the delivery month's IMF average GHS/US$ rate applies. Deliveries after the last
  observed rate (Jan 2026) are dropped, never extrapolated. Nothing is re-simulated and no frozen
  setting changes; (c) would need its own validation → freeze cycle and was not asked for.
- **Consequence:** all strategies on a date share one rate, so the per-date winner and the win
  rates are the same as in US$; only the weighting across dates changes (weak-cedi years count more).

## D18 · 2026-11-05 · Futures-curve data: not available under open terms; forward price clarified
- **Context:** the author asked for futures-curve data to replace the forward-price approximation.
- **Search:** ICE's end-of-day settlement history is sold by subscription; Barchart's per-contract
  downloads need a login and its terms forbid redistribution; Databento sells ICE US data (from
  Dec 2018) on paid plans; Investing.com, Yahoo and Stooq offer only a continuous front-month
  series under restrictive terms; TradingCharts has charts only; ICCO publishes one averaged daily
  price. Nothing openly licensed gives each contract month for 2019–2026.
- **Finding:** the World Bank Pink Sheet (`Description` sheet of the raw file) defines the cocoa
  price as the ICCO daily price, "average of the first three positions on the terminal markets of
  New York and London, nearest three future trading months". The `spot_proxy` forward price is
  therefore already a near-futures average, not a spot price; earlier docs called it spot.
- **Choice:** no new data and no change to any result or frozen setting. Docs, figure titles and
  the dashboard now describe the forward price correctly. The mode keeps its `spot_proxy` name
  because it is recorded in frozen artifacts. `CsvFuturesProvider` stays ready for licensed
  contract data (`forward_price_mode: futures`); such a run would be a separately labelled
  sensitivity, since the frozen rule was chosen with the current forward price.

## D19 · 2026-11-05 · Insights: scorecards, ex-ante confidence, learning track, live forecasts
- **Context:** the author asked for success/failure rates, a probability or confidence for each
  predicted price, plain-language explanations, a comparison of predictions with the latest market
  prices, models that keep learning from new prices, and more visualisation.
- **Conflict with D2:** "keep learning" contradicts "never train on data after 2019". Both are
  kept, as separate, labelled tracks. The *frozen* track (D2) is unchanged and remains the
  held-out test. The *learning* track refits every model at every decision month *t* on
  `history[:t]`; leakage test 6 checks that a forecast made at *t* is unchanged by any data after
  *t*. Hyperparameters, training starts and features stay at the frozen choice (tuned on pre-2020
  data), so nothing is tuned on the test period; only fitted parameters and conformal interval
  widths are re-estimated. The LSTM is not refitted monthly (too slow); it stays frozen-only.
  Learning-track forecasts start at target 2018-01 so that 2018–2019 errors can seed the
  confidence estimates; scoring starts at 2020-01 like the frozen track.
- **Success criteria (fixed in `configs/insights.yaml` before any results):** within ±5 %, within
  ±10 %, direction correct (forecast and actual moved the same way from the decision-month price;
  undefined for "no change"), actual inside the 80 % interval.
- **Confidence (ex ante):** for a forecast made at *t*, only the same model's earlier errors at
  the same horizon whose target month is ≤ *t* are used (frozen track: seeded with its
  development-stage validation errors). Each past log error is scaled by current 12-month
  volatility / volatility when it was made. From these: p(within ±5 %), p(within ±10 %),
  p(price above the decision-month level). Label: High if p(±10 %) ≥ 0.70 and the non-naive
  models' forecasts differ by ≤ 5 % of the price; Low if p(±10 %) < 0.50 or they differ by
  > 15 %; otherwise Medium; fewer than 12 past errors → "Not enough history". Whether the labels
  meant anything is checked against outcomes (`artifacts/metrics/confidence_check.md`), and p_up
  is checked with a reliability table and Brier score.
- **Live data:** `cocoasense update` (and a dashboard button) re-downloads the World Bank and FRED
  files (old copies archived), validates, refits the learning track incrementally, and scores
  earlier forecasts as their actual prices are published. Stored forecasts are never rewritten.
  Free data is monthly, so "live" means the latest published monthly average.
- **Not done:** no intraday prices (no open licence); no trading advice; the confidence labels
  describe past performance of similar forecasts, not guarantees.
