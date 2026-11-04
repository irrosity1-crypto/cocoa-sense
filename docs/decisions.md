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
