# Methodology

## 1. Research question

Can machine-learning cocoa price forecasts improve the simulated forward-sales decisions of a
cocoa-exporting country compared with conventional selling strategies?

Sub-questions: (1) which model has the lowest out-of-sample error at 1- and 3-month horizons;
(2) does the GHS/USD exchange rate improve forecasts; (3) do forecast-driven strategies beat fixed
strategies on revenue and downside risk; (4) when are forecasts too uncertain to act on?

## 2. Timing convention

A row dated month *t* represents the decision taken at the **end** of month *t*, when that month's
average price is known. Features use data up to and including *t*. The target for horizon *h* is
the average price of month *t + h* (modelled as the log return `log(P_{t+h} / P_t)`).

## 3. Chronological protocol (the training cut-off)

| Stage | Models fitted on target months | Forecast target months | Purpose |
|---|---|---|---|
| Development | train_start – 2015-12 | 2016-01 – 2019-12 (validation) | choose training start, hyperparameter procedure, simulation rule |
| Freeze | — | — | write and commit `configs/frozen.yaml` |
| Final | train_start – **2019-12** | **2020-01 – latest month** (test) | the reported results, run once |

In both stages each model is **fitted once and then frozen**. It forecasts each evaluation month
in turn, receiving the actual history up to that decision month as input, but its parameters are
never updated. No model is ever trained on a target month after December 2019. The test period is
locked in code: `cocoasense evaluate --final` refuses to run without a frozen configuration, and
each final run is appended to `artifacts/final_runs.log` with the git commit and the SHA-256 of
the frozen configuration.

Why frozen rather than expanding-window refits: the project requirement is that models are not
trained on post-2019 data, so that forecasts can be compared honestly with what happened. This is
a stricter test than the usual expanding window, and the gap between the two is a limitation to
discuss rather than hide.

## 4. Data

Target: World Bank Pink Sheet monthly cocoa price, US$/tonne, 1960-01 – 2026-10 (802 months, no
gaps), cross-checked against IMF/FRED (mean absolute difference 0.43 %). Exchange rate: IMF,
cedis per US$, 1990-01 – 2026-01. See `data_sources.md` and `artifacts/data_quality.md`.

## 5. Features (past data only)

Produced by `features/build.py::build_features(history, as_of)`:

| Feature | Definition at month t |
|---|---|
| `log_price` | log P_t |
| `ret_{k}m`, k ∈ {1,2,3,6,12} | log P_t − log P_{t−k} |
| `vol_{w}m`, w ∈ {3,6,12} | standard deviation of the last w monthly log returns |
| `gap_ma_{w}m`, w ∈ {3,6,12} | log P_t − mean(log P over the last w months) |
| `month_sin`, `month_cos` | sin/cos(2π · month / 12) |
| `fx_ret_{k}m`, k ∈ {1,3,12} (FX experiment only) | log FX_t − log FX_{t−k} |

The documented feature table (`data/processed/features.parquet`) has **802 rows × 21 columns**
(price, FX, 17 features including FX, 2 targets). **Warm-up rows** whose look-back window is
incomplete are kept as NaN in the table and dropped explicitly when building training rows: the
first complete price-feature row is 1961-01 and the first complete FX-feature row is 1991-01
(`data/processed/features_summary.json`). Rows whose target lies beyond the training cut-off are
also dropped, so no label can leak.

## 6. Models

| Model | Method | Tuning (training data only) | Interval (80 %) |
|---|---|---|---|
| Naive | forecast = last observed price | none | empirical quantiles of h-month returns |
| ARIMA | ARIMA(p,1,q) on log price, no drift | p, q ∈ 0..3 by AIC | native, transformed from log |
| Random Forest | predicts h-month log return from features | grid search, `TimeSeriesSplit(5, gap=h)`, MAE | split-conformal on out-of-fold residuals |
| XGBoost | as above | as above | as above |

ARIMA "frozen" forecasting uses `SARIMAXResults.apply(new_data, refit=False)`: the Kalman filter
absorbs new observations, the coefficients stay fixed. Tree models are refitted from scratch only
at the cut-off; forecasts reuse the fitted estimator. Seeds come from `configs/models.yaml`.

## 7. Evaluation

Metrics per model, horizon and period (full test, pre-surge, surge, post-peak, by target month):
MAE and RMSE (US$/t), MAPE (%), bias, directional accuracy, share within ±5 % and ±10 %, 80 %
interval coverage and width, MAE relative to naive on identical months, and a Diebold–Mariano test
against naive (squared-error loss, Harvey–Leybourne–Newbold correction; indicative only at these
sample sizes). Validation results are in `artifacts/metrics/validation_*`; test results in
`artifacts/metrics/test_*` and `docs/results.md`.

## 8. Leakage controls (automated, `tests/leakage/`)

1. **Truncation** – features from full history equal features from history cut at `as_of`.
2. **Perturbation** – multiplying all prices after `as_of` by 10 changes no feature and no
   forecast made at or before `as_of`, for every model.
3. **Walk-forward** – every model reports its last training label; the harness rejects any model
   trained past the cut-off, and the test period is locked without `--final`.
4. **Fit on training only** – hyperparameters, conformal widths and ARIMA coefficients are
   identical whether or not post-cut-off data is perturbed.
5. **Strategies** – forward shares are unchanged by perturbing prices after the decision month.

## 9. Simulation

See `simulation_assumptions.md`. The forecast-driven rule and its driving model are chosen on
validation forecasts and frozen before the test run.
