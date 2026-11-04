# Results

All numbers below come from files in `artifacts/`, produced by the single logged final run
(`artifacts/final_runs.log`: 2026-11-04, git commit `4c076fa`, frozen configuration committed
before the run). Models were trained on data up to **December 2019** and then frozen; the test
period covers target months **January 2020 – October 2026** (82 forecasts per model and horizon).
MAPE is the mean absolute percentage error between forecast and actual monthly price.

## 1. Headline

- **No model clearly beat the naive "last price" forecast out of sample.** The best models
  reduced error by 1–4 % relative to naive over the full test period, and none of those gains is
  statistically significant.
- **The exchange rate did not improve forecasts** (sub-question 2).
- **The forecast-driven selling rule did not beat a simple 50/50 split** over the test period. It
  lost most during the 2023–25 surge, when the frozen models expected high prices to fall and
  therefore sold forward too early.

This is an acceptable and informative finding: it is what the efficient-markets view of commodity
prices predicts, and the evaluation shows *when* and *why* the models failed.

## 2. Forecast accuracy, full test period

Source: `artifacts/metrics/test_leaderboard_full_h1.md`, `..._h3.md`.

| Horizon | Model | MAPE % | MAE US$/t | MAE vs naive | Direction correct | Within ±10 % | 80 % interval coverage | DM p vs naive |
|---|---|---|---|---|---|---|---|---|
| 1 month | ARIMA | 6.87 | 380 | 0.964 | 58.5 % | 79.3 % | 72.0 % | 0.77 |
| 1 month | XGBoost | 7.17 | 386 | 0.978 | 53.7 % | 75.6 % | 72.0 % | 0.32 |
| 1 month | Random Forest | 7.11 | 390 | 0.988 | 56.1 % | 76.8 % | 69.5 % | 0.38 |
| 1 month | Naive | 7.20 | 395 | 1.000 | – | 79.3 % | 70.7 % | – |
| 3 months | ARIMA | 14.88 | 818 | 0.989 | 62.2 % | 51.2 % | 61.0 % | 0.60 |
| 3 months | XGBoost | 15.12 | 819 | 0.990 | 39.0 % | 48.8 % | 67.1 % | 0.71 |
| 3 months | Random Forest | 14.95 | 820 | 0.991 | 43.9 % | 50.0 % | 56.1 % | 0.75 |
| 3 months | Naive | 15.05 | 827 | 1.000 | – | 52.4 % | 61.0 % | – |

**Sub-question 1.** ARIMA has the lowest error at both horizons, but the margins over naive
(3.6 % at 1 month, 1.1 % at 3 months) are small and not significant. On validation (2016–2019)
the best models were about 5.5–5.9 % better than naive
(`artifacts/metrics/validation_leaderboard_h*.md`); that advantage mostly disappeared out of
sample.

## 3. Accuracy by period

Source: `artifacts/metrics/test_metrics.csv` (figure: `artifacts/figures/test_error_by_period.png`).

| Period (target months) | n | Best, 1 month (MAPE, MAE vs naive) | Naive MAPE, 1 month | Best, 3 months (MAPE, MAE vs naive) | Naive MAPE, 3 months |
|---|---|---|---|---|---|
| Pre-surge 2020-01 – 2022-12 | 36 | ARIMA 3.62 %, 0.979 | 3.69 % | **Naive** 5.14 % | 5.14 % |
| Surge 2023-01 – 2025-01 | 25 | ARIMA 8.99 %, 0.971 | 9.31 % | ARIMA 19.00 %, 0.978 | 19.52 % |
| Post-peak 2025-02 – 2026-10 | 21 | ARIMA 9.91 %, 0.950 | 10.73 % | XGBoost 24.97 %, 0.908 | 26.70 % |

- In the calm **pre-surge** period naive was best at 3 months; XGBoost (MAE ratio 1.130,
  DM p = 0.075) and ARIMA (1.016, DM p = 0.016) were *worse* than naive. That ARIMA result is the
  only one significant at the 5 % level in the test period, and it is a loss.
- Errors roughly tripled in the **surge** and **post-peak** periods. Every model, including
  naive, was biased low during the surge (3-month bias between −833 and −1,096 US$/t) and high
  after the peak (+153 to +545 US$/t).
- **Interval coverage collapsed outside the calm period.** At 3 months, nominal 80 % intervals
  covered 94–100 % of outcomes before 2023, 32–48 % in the surge and 19–33 % after the peak.
  The intervals were calibrated on 1980/1991–2019 data; the standard deviation of monthly log
  returns since 2023 (13.5 %) is about twice the pre-2020 level (6.5 %).

### Why the machine-learning models struggled in the surge

`log_price` was XGBoost's most important feature on validation
(`artifacts/metrics/validation_permutation_importance.csv`). In the test period the frozen
XGBoost predicted an average 3-month fall of 4.8 % when the price was US$3,000–5,000/t, 5.4 % at
US$5,000–8,000/t and 7.3 % above US$8,000/t (computed from
`artifacts/forecasts/test_predictions.parquet`). Trained on 1991–2019, when high cocoa prices
tended to revert, the model learned "high prices fall". In 2023–24 that was wrong for many months,
giving XGBoost the most negative surge bias (−1,096 US$/t at 3 months) and the worst surge
3-month MAE ratio (1.047). Tree models also cannot predict prices outside the range they were
trained on.

### Largest misses (turning points)

Source: `artifacts/metrics/test_largest_misses.md`. The largest errors came from sharp moves that
no model anticipated:

- **February 2026:** a 27.8 % fall in one month; 1-month errors of 35–40 % for all models.
- **April–May 2024:** +37.4 % then −22.6 % in consecutive months at the height of the surge.
- **3 months ahead:** January → April 2024 (+121 % actual move, errors of about −55 %) and
  December 2025 → March 2026 (−43.9 %, errors of about +76 to +80 %).

Every model's forecast in these months was close to the naive forecast: none had information about
the supply shocks behind the moves.

## 4. Exchange rate (sub-question 2)

Source: `artifacts/metrics/test_fx_experiment.md` and `validation_fx_experiment.md`. Paired
comparison: same models, same training start (1991-01), same 73 test months (FX data end
2026-01).

| Model | Horizon | MAE with FX ÷ MAE without | DM p |
|---|---|---|---|
| Random Forest | 1 | 1.006 | 0.08 |
| Random Forest | 3 | 0.994 | 0.57 |
| XGBoost | 1 | 1.008 | 0.96 |
| XGBoost | 3 | 0.998 | 0.72 |

Adding lagged GHS/USD changes moved error by less than 1 % in either direction, and made it
slightly worse on validation (1.008–1.018). **The exchange rate does not help forecast the world
cocoa price.** This is expected: the price is set in London and New York in US dollars, and the
cedi responds to cocoa export earnings more than it drives world prices.

## 5. Forward-sales simulation (sub-question 3)

Frozen rule: XGBoost-driven, `a = 2.0`, horizon 3 months, forward price = decision-month spot
price (spot proxy), financing 10 % a year, 82 decision months 2019-10 – 2026-07. Source:
`artifacts/simulation/strategy_metrics.md`.

| Period | Strategy | Mean US$/t | Worst 10 % mean US$/t | Regret vs best fixed | Win rate vs 50/50 | Mean forward share |
|---|---|---|---|---|---|---|
| Full | Fixed 50/50 | 4,204 | 2,262 | 417 | – | 0.50 |
| Full | Forecast-driven | 4,171 | 2,233 | 451 | 45.1 % | 0.65 |
| Full | Sell forward | 4,195 | 2,264 | 427 | 50.0 % | 1.00 |
| Full | Wait | 4,214 | 2,205 | 407 | 50.0 % | 0.00 |
| Surge | Fixed 50/50 | 5,030 | 2,512 | 629 | – | 0.50 |
| Surge | Forecast-driven | 4,799 | 2,576 | 860 | 40.0 % | 0.76 |
| Surge | Wait | 5,406 | 2,613 | 252 | 84.0 % | 0.00 |
| Post-peak | Forecast-driven | 6,527 | 3,410 | 607 | 52.4 % | 0.88 |
| Post-peak | Fixed 50/50 | 6,367 | 3,911 | 767 | – | 0.50 |

- Over the full period the forecast-driven rule earned **US$33/t (0.8 %) less** than Fixed 50/50,
  with a slightly worse worst-10 % outcome. It beat 50/50 in 45 % of decision months.
- In the **surge** it sold forward 76 % on average and earned US$231/t less than 50/50 and
  US$607/t less than simply waiting. This mirrors the pattern the scope document describes for
  Ghana's forward sales in 2023/24 (`artifacts/figures/simulation_strategy_outcomes.png`).
- **After the peak** the same bias helped: the rule beat 50/50 by US$160/t but still trailed
  selling everything forward.
- **Sensitivity** (`artifacts/simulation/sensitivity.md`; reported only, nothing chosen from it):
  test revenue fell steadily as `a` rose from 0 (4,204, identical to 50/50) to 2 (4,171), so the
  validation choice of the grid's largest `a` (decision D14) did not carry over. At a 1-month
  horizon the rule beat 50/50 (4,320 vs 4,281 US$/t, win rate 52.4 %), and an ARIMA-driven rule
  earned 4,240 US$/t. These are post-hoc observations, not the system's result.
- No strategy dominates: which fixed rule wins depends on whether prices rose or fell over the next
  three months, which none of the models could reliably foresee.

## 6. When are forecasts too uncertain to act on? (sub-question 4)

- **When volatility is far above the training era's.** Interval coverage fell from roughly 90 %
  to 20–50 % once monthly volatility doubled in 2023. A practical guard is to treat the intervals
  as unreliable when 12-month realised volatility exceeds the highest level seen in training, and
  fall back to a fixed split.
- **When the price is outside the training range.** Tree models cannot extrapolate, and all models
  carried a mean-reversion belief learned in a different regime.
- Scaling the forward share by the interval width (`z`) does not protect against this, because the
  intervals were themselves too narrow in exactly those periods.

## 7. What this does and does not show

It shows that, under the stated assumptions and on public monthly prices, models frozen at the end
of 2019 would not have improved a simple forward-selling policy over 2020–2026, and it explains
why. It does not measure Ghana's or COCOBOD's actual gains or losses, and the spot-proxy forward
price probably overstates what forward selling earned during the surge (`limitations.md`).

## Figures

`artifacts/figures/`: `eda_price_history.png`, `eda_returns_volatility.png`,
`eda_exchange_rate.png`, `validation_actual_vs_pred_h{1,3}.png`, `test_actual_vs_pred_h{1,3}.png`,
`test_rolling_mape_h{1,3}.png`, `test_error_by_period.png`, `simulation_strategy_outcomes.png`.
