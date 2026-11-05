## Why the live forecasts are what they are (data through Oct 2026): contribution of each input to the forecast log return

| model | horizon | decision_date | feature | label | value | contribution | base | prediction |
|---|---|---|---|---|---|---|---|---|
| random_forest | 1 | 2026-10 | log_price | Price level | 8.657 | -0.006 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | ret_1m | Change over the last month | -0.029 | -0.003 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | ret_2m | Change over the last 2 months | -0.034 | 0.000 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | ret_3m | Change over the last 3 months | 0.025 | 0.001 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | ret_6m | Change over the last 6 months | 0.525 | -0.001 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | ret_12m | Change over the last 12 months | -0.034 | -0.000 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | vol_3m | Volatility, last 3 months | 0.045 | -0.000 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | vol_6m | Volatility, last 6 months | 0.111 | -0.005 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | vol_12m | Volatility, last 12 months | 0.152 | -0.001 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | gap_ma_3m | Distance from 3-month average | -0.021 | -0.001 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | gap_ma_6m | Distance from 6-month average | 0.092 | 0.000 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | gap_ma_12m | Distance from 12-month average | 0.191 | -0.001 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | month_sin | Time of year | -0.866 | -0.000 | 0.004 | -0.012 |
| random_forest | 1 | 2026-10 | month_cos | Time of year | 0.500 | 0.000 | 0.004 | -0.012 |
| random_forest | 3 | 2026-10 | log_price | Price level | 8.657 | -0.017 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | ret_1m | Change over the last month | -0.029 | -0.003 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | ret_2m | Change over the last 2 months | -0.034 | 0.000 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | ret_3m | Change over the last 3 months | 0.025 | -0.000 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | ret_6m | Change over the last 6 months | 0.525 | -0.002 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | ret_12m | Change over the last 12 months | -0.034 | -0.002 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | vol_3m | Volatility, last 3 months | 0.045 | 0.001 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | vol_6m | Volatility, last 6 months | 0.111 | -0.004 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | vol_12m | Volatility, last 12 months | 0.152 | 0.004 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | gap_ma_3m | Distance from 3-month average | -0.021 | -0.003 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | gap_ma_6m | Distance from 6-month average | 0.092 | 0.000 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | gap_ma_12m | Distance from 12-month average | 0.191 | -0.001 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | month_sin | Time of year | -0.866 | -0.000 | 0.011 | -0.015 |
| random_forest | 3 | 2026-10 | month_cos | Time of year | 0.500 | 0.000 | 0.011 | -0.015 |
| xgboost | 1 | 2026-10 | log_price | Price level | 8.657 | -0.012 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | ret_1m | Change over the last month | -0.029 | -0.007 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | ret_2m | Change over the last 2 months | -0.034 | -0.000 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | ret_3m | Change over the last 3 months | 0.025 | 0.002 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | ret_6m | Change over the last 6 months | 0.525 | -0.013 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | ret_12m | Change over the last 12 months | -0.034 | -0.002 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | vol_3m | Volatility, last 3 months | 0.045 | -0.003 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | vol_6m | Volatility, last 6 months | 0.111 | -0.009 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | vol_12m | Volatility, last 12 months | 0.152 | 0.004 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | gap_ma_3m | Distance from 3-month average | -0.021 | -0.001 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | gap_ma_6m | Distance from 6-month average | 0.092 | 0.000 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | gap_ma_12m | Distance from 12-month average | 0.191 | -0.001 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | month_sin | Time of year | -0.866 | 0.000 | 0.003 | -0.038 |
| xgboost | 1 | 2026-10 | month_cos | Time of year | 0.500 | 0.000 | 0.003 | -0.038 |
| xgboost | 3 | 2026-10 | log_price | Price level | 8.657 | -0.067 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | ret_1m | Change over the last month | -0.029 | -0.004 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | ret_2m | Change over the last 2 months | -0.034 | 0.002 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | ret_3m | Change over the last 3 months | 0.025 | 0.002 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | ret_6m | Change over the last 6 months | 0.525 | -0.003 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | ret_12m | Change over the last 12 months | -0.034 | -0.010 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | vol_3m | Volatility, last 3 months | 0.045 | 0.000 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | vol_6m | Volatility, last 6 months | 0.111 | -0.014 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | vol_12m | Volatility, last 12 months | 0.152 | 0.014 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | gap_ma_3m | Distance from 3-month average | -0.021 | -0.003 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | gap_ma_6m | Distance from 6-month average | 0.092 | -0.000 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | gap_ma_12m | Distance from 12-month average | 0.191 | 0.006 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | month_sin | Time of year | -0.866 | -0.000 | 0.010 | -0.067 |
| xgboost | 3 | 2026-10 | month_cos | Time of year | 0.500 | 0.001 | 0.010 | -0.067 |
