## Exchange-rate experiment (validation period)

| stage | period | horizon | model | n | mae | rmse | mape | bias | directional_acc | within_5pct | within_10pct | coverage | interval_width_pct | rel_mae | rel_rmse | rel_mape | dm_stat | dm_pvalue |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation | validation | 1 | random_forest_fx@1991 | 48.000 | 92.121 | 127.063 | 3.873 | 18.282 | 0.583 | 0.667 | 0.896 | 0.854 | 14.660 | 1.011 | 0.997 | 1.012 | -0.500 | 0.619 |
| validation | validation | 1 | random_forest_nofx@1991 | 48.000 | 91.131 | 127.490 | 3.828 | 19.856 | 0.604 | 0.688 | 0.896 | 0.854 | 14.361 | 1.000 | 1.000 | 1.000 | – | – |
| validation | validation | 3 | random_forest_fx@1991 | 48.000 | 197.649 | 265.229 | 8.200 | 52.043 | 0.604 | 0.396 | 0.729 | 0.812 | 27.978 | 1.008 | 1.006 | 1.009 | 0.445 | 0.658 |
| validation | validation | 3 | random_forest_nofx@1991 | 48.000 | 196.052 | 263.702 | 8.129 | 56.225 | 0.625 | 0.417 | 0.750 | 0.792 | 27.820 | 1.000 | 1.000 | 1.000 | – | – |
| validation | validation | 1 | xgboost_fx@1991 | 48.000 | 94.413 | 125.679 | 3.964 | 8.064 | 0.458 | 0.646 | 0.917 | 0.896 | 16.471 | 1.013 | 1.011 | 1.013 | 0.926 | 0.359 |
| validation | validation | 1 | xgboost_nofx@1991 | 48.000 | 93.198 | 124.367 | 3.915 | 6.338 | 0.479 | 0.604 | 0.917 | 0.917 | 17.121 | 1.000 | 1.000 | 1.000 | – | – |
| validation | validation | 3 | xgboost_fx@1991 | 48.000 | 193.497 | 255.812 | 8.070 | 24.580 | 0.479 | 0.375 | 0.750 | 0.854 | 33.396 | 1.018 | 1.016 | 1.021 | 0.770 | 0.445 |
| validation | validation | 3 | xgboost_nofx@1991 | 48.000 | 190.055 | 251.671 | 7.901 | 28.556 | 0.604 | 0.396 | 0.771 | 0.812 | 30.530 | 1.000 | 1.000 | 1.000 | – | – |
