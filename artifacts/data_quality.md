# Data-quality report

## Series summary

| series | rows | first | last | min | max |
|---|---|---|---|---|---|
| price_usd_t | 802 | 1960-01 | 2026-10 | 260.0 | 10750.0 |
| ghs_per_usd | 433 | 1990-01 | 2026-01 | 0.0304 | 15.9889 |
| crosscheck_price_usd_t | 417 | 1992-01 | 2026-09 | 860.74 | 10710.3455 |

## Findings

### Warnings (2)

- **price_usd_t** · large_moves: monthly |log change| > 0.35: 1973-01 (+0.57)
- **price_usd_t** · crosscheck_months: months differing by more than 5.0%: 1993-06 (5.1%), 2001-02 (23.1%)

### Infos (4)

- **price_usd_t** · gaps: no missing months between first and last observation
- **ghs_per_usd** · gaps: no missing months between first and last observation
- **ghs_per_usd** · redenomination: no break around July 2007; series is in new cedis throughout. May–Sep 2007: {'2007-05': 0.9256, '2007-06': 0.9264, '2007-07': 0.927, '2007-08': 0.9298, '2007-09': 0.9354}
- **price_usd_t** · crosscheck: 417 overlapping months; mean |diff| 0.43%, median 0.18%, correlation 0.9998

