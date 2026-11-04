from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cocoasense.config import FeatureConfig
from cocoasense.features.build import (
    build_features,
    build_targets,
    feature_columns,
    training_table,
)

CFG = FeatureConfig(return_windows=[1, 3], rolling_windows=[3], horizons=[1, 3])


def _toy() -> pd.DataFrame:
    """SYNTHETIC: prices 100, 110, 121, ... (constant 10% growth) plus a flat FX series."""
    idx = pd.date_range("2000-01", periods=8, freq="MS")
    return pd.DataFrame({"price_usd_t": 100 * 1.1 ** np.arange(8), "ghs_per_usd": 2.0}, index=idx)


def test_hand_computed_values() -> None:
    f = build_features(_toy(), None, CFG)
    assert np.isnan(f["ret_1m"].iloc[0])
    np.testing.assert_allclose(f["ret_1m"].iloc[1:], np.log(1.1))
    np.testing.assert_allclose(f["ret_3m"].iloc[3:], 3 * np.log(1.1))
    np.testing.assert_allclose(f["vol_3m"].iloc[3:], 0.0, atol=1e-12)  # constant returns
    # gap to 3-month mean of log price: lp_t - mean(lp_t, lp_t-1, lp_t-2) = log(1.1)
    np.testing.assert_allclose(f["gap_ma_3m"].iloc[2:], np.log(1.1))
    assert f["month_sin"].iloc[2] == pytest.approx(np.sin(2 * np.pi * 3 / 12))
    assert list(f.columns) == feature_columns(CFG)


def test_targets_look_forward_only_for_labels() -> None:
    t = build_targets(_toy(), [1, 3])
    np.testing.assert_allclose(t["target_3m"].iloc[:5], 3 * np.log(1.1))
    assert t["target_3m"].iloc[-3:].isna().all()


def test_fx_features_need_fx_column() -> None:
    cfg = CFG.model_copy(update={"use_fx": True, "fx_return_windows": [1]})
    assert "fx_ret_1m" in build_features(_toy(), None, cfg)
    with pytest.raises(KeyError):
        build_features(_toy().drop(columns="ghs_per_usd"), None, cfg)


def test_as_of_truncates() -> None:
    f = build_features(_toy(), pd.Timestamp("2000-04-01"), CFG)
    assert f.index.max() == pd.Timestamp("2000-04-01")


def test_training_table_drops_warmup_and_unknown_targets(history: pd.DataFrame) -> None:
    hist = history.loc[:"2000-12"]
    X, y = training_table(hist, CFG, horizon=3, train_start=pd.Timestamp("1980-01"))
    assert X.notna().all(axis=None) and y.notna().all()
    assert X.index.min() == pd.Timestamp("1980-01-01")
    # last label is for t = 2000-09, whose target month 2000-12 is inside the history
    assert X.index.max() == pd.Timestamp("2000-09-01")
