from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from cocoasense.config import PROJECT_ROOT, EvaluationConfig, FeatureConfig, load_config


def _eval(**overrides: object) -> dict[str, object]:
    path = PROJECT_ROOT / "configs" / "evaluation.yaml"
    base = yaml.safe_load(path.read_text(encoding="utf-8"))
    base.update(overrides)
    return base


def test_repo_configs_load() -> None:
    cfg = load_config()
    assert cfg.data.primary_price == "worldbank_pinksheet"
    assert cfg.evaluation.ts("final_train_end") < cfg.evaluation.ts("test_start")
    assert 3 in cfg.features.horizons


def test_cutoff_must_precede_test() -> None:
    with pytest.raises(ValidationError, match="strictly before test_start"):
        EvaluationConfig(**_eval(final_train_end="2020-01", test_start="2020-01"))


def test_chronology_enforced() -> None:
    with pytest.raises(ValidationError):
        EvaluationConfig(**_eval(train_end="2017-01"))


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        EvaluationConfig(**_eval(typo_field=1))


def test_feature_windows_validated() -> None:
    with pytest.raises(ValidationError):
        FeatureConfig(return_windows=[0, 1], rolling_windows=[3])
    with pytest.raises(ValidationError):
        FeatureConfig(return_windows=[1], rolling_windows=[1])
    f = FeatureConfig(
        return_windows=[3, 1, 3], rolling_windows=[12], use_fx=True, fx_return_windows=[24]
    )
    assert f.return_windows == [1, 3]
    assert f.max_lookback == 24


def test_period_bounds() -> None:
    cfg = load_config()
    bounds = cfg.evaluation.period_bounds()
    assert bounds["post_peak"][1] is None
    assert bounds["surge"][0].strftime("%Y-%m") == "2023-01"
