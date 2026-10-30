import numpy as np
import pandas as pd
import pytest
from strategies.baselines import BuyAndHold, TimeSeriesMomentum, MeanReversionZScore, WalkForwardXGBoost


@pytest.fixture
def sample_market_data():
    dates = pd.date_range("2023-01-01", periods=600, freq="B")
    np.random.seed(42)
    drift = 0.0004
    vol = 0.015
    ret = np.random.normal(drift, vol, size=len(dates))
    close = 1000 * np.exp(np.cumsum(ret))

    df = pd.DataFrame({
        'open': close * 0.998,
        'high': close * 1.01,
        'low': close * 0.99,
        'close': close,
        'volume': np.random.uniform(100000, 500000, size=len(dates))
    }, index=dates)
    return df


def test_buy_and_hold_weights(sample_market_data):
    strat = BuyAndHold()
    w = strat.generate_weights(sample_market_data)
    assert len(w) == len(sample_market_data)
    assert (w == 1.0).all()


def test_momentum_bounds(sample_market_data):
    strat = TimeSeriesMomentum(lookback=20)
    w = strat.generate_weights(sample_market_data)
    assert w.isin([-1.0, 0.0, 1.0]).all()


def test_mean_reversion_bounds(sample_market_data):
    strat = MeanReversionZScore(window=20)
    w = strat.generate_weights(sample_market_data)
    assert w.isin([-1.0, 0.0, 1.0]).all()


def test_xgboost_walk_forward_execution(sample_market_data):
    strat = WalkForwardXGBoost(train_window=200, retrain_step=50, n_estimators=10)
    w = strat.generate_weights(sample_market_data)
    assert len(w) == len(sample_market_data)
    # The first train_window entries should be 0.0 as it had no prior data to predict
    assert (w.iloc[:200] == 0.0).all()
    assert set(w.unique()).issubset({-1.0, 0.0, 1.0})
