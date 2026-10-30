import numpy as np
import pandas as pd
import pytest
from backtest.engine import EventDrivenBacktester, TransactionCostModel


def test_engine_prevents_lookahead_execution():
    """
    Injects a signal that knows next day's Open price move.
    Verifies that the engine enforces the T+1 execution lag and prevents 
    capturing t+1 returns immediately at close t.
    """
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    
    # Generate prices
    base_price = 1000.0
    returns = np.random.normal(0.0005, 0.015, size=n)
    close_prices = base_price * np.exp(np.cumsum(returns))
    open_prices = close_prices * np.exp(np.random.normal(0, 0.005, size=n))

    df = pd.DataFrame({
        'open': open_prices,
        'high': np.maximum(open_prices, close_prices) * 1.01,
        'low': np.minimum(open_prices, close_prices) * 0.99,
        'close': close_prices,
        'volume': np.random.uniform(500000, 2000000, size=n)
    }, index=dates)

    # Intentionally leak future Open move: Open[t+1] vs Open[t]
    next_day_open_move = df['open'].shift(-1) - df['open']
    cheating_signal = pd.Series(np.where(next_day_open_move > 0, 1.0, -1.0), index=dates)

    # Zero cost model to isolate execution mechanics
    zero_cost = TransactionCostModel(brokerage_bps=0, stt_bps=0, exchange_sebi_bps=0, stamp_duty_bps=0, impact_coef=0)
    engine = EventDrivenBacktester(df, cost_model=zero_cost)
    results = engine.run(cheating_signal)

    # The cheat predicts Open[t] -> Open[t+1].
    # But signal generated at t executes at Open[t+1] and earns Open[t+1] -> Open[t+2].
    # Therefore, correlation with the predicted return should break.
    predicted_direction = np.sign(next_day_open_move.reindex(results.index))
    realized_direction = np.sign(results['gross_ret'])

    hit_rate = (predicted_direction == realized_direction).mean()
    assert hit_rate < 0.80, f"Engine leaked future prices! Suspicious hit rate: {hit_rate:.2%}"


def test_zero_trade_zero_cost():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        'open': [100.0] * 10,
        'high': [101.0] * 10,
        'low': [99.0] * 10,
        'close': [100.0] * 10,
        'volume': [100000] * 10
    }, index=dates)

    flat_signal = pd.Series(0.0, index=dates)
    engine = EventDrivenBacktester(df)
    res = engine.run(flat_signal)
    
    assert res['cost'].sum() == 0.0, "Non-zero costs recorded on zero turnover trades"
