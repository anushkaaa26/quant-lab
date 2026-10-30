import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TransactionCostModel:
    brokerage_bps: float = 2.0 / 10000.0       # 2 bps institutional rate
    stt_bps: float = 10.0 / 10000.0             # 10 bps delivery STT
    exchange_sebi_bps: float = 0.35 / 10000.0   # Exchange + SEBI charges
    stamp_duty_bps: float = 1.5 / 10000.0       # Stamp duty
    impact_coef: float = 0.05                   # Slippage parameter

    def calculate_cost(self, trade_volume: float, adv: float, volatility: float) -> float:
        if trade_volume == 0 or adv == 0:
            return 0.0
        pct_fees = self.brokerage_bps + self.stt_bps + self.exchange_sebi_bps + self.stamp_duty_bps
        slippage = self.impact_coef * volatility * np.sqrt(trade_volume / adv)
        return trade_volume * (pct_fees + slippage)


class EventDrivenBacktester:
    """
    Guarantees zero lookahead:
    Signals decided on Close[t] are active starting from Open[t+1].
    """
    def __init__(self, data: pd.DataFrame, cost_model: TransactionCostModel = TransactionCostModel()):
        self.data = data.sort_index().copy()
        self.cost_model = cost_model
        self._validate_schema()

    def _validate_schema(self):
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(self.data.columns):
            raise ValueError(f"Data missing required columns: {required - set(self.data.columns)}")

    def run(self, target_weights: pd.Series) -> pd.DataFrame:
        df = self.data.copy()
        
        # Target weight decided at Close[t]
        df['target_w'] = target_weights.reindex(df.index).fillna(0.0)
        
        # Position execution happens at Open[t+1]
        df['active_w'] = df['target_w'].shift(1).fillna(0.0)

        # Asset return captured: Open[t] to Open[t+1]
        df['ret_asset'] = df['open'].shift(-1) / df['open'] - 1.0

        # Turnover incurred at Open[t]
        df['turnover'] = (df['active_w'] - df['active_w'].shift(1).fillna(0.0)).abs()
        df['rolling_vol'] = df['close'].pct_change().rolling(20).std().fillna(0.01)
        df['adv'] = df['volume'].rolling(20).mean().fillna(df['volume'])

        # Apply transaction costs
        costs = []
        for _, row in df.iterrows():
            c = self.cost_model.calculate_cost(row['turnover'], row['adv'], row['rolling_vol'])
            costs.append(c)
        df['cost'] = costs

        # Gross and Net returns
        df['gross_ret'] = df['active_w'] * df['ret_asset']
        df['net_ret'] = df['gross_ret'] - df['cost']

        df['cum_gross'] = (1.0 + df['gross_ret'].fillna(0.0)).cumprod()
        df['cum_net'] = (1.0 + df['net_ret'].fillna(0.0)).cumprod()

        return df.dropna(subset=['net_ret'])


def calculate_performance_metrics(net_returns: pd.Series, risk_free_annual: float = 0.07) -> Dict[str, Any]:
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0
    excess_returns = net_returns - rf_daily

    mean_ret = net_returns.mean() * 252
    vol = net_returns.std() * np.sqrt(252)
    sharpe = (excess_returns.mean() * 252) / vol if vol > 0 else 0.0

    # Downside volatility for Sortino
    downside_returns = net_returns[net_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-6
    sortino = (excess_returns.mean() * 252) / downside_vol

    # Drawdown metrics
    equity_curve = (1.0 + net_returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    calmar = mean_ret / abs(max_dd) if max_dd < 0 else 0.0

    return {
        "Annualized Return": mean_ret,
        "Annualized Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar
    }
