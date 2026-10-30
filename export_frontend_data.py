import json
import numpy as np
import pandas as pd
from data.loader import NiftyDataLoader
from backtest.engine import EventDrivenBacktester, TransactionCostModel, calculate_performance_metrics
from strategies.baselines import BuyAndHold, TimeSeriesMomentum, MeanReversionZScore, WalkForwardXGBoost

print("Loading Nifty 50 historical data for export...")
loader = NiftyDataLoader()
df = loader.fetch_ticker("RELIANCE.NS", start="2023-01-01", end="2025-12-31")

cost_model = TransactionCostModel(brokerage_bps=0.0002, stt_bps=0.0010, impact_coef=0.05)
bt = EventDrivenBacktester(df, cost_model=cost_model)

strategies = {
    "Buy & Hold": BuyAndHold(),
    "TS Momentum (60d)": TimeSeriesMomentum(lookback=60),
    "Mean Reversion (20d)": MeanReversionZScore(window=20),
    "Walk-Forward XGBoost": WalkForwardXGBoost(train_window=200, retrain_step=50, n_estimators=30)
}

payload = {
    "dates": [d.strftime("%Y-%m-%d") for d in df.index],
    "strategies": {}
}

for name, strat in strategies.items():
    print(f"Exporting series for: {name}...")
    w = strat.generate_weights(df)
    res = bt.run(w)
    m = calculate_performance_metrics(res['net_ret'])
    
    cum_net = (1.0 + res['net_ret']).cumprod().tolist()
    turnover = res['turnover'].tolist()
    gross_ret = res['gross_ret'].tolist()

    payload["strategies"][name] = {
        "cum_net": [round(x, 4) for x in cum_net],
        "turnover": [round(x, 4) for x in turnover],
        "gross_ret": [round(x, 6) for x in gross_ret],
        "ann_return": round(m["Annualized Return"] * 100, 2),
        "volatility": round(m["Annualized Volatility"] * 100, 2),
        "sharpe": round(m["Sharpe Ratio"], 2),
        "max_dd": round(m["Max Drawdown"] * 100, 2),
        "ann_turnover": round(res['turnover'].mean() * 252, 2)
    }

with open("dashboard_data.json", "w") as f:
    json.dump(payload, f)

print("Exported dashboard_data.json successfully.")
