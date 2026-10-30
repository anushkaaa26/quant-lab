import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.loader import NiftyDataLoader
from backtest.engine import EventDrivenBacktester, TransactionCostModel
from strategies.baselines import BuyAndHold, TimeSeriesMomentum, MeanReversionZScore, WalkForwardXGBoost
from stats.hypothesis import stationary_block_bootstrap_ci

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14
})


def generate_plots(ticker: str = "RELIANCE.NS", test_start: str = "2023-01-01", test_end: str = "2025-12-31"):
    os.makedirs("reports", exist_ok=True)
    loader = NiftyDataLoader()
    df = loader.fetch_ticker(ticker, start="2015-01-01", end=test_end)

    cost_model = TransactionCostModel(
        brokerage_bps=2.0 / 10000.0,
        stt_bps=10.0 / 10000.0,
        exchange_sebi_bps=0.35 / 10000.0,
        stamp_duty_bps=1.5 / 10000.0,
        impact_coef=0.05
    )
    backtester = EventDrivenBacktester(df, cost_model=cost_model)

    strategies = {
        "Buy & Hold": BuyAndHold(),
        "TS Momentum (60d)": TimeSeriesMomentum(lookback=60),
        "Mean Reversion (20d)": MeanReversionZScore(window=20),
        "Walk-Forward XGBoost": WalkForwardXGBoost(train_window=504, retrain_step=63, n_estimators=40)
    }

    results = {}
    for name, strat in strategies.items():
        w = strat.generate_weights(df)
        res = backtester.run(w).loc[test_start:test_end]
        if not res.empty:
            results[name] = res

    # Check for RL PPO results
    rl_weights_path = "data/cache/rl_oos_weights.parquet"
    if os.path.exists(rl_weights_path):
        rl_w = pd.read_parquet(rl_weights_path).mean(axis=1)
        full_w = pd.Series(0.0, index=df.index)
        full_w.update(rl_w)
        rl_res = backtester.run(full_w).loc[test_start:test_end]
        if not rl_res.empty:
            results["PPO Agent (Ensemble)"] = rl_res

    # -------------------------------------------------------------
    # Plot 1: Cumulative Performance and Drawdowns
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    for name, res in results.items():
        cum_ret = (1.0 + res['net_ret']).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max

        ax1.plot(cum_ret.index, cum_ret, label=name, lw=1.6)
        ax2.plot(drawdown.index, drawdown * 100, label=name, lw=1.2)

    ax1.set_title(f"Out-of-Sample Performance: {ticker} ({test_start} to {test_end})")
    ax1.set_ylabel("Net Cumulative Return")
    ax1.legend(loc="upper left", frameon=True)

    ax2.set_title("Underwater Plot (Drawdown %)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    plt.tight_layout()
    fig.savefig("reports/equity_curves_drawdown.png", dpi=300)
    plt.close()
    print("Saved reports/equity_curves_drawdown.png")

    # -------------------------------------------------------------
    # Plot 2: Block Bootstrap Sharpe Ratio Distributions
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    num_bootstraps = 1500
    block_size = 20
    rng = np.random.default_rng(42)

    for name, res in results.items():
        rets = res['net_ret'].values
        n = len(rets)
        p = 1.0 / block_size
        boot_sharpes = np.empty(num_bootstraps)

        for b in range(num_bootstraps):
            indices = np.empty(n, dtype=int)
            cur = rng.integers(0, n)
            for i in range(n):
                indices[i] = cur
                cur = rng.integers(0, n) if rng.random() < p else (cur + 1) % n
            sample = rets[indices]
            s_std = np.std(sample, ddof=1)
            boot_sharpes[b] = (np.mean(sample) / s_std * np.sqrt(252)) if s_std > 0 else 0.0

        sns.kdeplot(boot_sharpes, label=name, ax=ax, fill=True, alpha=0.15, lw=1.8)

    ax.axvline(0.0, color="crimson", linestyle="--", lw=1.5, label="Null (SR = 0)")
    ax.set_title("Stationary Block Bootstrap: Net Sharpe Ratio Distributions")
    ax.set_xlabel("Annualized Sharpe Ratio")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig("reports/bootstrap_sharpe_distributions.png", dpi=300)
    plt.close()
    print("Saved reports/bootstrap_sharpe_distributions.png")

    # -------------------------------------------------------------
    # Plot 3: Friction Erosion (Gross vs Net Sharpe)
    # -------------------------------------------------------------
    gross_sharpes = []
    net_sharpes = []
    strat_names = []

    for name, res in results.items():
        g_std = res['gross_ret'].std() * np.sqrt(252)
        n_std = res['net_ret'].std() * np.sqrt(252)
        g_sr = (res['gross_ret'].mean() * 252) / g_std if g_std > 0 else 0.0
        n_sr = (res['net_ret'].mean() * 252) / n_std if n_std > 0 else 0.0
        
        strat_names.append(name)
        gross_sharpes.append(g_sr)
        net_sharpes.append(n_sr)

    x = np.arange(len(strat_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, gross_sharpes, width, label='Gross Sharpe (Pre-Cost)', color='#4c72b0')
    ax.bar(x + width/2, net_sharpes, width, label='Net Sharpe (Post-Cost: STT+Slippage)', color='#dd8452')

    ax.set_ylabel('Annualized Sharpe Ratio')
    ax.set_title('Alpha Degradation: Impact of Transaction Friction')
    ax.set_xticks(x)
    ax.set_xticklabels(strat_names, rotation=15, ha='right')
    ax.axhline(0, color='grey', lw=0.8)
    ax.legend()
    plt.tight_layout()
    fig.savefig("reports/friction_erosion.png", dpi=300)
    plt.close()
    print("Saved reports/friction_erosion.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="RELIANCE.NS")
    parser.add_argument("--test_start", type=str, default="2023-01-01")
    parser.add_argument("--test_end", type=str, default="2025-12-31")
    args = parser.parse_args()

    generate_plots(ticker=args.ticker, test_start=args.test_start, test_end=args.test_end)
