import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List

from data.loader import NiftyDataLoader
from backtest.engine import EventDrivenBacktester, TransactionCostModel, calculate_performance_metrics
from strategies.baselines import BuyAndHold, TimeSeriesMomentum, MeanReversionZScore, WalkForwardXGBoost
from stats.hypothesis import deflated_sharpe_ratio, stationary_block_bootstrap_ci


def run_tournament(
    ticker: str = "RELIANCE.NS",
    test_start: str = "2023-01-01",
    test_end: str = "2025-12-31"
):
    print(f"================================================================")
    print(f"  QuantLab Tournament: {ticker} (Out-of-Sample: {test_start} to {test_end})")
    print(f"================================================================")

    loader = NiftyDataLoader()
    # Load 10-year historical dataset
    df = loader.fetch_ticker(ticker, start="2015-01-01", end=test_end)

    # Instantiate strategies
    strategies = {
        "Buy & Hold": BuyAndHold(),
        "TS Momentum (60d)": TimeSeriesMomentum(lookback=60),
        "Mean Reversion (20d)": MeanReversionZScore(window=20, entry_threshold=1.5, exit_threshold=0.5),
        "Walk-Forward XGBoost": WalkForwardXGBoost(train_window=504, retrain_step=63, n_estimators=40, max_depth=3)
    }

    # Realistic Indian equity transaction cost model
    cost_model = TransactionCostModel(
        brokerage_bps=2.0 / 10000.0,
        stt_bps=10.0 / 10000.0,
        exchange_sebi_bps=0.35 / 10000.0,
        stamp_duty_bps=1.5 / 10000.0,
        impact_coef=0.05
    )

    backtester = EventDrivenBacktester(df, cost_model=cost_model)
    results_table = []
    oos_returns_dict = {}

    # 1. Evaluate baseline strategies
    for name, strat in strategies.items():
        print(f"Running strategy: {name}...")
        weights = strat.generate_weights(df)
        bt_df = backtester.run(weights)

        # Filter strictly for out-of-sample held-out period
        oos_df = bt_df.loc[test_start:test_end]
        if oos_df.empty:
            continue

        oos_returns_dict[name] = oos_df['net_ret']
        metrics = calculate_performance_metrics(oos_df['net_ret'])

        # Calculate annual turnover
        turnover_ann = oos_df['turnover'].mean() * 252

        results_table.append({
            "Strategy": name,
            "Annualized Return": f"{metrics['Annualized Return'] * 100:.2f}%",
            "Volatility": f"{metrics['Annualized Volatility'] * 100:.2f}%",
            "Gross Sharpe": f"{(oos_df['gross_ret'].mean() * 252) / (oos_df['gross_ret'].std() * np.sqrt(252)):.2f}",
            "Net Sharpe": metrics["Sharpe Ratio"],
            "Max Drawdown": f"{metrics['Max Drawdown'] * 100:.2f}%",
            "Annual Turnover": f"{turnover_ann:.2f}x",
            "raw_net_ret": oos_df['net_ret'].values
        })

    # 2. Check and integrate RL PPO agent if trained
    rl_weights_path = "data/cache/rl_oos_weights.parquet"
    if os.path.exists(rl_weights_path):
        print("Integrating RL PPO Out-of-Sample policies...")
        rl_weights_df = pd.read_parquet(rl_weights_path)
        # Average ensemble across tested seeds
        ensemble_rl_w = rl_weights_df.mean(axis=1)
        full_rl_w = pd.Series(0.0, index=df.index)
        full_rl_w.update(ensemble_rl_w)

        rl_bt_df = backtester.run(full_rl_w)
        rl_oos_df = rl_bt_df.loc[test_start:test_end]

        if not rl_oos_df.empty:
            oos_returns_dict["PPO Agent (Ensemble)"] = rl_oos_df['net_ret']
            metrics_rl = calculate_performance_metrics(rl_oos_df['net_ret'])
            rl_turnover = rl_oos_df['turnover'].mean() * 252

            results_table.append({
                "Strategy": "PPO Agent (Ensemble)",
                "Annualized Return": f"{metrics_rl['Annualized Return'] * 100:.2f}%",
                "Volatility": f"{metrics_rl['Annualized Volatility'] * 100:.2f}%",
                "Gross Sharpe": f"{(rl_oos_df['gross_ret'].mean() * 252) / (rl_oos_df['gross_ret'].std() * np.sqrt(252)):.2f}",
                "Net Sharpe": metrics_rl["Sharpe Ratio"],
                "Max Drawdown": f"{metrics_rl['Max Drawdown'] * 100:.2f}%",
                "Annual Turnover": f"{rl_turnover:.2f}x",
                "raw_net_ret": rl_oos_df['net_ret'].values
            })

    # 3. Statistical Validation: Block Bootstrap & Deflated Sharpe
    num_trials = len(results_table)
    all_net_sharpes = [r["Net Sharpe"] for r in results_table]
    sr_variance = float(np.var(all_net_sharpes)) if len(all_net_sharpes) > 1 else 0.5

    for row in results_table:
        net_ret = row.pop("raw_net_ret")
        obs_sr = row["Net Sharpe"]

        # Stationary Block Bootstrap 95% Confidence Interval
        ci_lower, ci_upper = stationary_block_bootstrap_ci(
            net_ret, num_bootstraps=1000, block_size=20, alpha=0.05
        )
        row["Bootstrap 95% CI"] = f"[{ci_lower:.2f}, {ci_upper:.2f}]"

        # Deflated Sharpe Ratio (DSR p-value)
        skew = float(pd.Series(net_ret).skew())
        kurt = float(pd.Series(net_ret).kurtosis()) + 3.0
        dsr_p = deflated_sharpe_ratio(
            observed_sr=obs_sr,
            sr_variance=sr_variance,
            nb_trials=num_trials,
            skewness=skew,
            kurtosis=kurt,
            sample_length=len(net_ret)
        )
        row["DSR p-value"] = f"{dsr_p:.4f}"
        row["Net Sharpe"] = f"{obs_sr:.2f}"

    # Print Final Summary Table
    summary_df = pd.DataFrame(results_table)
    print("\n" + "=" * 100)
    print("                              OUT-OF-SAMPLE PERFORMANCE & RIGOR")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    # Save to reports
    os.makedirs("reports", exist_ok=True)
    summary_df.to_csv("reports/tournament_summary.csv", index=False)
    print("Summary saved to reports/tournament_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="RELIANCE.NS")
    parser.add_argument("--test_start", type=str, default="2023-01-01")
    parser.add_argument("--test_end", type=str, default="2025-12-31")
    args = parser.parse_args()

    run_tournament(ticker=args.ticker, test_start=args.test_start, test_end=args.test_end)
