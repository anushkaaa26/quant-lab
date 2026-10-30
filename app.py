import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.loader import NiftyDataLoader
from backtest.engine import EventDrivenBacktester, TransactionCostModel, calculate_performance_metrics
from strategies.baselines import BuyAndHold, TimeSeriesMomentum, MeanReversionZScore, WalkForwardXGBoost
from stats.hypothesis import stationary_block_bootstrap_ci, deflated_sharpe_ratio

st.set_page_config(page_title="QuantLab Research Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 QuantLab: Quantitative Research & RL Strategy Studio")
st.caption("Walk-forward statistical validation & realistic friction testing on Indian Equities")

# Sidebar Controls
st.sidebar.header("Configuration & Universe")
ticker = st.sidebar.selectbox("Select Asset", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"], index=0)
test_start = st.sidebar.date_input("Out-of-Sample Start", pd.to_datetime("2023-01-01"))
test_end = st.sidebar.date_input("Out-of-Sample End", pd.to_datetime("2025-12-31"))

st.sidebar.subheader("Friction & Microstructure")
apply_stt = st.sidebar.checkbox("Include STT (10 bps)", value=True)
stt_rate = 10.0 / 10000.0 if apply_stt else 0.0
brokerage_bps = st.sidebar.slider("Brokerage (bps)", min_value=0.0, max_value=10.0, value=2.0) / 10000.0
impact_coef = st.sidebar.slider("Slippage Impact Factor (η)", min_value=0.0, max_value=0.15, value=0.05, step=0.01)

# Cache data loading
@st.cache_data
def load_data(sym):
    loader = NiftyDataLoader()
    return loader.fetch_ticker(sym, start="2015-01-01", end="2025-12-31")

with st.spinner("Loading market data & executing backtests..."):
    df = load_data(ticker)

cost_model = TransactionCostModel(
    brokerage_bps=brokerage_bps,
    stt_bps=stt_rate,
    exchange_sebi_bps=0.35 / 10000.0 if apply_stt else 0.0,
    stamp_duty_bps=1.5 / 10000.0 if apply_stt else 0.0,
    impact_coef=impact_coef
)

backtester = EventDrivenBacktester(df, cost_model=cost_model)

strategies = {
    "Buy & Hold": BuyAndHold(),
    "TS Momentum (60d)": TimeSeriesMomentum(lookback=60),
    "Mean Reversion (20d)": MeanReversionZScore(window=20),
    "Walk-Forward XGBoost": WalkForwardXGBoost(train_window=504, retrain_step=63, n_estimators=40)
}

res_dict = {}
for name, strat in strategies.items():
    w = strat.generate_weights(df)
    res = backtester.run(w).loc[str(test_start):str(test_end)]
    if not res.empty:
        res_dict[name] = res

# Display KPI cards
st.subheader("Performance Summary (Out-of-Sample)")
kpi_cols = st.columns(len(res_dict))

summary_rows = []
for idx, (name, res) in enumerate(res_dict.items()):
    metrics = calculate_performance_metrics(res['net_ret'])
    turnover = res['turnover'].mean() * 252
    ci_low, ci_high = stationary_block_bootstrap_ci(res['net_ret'].values, num_bootstraps=500, block_size=20)
    
    with kpi_cols[idx]:
        st.metric(
            label=name,
            value=f"{metrics['Sharpe Ratio']:.2f} SR",
            delta=f"{metrics['Annualized Return']*100:.1f}% Ann."
        )
        st.caption(f"Max DD: {metrics['Max Drawdown']*100:.1f}% | Turn: {turnover:.1f}x")

    summary_rows.append({
        "Strategy": name,
        "Ann Return": f"{metrics['Annualized Return']*100:.2f}%",
        "Volatility": f"{metrics['Annualized Volatility']*100:.2f}%",
        "Net Sharpe": round(metrics["Sharpe Ratio"], 2),
        "Max Drawdown": f"{metrics['Max Drawdown']*100:.2f}%",
        "Annual Turnover": f"{turnover:.2f}x",
        "Bootstrap 95% CI": f"[{ci_low:.2f}, {ci_high:.2f}]"
    })

# Main Charts
tab1, tab2, tab3 = st.tabs(["📈 Equity & Drawdowns", "📊 Alpha Degradation (Costs)", "🎲 Bootstrap Distributions"])

with tab1:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    for name, res in res_dict.items():
        cum = (1.0 + res['net_ret']).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        ax1.plot(cum.index, cum, label=name, lw=1.6)
        ax2.plot(dd.index, dd * 100, label=name, lw=1.2)
    
    ax1.set_ylabel("Net Cumulative Return")
    ax1.legend(loc="upper left")
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    st.markdown("#### Impact of Indian Statutory Taxes (STT) and Execution Friction")
    names, gross_sr, net_sr = [], [], []
    for name, res in res_dict.items():
        g_vol = res['gross_ret'].std() * np.sqrt(252)
        n_vol = res['net_ret'].std() * np.sqrt(252)
        names.append(name)
        gross_sr.append((res['gross_ret'].mean() * 252) / g_vol if g_vol > 0 else 0)
        net_sr.append((res['net_ret'].mean() * 252) / n_vol if n_vol > 0 else 0)

    f_df = pd.DataFrame({"Gross Sharpe (Pre-Cost)": gross_sr, "Net Sharpe (Post-Cost)": net_sr}, index=names)
    st.bar_chart(f_df)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

with tab3:
    st.markdown("#### Stationary Block Bootstrap Sharpe Densities (1,500 samples)")
    fig_b, ax_b = plt.subplots(figsize=(9, 4))
    rng = np.random.default_rng(42)
    for name, res in res_dict.items():
        rets = res['net_ret'].values
        n = len(rets)
        boot = []
        for _ in range(800):
            idx = rng.choice(n, size=n, replace=True)
            s = rets[idx]
            boot.append((np.mean(s) / np.std(s, ddof=1) * np.sqrt(252)) if np.std(s) > 0 else 0)
        sns.kdeplot(boot, label=name, ax=ax_b, fill=True, alpha=0.12)
    ax_b.axvline(0, color="crimson", linestyle="--", label="Null (SR=0)")
    ax_b.set_xlabel("Sharpe Ratio")
    ax_b.legend()
    st.pyplot(fig_b)
    plt.close()
