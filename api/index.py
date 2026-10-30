from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import scipy.stats as stats
import json
import os

app = FastAPI(docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "dashboard_data.json")

def load_base_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"dates": [], "strategies": {}}

def compute_dsr(observed_sr: float, sr_var: float, nb_trials: int, skew: float, kurt: float, n: int) -> float:
    if n <= 2 or nb_trials <= 1:
        return 0.5000
    euler = 0.5772156649
    z_n = (1.0 - euler) * stats.norm.ppf(1.0 - 1.0 / nb_trials) + euler * stats.norm.ppf(1.0 - 1.0 / (nb_trials * np.e))
    exp_max_sr = np.sqrt(max(sr_var, 1e-4)) * z_n
    denom = np.sqrt(max((1.0 + 0.5 * observed_sr**2 - skew * observed_sr + ((kurt - 3.0) / 4.0) * observed_sr**2) / (n - 1), 1e-6))
    return float(stats.norm.cdf((observed_sr - exp_max_sr) / denom))

@app.get("/api/simulate")
def simulate(
    stt_bps: float = Query(10.0, ge=0.0, le=50.0),
    brokerage_bps: float = Query(2.0, ge=0.0, le=20.0),
    slippage_bps: float = Query(5.0, ge=0.0, le=30.0),
):
    base_data = load_base_data()
    dates = base_data.get("dates", [])
    strategies = base_data.get("strategies", {})
    
    if not dates or not strategies:
        return {"meta": {}, "summary": {}, "chart_data": []}

    total_fee_rate = (stt_bps + brokerage_bps + slippage_bps) / 10000.0
    
    # Align length strictly to the shortest array to prevent off-by-one IndexError
    min_strat_len = min(len(s["gross_ret"]) for s in strategies.values())
    series_len = min(len(dates), min_strat_len)
    dates = dates[:series_len]

    results = {}
    nav_trajectories = {}
    dd_trajectories = {}
    net_sharpes_list = []

    for name, s in strategies.items():
        gross_arr = np.array(s["gross_ret"][:series_len], dtype=float)
        turn_arr = np.array(s["turnover"][:series_len], dtype=float)
        
        friction = turn_arr * total_fee_rate
        net_ret = gross_arr - friction
        
        nav_series = np.cumprod(1.0 + net_ret)

        rf_daily = 0.07 / 252.0
        ann_ret = float((nav_series[-1] ** (252.0 / max(series_len, 1)) - 1.0) * 100)
        ann_vol = float(np.std(net_ret, ddof=1) * np.sqrt(252) * 100) if series_len > 1 else 15.0
        
        gross_vol = float(np.std(gross_arr, ddof=1) * np.sqrt(252))
        gross_sr = float(((np.mean(gross_arr) - rf_daily) * 252) / gross_vol) if gross_vol > 0 else 0.0
        
        net_vol_dec = ann_vol / 100.0
        net_sr = float(((np.mean(net_ret) - rf_daily) * 252) / net_vol_dec) if net_vol_dec > 0 else 0.0
        net_sharpes_list.append(net_sr)

        downside = net_ret[net_ret < 0]
        down_vol = float(np.std(downside, ddof=1) * np.sqrt(252)) if len(downside) > 1 else 1e-4
        sortino = float(((np.mean(net_ret) - rf_daily) * 252) / down_vol)

        running_max = np.maximum.accumulate(nav_series)
        dd_series = ((nav_series - running_max) / running_max) * 100.0
        max_dd = float(np.min(dd_series))

        se_sr = np.sqrt((1.0 + 0.5 * net_sr**2) / max(series_len / 5.0, 1))
        ci_low = net_sr - 1.96 * se_sr
        ci_high = net_sr + 1.96 * se_sr

        skew = float(stats.skew(net_ret)) if series_len > 3 else 0.0
        kurt = float(stats.kurtosis(net_ret, fisher=False)) if series_len > 3 else 3.0

        results[name] = {
            "final_nav": round(float(nav_series[-1]), 4),
            "ann_return": round(ann_ret, 2),
            "ann_vol": round(ann_vol, 2),
            "gross_sharpe": round(gross_sr, 2),
            "net_sharpe": round(net_sr, 2),
            "sharpe_decay": round(gross_sr - net_sr, 2),
            "sortino": round(sortino, 2),
            "max_dd": round(max_dd, 2),
            "turnover_ann": round(float(np.mean(turn_arr) * 252), 2),
            "ci_95": f"[{ci_low:.2f}, {ci_high:.2f}]",
            "skew": skew,
            "kurt": kurt
        }
        nav_trajectories[name] = nav_series
        dd_trajectories[name] = dd_series

    sr_var = float(np.var(net_sharpes_list)) if len(net_sharpes_list) > 1 else 0.25
    for name, r in results.items():
        if name == "Buy & Hold":
            r["dsr_p"] = "BENCHMARK"
        else:
            p_val = compute_dsr(r["net_sharpe"], sr_var, len(results), r.pop("skew"), r.pop("kurt"), series_len)
            r["dsr_p"] = f"{p_val:.4f}"
        r.pop("skew", None)
        r.pop("kurt", None)

    chart_series = []
    step = max(1, series_len // 120)
    for i in range(0, series_len, step):
        pt = {"date": dates[i]}
        for name in strategies.keys():
            pt[name] = round(float(nav_trajectories[name][i]), 4)
            pt[f"{name}_dd"] = round(float(dd_trajectories[name][i]), 2)
        chart_series.append(pt)

    return {
        "meta": {
            "asset": "RELIANCE.NS",
            "exchange": "NSE",
            "sample_days": series_len,
            "total_friction_bps": round(stt_bps + brokerage_bps + slippage_bps, 2)
        },
        "summary": results,
        "chart_data": chart_series
    }
