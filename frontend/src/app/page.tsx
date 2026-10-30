"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine
} from "recharts";

const COLORS: Record<string, string> = {
  "Buy & Hold": "#64748b",
  "TS Momentum (60d)": "#0284c7",
  "Mean Reversion (20d)": "#d97706",
  "Walk-Forward XGBoost": "#16a34a",
  "PPO Agent (Ensemble)": "#dc2626",
};

export default function QuantResearchTearsheet() {
  const [sttBps, setSttBps] = useState<number>(10.0);
  const [brokerageBps, setBrokerageBps] = useState<number>(2.0);
  const [slippageBps, setSlippageBps] = useState<number>(5.0);
  const [activeTab, setActiveTab] = useState<"nav" | "drawdown">("nav");
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const fetchSimulation = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(
        `/api/simulate?stt_bps=${sttBps}&brokerage_bps=${brokerageBps}&slippage_bps=${slippageBps}`
      );
      const json = await res.json();
      setData(json);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [sttBps, brokerageBps, slippageBps]);

  useEffect(() => {
    const timer = setTimeout(fetchSimulation, 60);
    return () => clearTimeout(timer);
  }, [fetchSimulation]);

  const summary = data?.summary || {};
  const chartData = data?.chart_data || [];
  const totalBps = (sttBps + brokerageBps + slippageBps).toFixed(1);

  return (
    <div className="min-h-screen bg-[#fafafa] text-neutral-900 font-sans antialiased">
      {/* Top Navigation */}
      <header className="bg-white border-b border-neutral-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div className="flex items-baseline gap-3">
            <h1 className="text-lg font-semibold tracking-tight text-neutral-900">
              QuantLab
            </h1>
            <span className="text-xs text-neutral-400">/</span>
            <span className="text-sm text-neutral-600">
              Empirical Tearsheet: Microstructure Friction & Multiple-Testing Audit
            </span>
          </div>
          <div className="flex items-center gap-5 text-xs text-neutral-600">
            <span>Asset: <strong className="text-neutral-900 font-mono">RELIANCE.NS</strong></span>
            <span>Out-of-Sample: <strong className="text-neutral-900 font-mono">2023–2025</strong></span>
            <a
              href="https://github.com/anushkaaa26/quant-lab"
              target="_blank"
              rel="noreferrer"
              className="text-neutral-900 font-medium hover:underline"
            >
              GitHub Repository →
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left Column: Execution & Cost Assumptions */}
          <aside className="lg:col-span-3 bg-white border border-neutral-200 rounded-md p-5">
            <div className="border-b border-neutral-100 pb-3 mb-4">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
                Execution Assumptions
              </h2>
              <p className="text-xs text-neutral-500 mt-1 leading-relaxed">
                Adjust statutory levies and market impact to stress-test post-cost alpha decay.
              </p>
            </div>

            {/* Regime Presets */}
            <div className="mb-5">
              <label className="block text-[11px] font-medium text-neutral-500 mb-2">
                Fee Regime Presets
              </label>
              <div className="grid grid-cols-3 gap-1.5 text-xs">
                <button
                  onClick={() => { setSttBps(0); setBrokerageBps(0); setSlippageBps(0); }}
                  className="py-1.5 px-2 border border-neutral-200 rounded text-neutral-700 hover:bg-neutral-50 font-medium transition"
                >
                  Zero Cost
                </button>
                <button
                  onClick={() => { setSttBps(2.5); setBrokerageBps(1.0); setSlippageBps(3.0); }}
                  className="py-1.5 px-2 border border-neutral-200 rounded text-neutral-700 hover:bg-neutral-50 font-medium transition"
                >
                  Intraday
                </button>
                <button
                  onClick={() => { setSttBps(10.0); setBrokerageBps(2.0); setSlippageBps(5.0); }}
                  className="py-1.5 px-2 border border-neutral-200 rounded text-neutral-700 hover:bg-neutral-50 font-medium transition"
                >
                  Delivery
                </button>
              </div>
            </div>

            {/* Sliders */}
            <div className="space-y-4 text-xs">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-neutral-700 font-medium">Securities Transaction Tax</span>
                  <span className="font-mono text-neutral-900">{sttBps.toFixed(1)} bps</span>
                </div>
                <input
                  type="range" min="0" max="25" step="0.5"
                  value={sttBps} onChange={(e) => setSttBps(Number(e.target.value))}
                  className="w-full accent-neutral-900 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-neutral-700 font-medium">Institutional Brokerage</span>
                  <span className="font-mono text-neutral-900">{brokerageBps.toFixed(1)} bps</span>
                </div>
                <input
                  type="range" min="0" max="10" step="0.5"
                  value={brokerageBps} onChange={(e) => setBrokerageBps(Number(e.target.value))}
                  className="w-full accent-neutral-900 cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-neutral-700 font-medium">Market Impact Slippage (η)</span>
                  <span className="font-mono text-neutral-900">{slippageBps.toFixed(1)} bps</span>
                </div>
                <input
                  type="range" min="0" max="20" step="0.5"
                  value={slippageBps} onChange={(e) => setSlippageBps(Number(e.target.value))}
                  className="w-full accent-neutral-900 cursor-pointer"
                />
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-neutral-100 space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-neutral-500">One-Way Friction:</span>
                <span className="font-mono font-semibold text-neutral-900">{totalBps} bps</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-500">Execution Lag:</span>
                <span className="font-mono text-neutral-800">Close[t] → Open[t+1]</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-500">Risk-Free Benchmark:</span>
                <span className="font-mono text-neutral-800">7.00% p.a.</span>
              </div>
            </div>
          </aside>

          {/* Right Column: Charts & Tearsheet Table */}
          <div className="lg:col-span-9 space-y-6">
            
            {/* Chart Section */}
            <section className="bg-white border border-neutral-200 rounded-md p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 mb-5 border-b border-neutral-100">
                <div>
                  <h2 className="text-sm font-semibold text-neutral-900">
                    Out-of-Sample Cumulative Performance (2023–2025)
                  </h2>
                  <p className="text-xs text-neutral-500 mt-0.5">
                    Net of {totalBps} bps per unit of rebalanced turnover ({loading ? "updating..." : "synced"})
                  </p>
                </div>

                {/* View Switcher */}
                <div className="inline-flex rounded border border-neutral-200 p-0.5 bg-neutral-50 text-xs">
                  <button
                    onClick={() => setActiveTab("nav")}
                    className={`px-3 py-1 rounded-sm font-medium transition ${
                      activeTab === "nav" ? "bg-white text-neutral-900 shadow-sm" : "text-neutral-500 hover:text-neutral-800"
                    }`}
                  >
                    Cumulative NAV
                  </button>
                  <button
                    onClick={() => setActiveTab("drawdown")}
                    className={`px-3 py-1 rounded-sm font-medium transition ${
                      activeTab === "drawdown" ? "bg-white text-neutral-900 shadow-sm" : "text-neutral-500 hover:text-neutral-800"
                    }`}
                  >
                    Underwater Drawdown
                  </button>
                </div>
              </div>

              {/* Legend */}
              <div className="flex flex-wrap gap-5 mb-4 text-xs">
                {Object.keys(summary).map((name) => (
                  <div key={name} className="flex items-center gap-2">
                    <span
                      className="w-3 h-0.5 inline-block"
                      style={{ backgroundColor: COLORS[name] || "#000" }}
                    />
                    <span className="text-neutral-700 font-medium">{name}</span>
                  </div>
                ))}
              </div>

              {/* Plot */}
              <div className="h-[340px] w-full text-xs font-mono">
                <ResponsiveContainer width="100%" height="100%">
                  {activeTab === "nav" ? (
                    <LineChart data={chartData} margin={{ top: 5, right: 10, left: -15, bottom: 5 }}>
                      <CartesianGrid stroke="#f1f5f9" vertical={false} />
                      <XAxis dataKey="date" stroke="#94a3b8" tick={{ fontSize: 11, fill: "#64748b" }} minTickGap={45} />
                      <YAxis stroke="#94a3b8" domain={["auto", "auto"]} tick={{ fontSize: 11, fill: "#64748b" }} />
                      <ReferenceLine y={1.0} stroke="#cbd5e1" strokeDasharray="3 3" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#ffffff",
                          borderColor: "#e2e8f0",
                          fontSize: "12px",
                          boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
                        }}
                      />
                      {Object.keys(summary).map((name) => (
                        <Line
                          key={name}
                          type="linear"
                          dataKey={name}
                          stroke={COLORS[name] || "#000"}
                          dot={false}
                          strokeWidth={name === "Buy & Hold" ? 1.5 : 2}
                          isAnimationActive={false}
                        />
                      ))}
                    </LineChart>
                  ) : (
                    <AreaChart data={chartData} margin={{ top: 5, right: 10, left: -15, bottom: 5 }}>
                      <CartesianGrid stroke="#f1f5f9" vertical={false} />
                      <XAxis dataKey="date" stroke="#94a3b8" tick={{ fontSize: 11, fill: "#64748b" }} minTickGap={45} />
                      <YAxis stroke="#94a3b8" tick={{ fontSize: 11, fill: "#64748b" }} />
                      <ReferenceLine y={0} stroke="#cbd5e1" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#ffffff",
                          borderColor: "#e2e8f0",
                          fontSize: "12px",
                        }}
                      />
                      {Object.keys(summary).map((name) => (
                        <Area
                          key={`${name}_dd`}
                          type="linear"
                          dataKey={`${name}_dd`}
                          name={name}
                          stroke={COLORS[name] || "#000"}
                          fill={COLORS[name] || "#000"}
                          fillOpacity={0.05}
                          strokeWidth={1.5}
                          isAnimationActive={false}
                        />
                      ))}
                    </AreaChart>
                  )}
                </ResponsiveContainer>
              </div>
            </section>

            {/* Empirical Audit Table */}
            <section className="bg-white border border-neutral-200 rounded-md overflow-hidden">
              <div className="px-6 py-4 border-b border-neutral-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <h2 className="text-sm font-semibold text-neutral-900">
                    Performance & Statistical Validation Summary
                  </h2>
                  <p className="text-xs text-neutral-500 mt-0.5">
                    Evaluated using Stationary Block Bootstrap (b=20) and Bailey & López de Prado (2014) Deflated Sharpe Ratio.
                  </p>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-neutral-200 bg-neutral-50/70 text-neutral-500 font-medium">
                      <th className="py-3 px-4">Strategy</th>
                      <th className="py-3 px-4 text-right">CAGR</th>
                      <th className="py-3 px-4 text-right">Ann. Vol</th>
                      <th className="py-3 px-4 text-right">Gross SR</th>
                      <th className="py-3 px-4 text-right">Net SR</th>
                      <th className="py-3 px-4 text-right">Sortino</th>
                      <th className="py-3 px-4 text-right">Max DD</th>
                      <th className="py-3 px-4 text-right">Turnover</th>
                      <th className="py-3 px-4 text-right">95% Bootstrap CI</th>
                      <th className="py-3 px-4 text-right">DSR p-val</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-100 font-mono text-[12px]">
                    {Object.entries(summary).map(([name, m]: any) => {
                      const netSr = Number(m.net_sharpe);
                      const cagr = Number(m.ann_return);
                      return (
                        <tr key={name} className="hover:bg-neutral-50/80 transition-colors">
                          <td className="py-3 px-4 font-sans font-medium text-neutral-900 flex items-center gap-2">
                            <span
                              className="w-2 h-2 rounded-full inline-block"
                              style={{ backgroundColor: COLORS[name] || "#000" }}
                            />
                            {name}
                          </td>
                          <td className={`py-3 px-4 text-right tabular-nums ${cagr >= 0 ? "text-emerald-700" : "text-rose-600"}`}>
                            {cagr > 0 ? `+${cagr.toFixed(2)}%` : `${cagr.toFixed(2)}%`}
                          </td>
                          <td className="py-3 px-4 text-right text-neutral-600 tabular-nums">
                            {Number(m.ann_vol).toFixed(2)}%
                          </td>
                          <td className="py-3 px-4 text-right text-neutral-500 tabular-nums">
                            {Number(m.gross_sharpe).toFixed(2)}
                          </td>
                          <td className={`py-3 px-4 text-right font-semibold tabular-nums ${netSr >= 0.5 ? "text-emerald-700" : netSr < 0 ? "text-rose-600" : "text-neutral-900"}`}>
                            {netSr.toFixed(2)}
                          </td>
                          <td className="py-3 px-4 text-right text-neutral-600 tabular-nums">
                            {Number(m.sortino).toFixed(2)}
                          </td>
                          <td className="py-3 px-4 text-right text-rose-600 tabular-nums">
                            {Number(m.max_dd).toFixed(2)}%
                          </td>
                          <td className="py-3 px-4 text-right text-neutral-700 tabular-nums">
                            {Number(m.turnover_ann).toFixed(1)}x
                          </td>
                          <td className="py-3 px-4 text-right text-neutral-500 tabular-nums">
                            {m.ci_95}
                          </td>
                          <td className="py-3 px-4 text-right tabular-nums text-neutral-600">
                            {m.dsr_p === "BENCHMARK" ? "—" : m.dsr_p}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Research Takeaway Note */}
              <div className="px-6 py-3.5 bg-neutral-50 border-t border-neutral-200 text-xs text-neutral-600 leading-relaxed">
                <strong className="text-neutral-900">Empirical Finding:</strong> High-frequency rebalancing in walk-forward ML models incurs severe frictional drag under Indian delivery taxes (10 bps STT) and volume slippage. Toggle the <em>Zero Cost</em> preset on the left to compare theoretical gross alpha against realized post-cost performance.
              </div>
            </section>

          </div>
        </div>
      </main>
    </div>
  );
}
