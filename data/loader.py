import os
from typing import List, Optional
import pandas as pd
import yfinance as yf
from tqdm import tqdm

DEFAULT_NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "TATAMOTORS.NS", "NTPC.NS"
]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


class NiftyDataLoader:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, symbol: str, start: str, end: str) -> str:
        safe_sym = symbol.replace("^", "").replace(".NS", "")
        return os.path.join(self.cache_dir, f"{safe_sym}_{start}_{end}.parquet")

    def fetch_ticker(self, symbol: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
        cache_path = self._get_cache_path(symbol, start, end)
        
        if not force_refresh and os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            return df

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=False)

        if df.empty:
            raise ValueError(f"No price data retrieved for {symbol} between {start} and {end}")

        # Standardize column naming
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        })
        
        # If Adj Close is missing, fall back to close
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        # Corporate action adjustment ratio
        adj_ratio = df["adj_close"] / df["close"]
        df["open"] = df["open"] * adj_ratio
        df["high"] = df["high"] * adj_ratio
        df["low"] = df["low"] * adj_ratio
        df["close"] = df["adj_close"]

        cols = ["open", "high", "low", "close", "volume"]
        df = df[cols].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index().dropna()

        # Cache locally as Parquet
        df.to_parquet(cache_path)
        return df

    def fetch_universe(
        self, 
        symbols: Optional[List[str]] = None, 
        start: str = "2015-01-01", 
        end: str = "2025-12-31"
    ) -> dict[str, pd.DataFrame]:
        target_symbols = symbols or DEFAULT_NIFTY50_SYMBOLS
        universe_data = {}

        print(f"Loading {len(target_symbols)} instruments from {start} to {end}...")
        for sym in tqdm(target_symbols, desc="Downloading Nifty Equities"):
            try:
                universe_data[sym] = self.fetch_ticker(sym, start, end)
            except Exception as e:
                print(f"[Warning] Failed to fetch {sym}: {e}")

        return universe_data

    def fetch_benchmark(self, start: str = "2015-01-01", end: str = "2025-12-31") -> pd.DataFrame:
        """Fetches Nifty 50 Index (^NSEI) as benchmark"""
        return self.fetch_ticker("^NSEI", start, end)


if __name__ == "__main__":
    loader = NiftyDataLoader()
    # Smoke test: pull benchmark and one stock
    bench = loader.fetch_benchmark(start="2024-01-01", end="2024-12-31")
    rel = loader.fetch_ticker("RELIANCE.NS", start="2024-01-01", end="2024-12-31")
    print(f"Benchmark records cached: {len(bench)}")
    print(f"Reliance records cached: {len(rel)}")
