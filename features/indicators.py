import numpy as np
import pandas as pd


def compute_log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))


def compute_rolling_volatility(log_returns: pd.Series, window: int = 20) -> pd.Series:
    return log_returns.rolling(window=window).std() * np.sqrt(252)


def compute_momentum(prices: pd.Series, window: int = 20) -> pd.Series:
    return (prices / prices.shift(window)) - 1.0


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_zscore(prices: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    return (prices - rolling_mean) / rolling_std.replace(0.0, np.nan)


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    past_avg_volume = volume.shift(1).rolling(window=window).mean()
    return volume / past_avg_volume.replace(0.0, np.nan)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    close = df["close"]
    vol = df["volume"]

    log_ret = compute_log_returns(close)
    feats["log_ret_1d"] = log_ret
    feats["vol_10d"] = compute_rolling_volatility(log_ret, window=10)
    feats["vol_20d"] = compute_rolling_volatility(log_ret, window=20)
    feats["vol_60d"] = compute_rolling_volatility(log_ret, window=60)

    feats["mom_5d"] = compute_momentum(close, window=5)
    feats["mom_20d"] = compute_momentum(close, window=20)
    feats["mom_60d"] = compute_momentum(close, window=60)

    feats["rsi_14d"] = compute_rsi(close, window=14)
    feats["zscore_20d"] = compute_zscore(close, window=20)
    feats["volume_ratio_20d"] = compute_volume_ratio(vol, window=20)

    return feats.dropna()
