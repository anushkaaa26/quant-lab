import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import xgboost as xgb
from features.indicators import extract_features


class BaseStrategy:
    """Base class for all QuantLab strategy implementations."""
    def __init__(self, name: str):
        self.name = name

    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Subclasses must implement generate_weights")


class BuyAndHold(BaseStrategy):
    def __init__(self):
        super().__init__(name="Buy_and_Hold")

    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        """Constant unit long exposure."""
        return pd.Series(1.0, index=df.index, name="weight")


class TimeSeriesMomentum(BaseStrategy):
    def __init__(self, lookback: int = 60, vol_target: bool = False):
        super().__init__(name=f"TS_Momentum_{lookback}d")
        self.lookback = lookback
        self.vol_target = vol_target

    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates trailing cumulative return over past `lookback` periods.
        Signal is strictly past-only: close[t] / close[t - lookback] - 1.
        """
        mom = df['close'] / df['close'].shift(self.lookback) - 1.0
        weights = pd.Series(0.0, index=df.index)
        weights[mom > 0] = 1.0
        weights[mom < 0] = -1.0

        if self.vol_target:
            rolling_vol = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            target_vol = 0.15
            scale = (target_vol / rolling_vol.replace(0.0, np.nan)).clip(0.1, 2.0)
            weights = weights * scale

        return weights.fillna(0.0)


class MeanReversionZScore(BaseStrategy):
    def __init__(self, window: int = 20, entry_threshold: float = 1.5, exit_threshold: float = 0.5):
        super().__init__(name=f"Mean_Reversion_{window}d")
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        """
        Contrarian strategy buying dips (z < -entry) and shorting rallies (z > entry).
        Flattens out when price reverts back within exit threshold.
        """
        rolling_mean = df['close'].rolling(self.window).mean()
        rolling_std = df['close'].rolling(self.window).std()
        zscore = (df['close'] - rolling_mean) / rolling_std.replace(0.0, np.nan)

        weights = pd.Series(0.0, index=df.index)
        position = 0.0
        positions = []

        for z in zscore:
            if np.isnan(z):
                positions.append(0.0)
                continue

            if z <= -self.entry_threshold:
                position = 1.0
            elif z >= self.entry_threshold:
                position = -1.0
            elif abs(z) <= self.exit_threshold:
                position = 0.0

            positions.append(position)

        return pd.Series(positions, index=df.index, name="weight").fillna(0.0)


class WalkForwardXGBoost(BaseStrategy):
    def __init__(
        self,
        train_window: int = 504,     # 2 years of trading days
        retrain_step: int = 63,      # Retrain every quarter (~3 months)
        max_depth: int = 3,
        n_estimators: int = 50,
        learning_rate: float = 0.05
    ):
        super().__init__(name="Walk_Forward_XGBoost")
        self.train_window = train_window
        self.retrain_step = retrain_step
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def generate_weights(self, df: pd.DataFrame) -> pd.Series:
        """
        Walk-forward dynamic re-estimation:
        - Target: sign of next-day Open-to-Open return
        - Trains model on [t - train_window : t]
        - Predicts allocation for test block [t : t + retrain_step]
        """
        features_df = extract_features(df)
        
        # Target is next period return: Open[t+1] -> Open[t+2] or Close[t] -> Close[t+1]
        next_ret = df['close'].pct_change().shift(-1)
        aligned = features_df.join(next_ret.rename("target_ret")).dropna()

        X = aligned.drop(columns=["target_ret"])
        y = (aligned["target_ret"] > 0).astype(int)

        weights = pd.Series(0.0, index=df.index)
        n_samples = len(X)

        if n_samples < self.train_window + self.retrain_step:
            return weights

        # Sliding/expanding walk-forward splits
        for start_idx in range(0, n_samples - self.train_window, self.retrain_step):
            train_end_idx = start_idx + self.train_window
            test_end_idx = min(train_end_idx + self.retrain_step, n_samples)

            X_train = X.iloc[start_idx:train_end_idx]
            y_train = y.iloc[start_idx:train_end_idx]
            X_test = X.iloc[train_end_idx:test_end_idx]

            if len(X_test) == 0:
                break

            # Model fits strictly on the past training window
            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                eval_metric="logloss",
                random_state=42
            )
            clf.fit(X_train, y_train)

            # Predict probabilities on out-of-sample block
            prob_up = clf.predict_proba(X_test)[:, 1]
            # Map probabilities > 0.52 to Long (+1.0), < 0.48 to Short (-1.0), else Flat (0.0)
            block_weights = np.where(prob_up > 0.52, 1.0, np.where(prob_up < 0.48, -1.0, 0.0))

            test_dates = X_test.index
            weights.loc[test_dates] = block_weights

        return weights
