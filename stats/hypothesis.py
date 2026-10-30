import numpy as np
import scipy.stats as stats
from typing import Tuple, Dict, Any


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_variance: float,
    nb_trials: int,
    skewness: float,
    kurtosis: float,
    sample_length: int,
) -> float:
    """
    Computes Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
    Corrects for:
      - Non-normality (skewness, kurtosis)
      - Sample length
      - Selection bias across multiple trials (family-wise exploration)
    Returns p-value: Prob(True SR <= 0 | Multiple Testing).
    """
    euler_mascheroni = 0.5772156649

    if nb_trials <= 1:
        expected_max_sr = 0.0
    else:
        # Expected maximum of N independent standard normals
        z_n = (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / nb_trials) + \
              euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (nb_trials * np.e))
        expected_max_sr = np.sqrt(sr_variance) * z_n

    # Asymptotic standard error under non-normality
    sr_std_err = np.sqrt(
        (1.0 + 0.5 * observed_sr**2 - skewness * observed_sr + ((kurtosis - 3.0) / 4.0) * observed_sr**2)
        / (sample_length - 1)
    )

    if sr_std_err <= 0:
        return 0.0

    z_stat = (observed_sr - expected_max_sr) / sr_std_err
    dsr_p_value = stats.norm.cdf(z_stat)
    return float(dsr_p_value)


def stationary_block_bootstrap_ci(
    returns: np.ndarray,
    num_bootstraps: int = 1500,
    block_size: int = 20,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Preserves serial autocorrelation and ARCH effects via Politis & Romano stationary block bootstrap.
    Returns (lower_bound, upper_bound) of the annualized Sharpe ratio at (1 - alpha) confidence.
    """
    rng = np.random.default_rng(random_state)
    n = len(returns)
    if n < block_size * 2:
        raise ValueError(f"Series length {n} is too short for block size {block_size}")

    p = 1.0 / block_size
    bootstrap_sharpes = np.empty(num_bootstraps)

    for b in range(num_bootstraps):
        indices = np.empty(n, dtype=int)
        cur_idx = rng.integers(0, n)
        for i in range(n):
            indices[i] = cur_idx
            if rng.random() < p:
                cur_idx = rng.integers(0, n)
            else:
                cur_idx = (cur_idx + 1) % n

        boot_sample = returns[indices]
        sample_mean = np.mean(boot_sample)
        sample_std = np.std(boot_sample, ddof=1)
        bootstrap_sharpes[b] = (sample_mean / sample_std * np.sqrt(252)) if sample_std > 0 else 0.0

    lower = float(np.percentile(bootstrap_sharpes, 100 * (alpha / 2)))
    upper = float(np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2)))
    return lower, upper


def signal_permutation_test(
    signals: np.ndarray,
    asset_returns: np.ndarray,
    num_permutations: int = 1000,
    random_state: int = 42,
) -> float:
    """
    Permutes trading signals to break alignment with future returns while
    preserving the unconditional distribution of signals and returns.
    Returns empirical p-value of the observed Sharpe ratio.
    """
    rng = np.random.default_rng(random_state)
    real_returns = signals * asset_returns
    real_sharpe = (np.mean(real_returns) / np.std(real_returns, ddof=1) * np.sqrt(252)) if np.std(real_returns) > 0 else 0.0

    perm_sharpes = np.empty(num_permutations)
    for p in range(num_permutations):
        shuffled_signals = rng.permutation(signals)
        perm_ret = shuffled_signals * asset_returns
        s_std = np.std(perm_ret, ddof=1)
        perm_sharpes[p] = (np.mean(perm_ret) / s_std * np.sqrt(252)) if s_std > 0 else 0.0

    empirical_p_val = np.mean(perm_sharpes >= real_sharpe)
    return float(empirical_p_val)
