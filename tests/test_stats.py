import numpy as np
import pytest
from stats.hypothesis import deflated_sharpe_ratio, stationary_block_bootstrap_ci, signal_permutation_test


def test_deflated_sharpe_penalizes_multiple_trials():
    """Testing 100 trials must lower statistical significance compared to 1 trial."""
    observed_sr = 1.2
    sr_var = 0.5
    skew = -0.2
    kurt = 4.0
    n = 750

    p_single = deflated_sharpe_ratio(observed_sr, sr_var, nb_trials=1, skewness=skew, kurtosis=kurt, sample_length=n)
    p_hundred = deflated_sharpe_ratio(observed_sr, sr_var, nb_trials=100, skewness=skew, kurtosis=kurt, sample_length=n)

    assert p_single > p_hundred, "DSR must penalize high trial counts"


def test_stationary_block_bootstrap_bounds():
    np.random.seed(42)
    # Drift-less white noise
    returns = np.random.normal(0.0, 0.01, size=500)
    lower, upper = stationary_block_bootstrap_ci(returns, num_bootstraps=500, block_size=10, alpha=0.05)

    assert lower < upper
    # Zero drift should have 0 inside the 95% confidence interval
    assert lower <= 0.0 <= upper, f"Confidence interval [{lower:.3f}, {upper:.3f}] does not contain 0"


def test_signal_permutation_random_distribution():
    np.random.seed(42)
    n = 300
    signals = np.random.choice([-1.0, 0.0, 1.0], size=n)
    returns = np.random.normal(0.0001, 0.01, size=n)

    p_val = signal_permutation_test(signals, returns, num_permutations=200)
    assert 0.0 <= p_val <= 1.0
