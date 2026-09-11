"""Тесты на аналитические функции — там, где ошибку легко не заметить."""

import numpy as np
import pandas as pd

from src import analysis


def make_rets(seed=1, n=400):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, n)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {
            "KZT": base + rng.normal(0, 0.01, n),
            "RUB": base + rng.normal(0, 0.01, n),   # общий фактор -> связь есть
            "XXX": rng.normal(0, 0.01, n),          # чистый шум -> связи нет
        },
        index=idx,
    )


def test_corr_ranking_puts_the_linked_currency_first():
    corr = analysis.corr_with(make_rets(), "KZT", ["RUB", "XXX"])
    assert corr.index[0] == "RUB"
    assert corr["RUB"] > corr["XXX"]


def test_ols_recovers_known_slope():
    x = np.linspace(-1, 1, 200)
    y = 3.0 + 0.5 * x
    beta, r2 = analysis.ols(y, x.reshape(-1, 1))
    assert np.isclose(beta[0], 3.0)
    assert np.isclose(beta[1], 0.5)
    assert np.isclose(r2, 1.0)


def test_stepwise_r2_is_monotone():
    steps = analysis.stepwise_r2(make_rets(), "KZT", ["RUB", "XXX"])
    assert list(steps["добавили"])[0] == "RUB"
    assert steps["r2"].is_monotonic_increasing
    assert (steps["прирост_r2"] >= -1e-12).all()


def test_lead_lag_finds_a_planted_shift():
    """Если сдвинуть ряд на два дня, тест обязан этот сдвиг найти."""
    rng = np.random.default_rng(3)
    n = 300
    src = rng.normal(0, 0.01, n)
    idx = pd.bdate_range("2022-01-03", periods=n)
    rets = pd.DataFrame({"RUB": src, "KZT": pd.Series(src, index=idx).shift(2)}, index=idx)
    profile = analysis.lead_lag_profile(rets.dropna(), "KZT", "RUB", max_lag=4)
    assert profile.loc[profile["corr"].idxmax(), "lag"] == 2


def test_asymmetry_splits_up_and_down_days():
    s = pd.Series([0.02, -0.01, -0.01, -0.01], index=pd.bdate_range("2022-01-03", periods=4))
    stats = analysis.asymmetry_stats(pd.DataFrame({"A": s}))
    assert np.isclose(stats.loc["A", "дней_ослабления_%"], 25.0)
    assert np.isclose(stats.loc["A", "отношение_силы"], 2.0)


def test_permutation_test_finds_a_planted_effect():
    idx = pd.bdate_range("2022-01-03", periods=400)
    rng = np.random.default_rng(11)
    values = pd.Series(rng.normal(0, 0.005, len(idx)), index=idx)
    mask = pd.Series(idx.day, index=idx).between(20, 25)
    values[mask] += 0.01  # подкладываем заведомый эффект
    result = analysis.permutation_test(values, mask, n_iter=2000, seed=5)
    assert result["p_value"] < 0.01
    assert result["observed_diff"] > 0


def test_permutation_test_keeps_quiet_on_noise():
    idx = pd.bdate_range("2022-01-03", periods=400)
    rng = np.random.default_rng(12)
    values = pd.Series(rng.normal(0, 0.005, len(idx)), index=idx)
    mask = pd.Series(idx.day, index=idx).between(20, 25)
    result = analysis.permutation_test(values, mask, n_iter=2000, seed=5)
    assert result["p_value"] > 0.05
