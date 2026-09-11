"""Тесты на длинный ряд: кратность, режимы, горизонты, устойчивость."""

import numpy as np
import pandas as pd

from src import analysis, clean, features


def test_tidy_long_handles_changing_quant():
    """Кратность менялась по ходу истории — делить нужно построчно."""
    raw = pd.DataFrame(
        [
            ("2013-06-03", "UZS", 1, 0.07),
            ("2016-06-03", "UZS", 100, 10.42),
            ("2016-06-03", "USD", 1, 333.29),
        ],
        columns=["date", "code", "quant", "value"],
    )
    out = clean.tidy_long(raw).set_index(["date", "code"])["rate_kzt"]
    assert out[(pd.Timestamp("2013-06-03"), "UZS")] == 0.07
    assert np.isclose(out[(pd.Timestamp("2016-06-03"), "UZS")], 0.1042)
    assert out[(pd.Timestamp("2016-06-03"), "USD")] == 333.29


def test_attach_brent_aligns_on_index():
    idx = pd.to_datetime(["2020-03-16", "2020-03-17", "2020-03-18"])
    prices = pd.DataFrame({"KZT": [420.0, 440.0, 445.0]}, index=idx)
    brent = pd.Series(
        [30.0, 28.0],
        index=pd.to_datetime(["2020-03-16", "2020-03-18"]),
    )
    out = features.attach_brent(prices, brent)
    assert out["BRENT"].iloc[0] == 30.0
    assert np.isnan(out["BRENT"].iloc[1])
    assert out["BRENT"].iloc[2] == 28.0


def test_split_by_regime_keeps_the_transition_day_out_of_both():
    """День перехода даёт первую строку нового куска, и его доходность теряется."""
    idx = pd.to_datetime(["2015-08-18", "2015-08-19", "2015-08-20", "2015-08-21"])
    prices = pd.DataFrame({"KZT": [188.0, 188.4, 255.3, 257.2]}, index=idx)
    parts = features.split_by_regime(prices)

    assert list(parts) == ["управляемый курс", "плавающий курс"]
    assert len(parts["управляемый курс"]) == 2
    assert parts["плавающий курс"].index[0] == pd.Timestamp("2015-08-20")

    float_rets = features.log_returns(parts["плавающий курс"])
    assert pd.Timestamp("2015-08-20") not in float_rets.index  # +30% никому не достаётся
    assert len(float_rets) == 1


def test_lagged_shifts_forward():
    idx = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame({"BRENT": [1.0, 2.0, 3.0]}, index=idx)
    out = features.lagged(df, "BRENT", 1)
    assert out.name == "BRENT_lag1"
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 1.0


def test_resample_prices_takes_period_close():
    idx = pd.bdate_range("2024-01-01", periods=25)
    prices = pd.DataFrame({"KZT": np.arange(25.0)}, index=idx)
    assert features.resample_prices(prices, "D").equals(prices)
    monthly = features.resample_prices(prices, "ME")
    assert monthly["KZT"].iloc[0] == prices["KZT"].loc[:"2024-01-31"].iloc[-1]


def test_rank_corr_ignores_monotone_rescaling():
    """Ранговая корреляция не должна зависеть от растяжения шкалы."""
    idx = pd.bdate_range("2024-01-01", periods=60)
    a = pd.Series(np.linspace(-1, 1, 60), index=idx)
    b = pd.Series(np.exp(np.linspace(-1, 1, 60)), index=idx)
    assert np.isclose(analysis.rank_corr(a, b), 1.0)


def test_robustness_profile_exposes_an_outlier_driven_link():
    """Связь, которая держится на одном выбросе, обязана провалить проверку."""
    idx = pd.bdate_range("2018-01-01", periods=900)
    rng = np.random.default_rng(4)

    oil = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    kzt = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    # один общий обвал: нефть рухнула, тенге рухнул
    kzt.iloc[500] = 0.30
    oil.iloc[500] = -0.60

    prices = pd.DataFrame(
        {"KZT": np.exp(kzt.cumsum()) * 400, "BRENT": np.exp(oil.cumsum()) * 60},
        index=idx,
    )
    table = analysis.robustness_profile(prices, "KZT", "BRENT", drop_year=2019)
    monthly = table.loc["месячные"]
    assert abs(monthly["Пирсон"]) > 0.4          # Пирсон видит "сильную связь"
    assert abs(monthly["Спирмен"]) < 0.25        # ранги её не подтверждают
    assert abs(monthly["без движений >40%"]) < 0.25


def test_jump_contribution_counts_the_tail():
    idx = pd.bdate_range("2024-01-01", periods=10)
    rets = pd.Series([0.10] + [0.001] * 9, index=idx)
    table = analysis.jump_contribution(rets, tops=(1,))
    assert np.isclose(table.loc[1, "вклад_%"], 10.0)
    assert table.loc[1, "доля_ослабления_%"] > 90


def test_horizon_profile_reports_every_horizon():
    idx = pd.bdate_range("2018-01-01", periods=800)
    rng = np.random.default_rng(9)
    shared = rng.normal(0, 0.01, len(idx))
    prices = pd.DataFrame(
        {
            "KZT": np.exp(np.cumsum(shared + rng.normal(0, 0.004, len(idx)))) * 400,
            "RUB": np.exp(np.cumsum(shared + rng.normal(0, 0.004, len(idx)))) * 70,
        },
        index=idx,
    )
    table = analysis.horizon_profile(prices, "KZT", ["RUB"])
    assert list(table.index) == ["дневные", "недельные", "месячные", "квартальные"]
    assert (table["RUB"] > 0.5).all()
    assert table.loc["дневные", "наблюдений"] > table.loc["месячные", "наблюдений"]


def test_excluding_jumps_volatility_drops_the_spikes():
    idx = pd.bdate_range("2009-01-01", periods=300)
    rets = pd.Series(np.full(300, 0.0005), index=idx)
    rets.iloc[100] = 0.16
    out = analysis.excluding_jumps_volatility(rets, n=1)
    assert out["волатильность_год_%"] > out["без_1_скачков_%"] * 5
    assert list(out["скачки"].values()) == [16.0]
