"""Тесты на пересчёт в доллары и доходности."""

import numpy as np
import pandas as pd

from src import features


def make_wide():
    idx = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"])
    return pd.DataFrame(
        {
            "USD": [450.0, 455.0, 460.0],
            "RUB": [5.00, 5.00, 4.60],
            "EUR": [495.0, 500.0, 506.0],
        },
        index=idx,
    )


def test_usd_base_math():
    """90 рублей за доллар = (450 тенге за доллар) / (5 тенге за рубль)."""
    out = features.to_usd_base(make_wide(), currencies=["RUB", "EUR"])
    assert out["KZT"].iloc[0] == 450.0
    assert out["RUB"].iloc[0] == 90.0
    assert round(out["RUB"].iloc[2], 4) == 100.0


def test_missing_currency_is_skipped():
    out = features.to_usd_base(make_wide(), currencies=["RUB", "JPY"])
    assert list(out.columns) == ["KZT", "RUB"]


def test_base_must_be_present():
    wide = make_wide().drop(columns=["USD"])
    try:
        features.to_usd_base(wide, currencies=["RUB"])
    except KeyError:
        return
    raise AssertionError("ожидали KeyError на отсутствующую базовую валюту")


def test_log_returns_sign():
    """Тенге подешевел с 450 до 455 — доходность положительная."""
    prices = features.to_usd_base(make_wide(), currencies=["RUB"])
    rets = features.log_returns(prices)
    assert rets["KZT"].iloc[0] > 0
    assert np.isclose(rets["KZT"].iloc[0], np.log(455 / 450))
    assert len(rets) == 2


def test_gap_returns_are_dropped():
    idx = pd.to_datetime(["2023-07-26", "2023-07-27", "2023-08-22"])
    prices = pd.DataFrame({"KZT": [440.0, 441.0, 452.0]}, index=idx)
    rets = features.log_returns(prices, max_gap_days=5)
    assert not np.isnan(rets["KZT"].iloc[0])
    assert np.isnan(rets["KZT"].iloc[1])


def test_trading_date_shift():
    idx = pd.to_datetime(["2024-01-09", "2024-01-10"])
    df = pd.DataFrame({"KZT": [0.01, -0.01]}, index=idx)
    out = features.to_trading_date(df)
    assert list(out.index.strftime("%Y-%m-%d")) == ["2024-01-08", "2024-01-09"]


def test_calendar_flags_tax_week():
    idx = pd.to_datetime(["2024-01-19", "2024-01-22", "2024-01-26"])
    df = pd.DataFrame({"KZT": [0.0, 0.0, 0.0]}, index=idx)
    out = features.add_calendar(df)
    assert list(out["tax_week"]) == [False, True, False]
