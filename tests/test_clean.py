"""Тесты на очистку — на маленьких таблицах, собранных вручную."""

import pandas as pd

from src import clean


def make_raw(rows):
    return pd.DataFrame(rows, columns=["date", "code", "name_ru", "quant", "value", "change"])


def test_quant_normalisation():
    """10 драмов за 12.66 тенге -> 1.266 тенге за драм."""
    raw = make_raw(
        [
            ("2024-01-09", "AMD", "АРМЯНСКИЙ ДРАМ", "10", "12.66", "+0.01"),
            ("2024-01-09", "USD", "ДОЛЛАР США", "1", "450.00", "-1.00"),
        ]
    )
    out = clean.tidy(raw).set_index("code")["rate_kzt"]
    assert out["AMD"] == 1.266
    assert out["USD"] == 450.0


def test_broken_rows_are_dropped():
    raw = make_raw(
        [
            ("2024-01-09", "USD", "ДОЛЛАР США", "1", "450.00", ""),
            ("2024-01-09", "XXX", "БИТАЯ", "0", "10", ""),
            ("2024-01-09", "YYY", "ТОЖЕ БИТАЯ", "1", "н/д", ""),
        ]
    )
    assert list(clean.tidy(raw)["code"]) == ["USD"]


def test_duplicates_keep_last():
    raw = make_raw(
        [
            ("2024-01-09", "USD", "ДОЛЛАР США", "1", "450.00", ""),
            ("2024-01-09", "USD", "ДОЛЛАР США", "1", "451.00", ""),
        ]
    )
    out = clean.tidy(raw)
    assert len(out) == 1
    assert out["rate_kzt"].iloc[0] == 451.0


def test_carryover_days_are_removed():
    """Выходной = курс в точности повторяет предыдущий день."""
    idx = pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"])
    wide = pd.DataFrame({"USD": [450.0, 450.0, 450.0, 452.0], "EUR": [490.0, 490.0, 490.0, 493.0]}, index=idx)
    kept, dropped = clean.drop_non_trading_days(wide)
    assert list(kept.index.strftime("%Y-%m-%d")) == ["2024-01-05", "2024-01-08"]
    assert len(dropped) == 2


def test_first_day_is_never_treated_as_carryover():
    idx = pd.to_datetime(["2024-01-05", "2024-01-08"])
    wide = pd.DataFrame({"USD": [450.0, 452.0]}, index=idx)
    kept, dropped = clean.drop_non_trading_days(wide)
    assert len(kept) == 2
    assert dropped.empty


def test_snapshot_echo_is_removed():
    """Строка из прошлого, дословно совпадающая с концом выборки, — заглушка."""
    idx = pd.to_datetime(["2021-05-07", "2021-05-08", "2021-05-09", "2026-09-10"])
    wide = pd.DataFrame(
        {
            "USD": [456.89, 426.99, 427.22, 456.89],
            "EUR": [531.36, 512.10, 513.40, 531.36],
            "RUB": [5.32, 5.75, 5.74, 5.32],
        },
        index=idx,
    )
    out = clean.drop_snapshot_echoes(wide, lookback=1, min_match=3)
    assert pd.Timestamp("2021-05-07") not in out.index
    assert pd.Timestamp("2026-09-10") in out.index
    assert len(out) == 3


def test_gaps_are_reported():
    idx = pd.to_datetime(["2023-07-25", "2023-07-27", "2023-08-22"])
    wide = pd.DataFrame({"USD": [440.0, 441.0, 452.0]}, index=idx)
    gaps = clean.report_gaps(wide, min_len=7)
    assert gaps == [("2023-07-27", "2023-08-22", 25)]
