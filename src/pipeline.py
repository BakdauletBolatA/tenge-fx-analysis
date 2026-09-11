"""Полный прогон: сырые данные -> чистая таблица -> цифры -> графики.

    python -m src.pipeline

Загрузку данных сюда не тащим специально: она ходит в сеть и занимает
несколько минут, поэтому живёт отдельной командой (python -m src.fetch_nbk).
"""

from __future__ import annotations

import json

import pandas as pd

from src import analysis as A
from src import clean, config, features, plots

TABLES_DIR = config.REPORTS_DIR / "tables"


def _save_table(df: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / name)
    print(f"  reports/tables/{name}")


def run() -> dict:
    print("[1/4] очистка")
    wide = clean.build()

    print("[2/4] признаки")
    prices = features.to_usd_base(wide)
    rets = features.log_returns(prices)
    rets_trade = features.to_trading_date(rets)
    peers = [c for c in rets.columns if c != "KZT"]

    print("[3/4] расчёты")
    corr = A.corr_with(rets, "KZT", peers)
    by_year = A.corr_by_year(rets, "KZT", peers)
    lead_lag = A.lead_lag_profile(rets, "KZT", "RUB", max_lag=5)
    steps = A.stepwise_r2(rets, "KZT", ["RUB", "NOK", "EUR", "CNY", "CAD", "GEL", "INR", "ZAR"])
    asym = A.asymmetry_stats(rets, ["KZT", "RUB", "CNY", "EUR", "UZS", "TRY", "NOK"])
    intramonth = A.intramonth_profile(rets_trade["KZT"])
    weekdays = A.weekday_profile(rets_trade["KZT"])

    tax_mask = pd.Series(rets_trade.index.day, index=rets_trade.index).between(20, 25)
    tax_test = A.permutation_test(rets_trade["KZT"], tax_mask)

    _save_table(corr.rename("corr_with_kzt").to_frame(), "corr_with_kzt.csv")
    _save_table(by_year, "corr_by_year.csv")
    _save_table(lead_lag.set_index("lag"), "lead_lag_kzt_rub.csv")
    _save_table(steps.set_index("добавили"), "stepwise_r2.csv")
    _save_table(asym, "asymmetry.csv")
    _save_table(intramonth, "intramonth_profile.csv")
    _save_table(weekdays, "weekday_profile.csv")

    print("[4/4] графики")
    plots.fig_levels(prices)
    plots.fig_corr_ranking(corr)
    plots.fig_rolling_corr(rets)
    plots.fig_conditional_corr(rets)
    plots.fig_scatter(rets)
    plots.fig_corr_by_year(by_year.loc[["RUB", "NOK", "EUR", "CNY", "TRY", "UZS", "KGS", "GEL"]])
    plots.fig_tail_contribution(rets)
    plots.fig_return_distribution(rets)
    plots.fig_intramonth(rets_trade)

    kzt = rets["KZT"].dropna()
    worst10 = kzt.nlargest(10)
    summary = {
        "период": [str(prices.index[0].date()), str(prices.index[-1].date())],
        "торговых_дней": int(len(prices)),
        "валют_в_выборке": int(wide.shape[1]),
        "тенге_к_доллару_итог_%": round(100 * (prices["KZT"].iloc[-1] / prices["KZT"].iloc[0] - 1), 2),
        "рубль_к_доллару_итог_%": round(100 * (prices["RUB"].iloc[-1] / prices["RUB"].iloc[0] - 1), 2),
        "corr_kzt_rub": round(float(corr["RUB"]), 3),
        "следующий_за_рублём": corr.index[1],
        "corr_второго_места": round(float(corr.iloc[1]), 3),
        "r2_только_рубль": round(A.explained_variance(rets, "KZT", ["RUB"])["r2"], 4),
        "r2_все_валюты": round(float(steps["r2"].iloc[-1]), 4),
        "beta_rub": round(A.explained_variance(rets, "KZT", ["RUB"])["betas"]["RUB"], 3),
        "лучший_лаг": int(lead_lag.loc[lead_lag["corr"].idxmax(), "lag"]),
        "corr_2026": round(float(by_year.loc["RUB", by_year.columns[-1]]), 3),
        "доля_дней_ослабления_%": round(100 * float((kzt > 0).mean()), 1),
        "асимметрия_тенге": round(float(kzt.skew()), 2),
        "вклад_10_худших_дней_%": round(100 * float(worst10.sum()), 1),
        "итог_без_10_худших_%": round(100 * float(kzt.drop(worst10.index).sum()), 1),
        "налоговая_неделя_p_value": round(tax_test["p_value"], 3),
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nКлючевые цифры:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


if __name__ == "__main__":
    run()
