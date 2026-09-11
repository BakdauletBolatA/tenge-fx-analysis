"""Полный прогон: сырые данные -> чистая таблица -> цифры -> графики.

    python -m src.pipeline            # обе части
    python -m src.pipeline --only short
    python -m src.pipeline --only long

Часть первая (short) — широкая выборка: 48 валют, но только с мая 2021 года.
Отвечает на вопрос «с кем тенге ходит в одну ногу».

Часть вторая (long) — длинный ряд с 2005 года плюс нефть. Валют меньше,
зато в выборку попадают девальвации 2009 и 2014 годов и переход к плаванию
в августе 2015-го. Отвечает на вопросы «рубль или нефть» и «что изменил
плавающий курс».

Загрузку данных сюда не тащим специально: она ходит в сеть и занимает
несколько минут, поэтому живёт отдельными командами (fetch_nbk,
fetch_nbk_archive, fetch_brent).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
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


def run_long() -> dict:
    """Длинный ряд: режимы курсовой политики и нефть."""
    print("[1/4] очистка длинного ряда")
    wide = clean.build_long()
    brent = clean.load_brent()

    print("[2/4] признаки")
    codes = [c for c in wide.columns if c != "USD"]
    prices = features.attach_brent(
        features.to_trading_date(features.to_usd_base(wide, currencies=codes)), brent
    )
    rets = features.log_returns(prices)
    labels = features.regime(rets.index)
    float_date = pd.Timestamp(config.FLOAT_DATE)

    print("[3/4] расчёты")
    peers = ["RUB", "BRENT", "CNY", "EUR", "NOK", "TRY", "UZS", "KGS"]
    horizons_all = A.horizon_profile(prices, "KZT", peers)
    horizons_pre = A.horizon_profile(prices[prices.index < float_date], "KZT", peers)
    horizons_post = A.horizon_profile(prices[prices.index >= float_date], "KZT", peers)
    horizons_post_rank = A.horizon_profile(
        prices[prices.index >= float_date], "KZT", peers, method="spearman"
    )
    robust_oil = A.robustness_profile(prices[prices.index >= float_date], "KZT", "BRENT")
    robust_rub = A.robustness_profile(prices[prices.index >= float_date], "KZT", "RUB")
    regimes = A.regime_table(rets, labels, "KZT", ["RUB", "BRENT", "CNY", "EUR"])
    brent_lag = A.lead_lag_profile(rets[rets.index >= float_date], "KZT", "BRENT", max_lag=4)
    jumps = A.jump_contribution(rets["KZT"])
    vol_pre = A.excluding_jumps_volatility(rets.loc[rets.index < float_date, "KZT"], n=3)

    # Регрессии: дневные и месячные, только эпоха плавания.
    post = prices[prices.index >= float_date]
    post_rets = features.log_returns(post)
    post_rets["BRENT_lag1"] = features.lagged(post_rets, "BRENT", 1)
    monthly = np.log(features.resample_prices(post, "ME")).diff().dropna()
    monthly["BRENT_lag1"] = monthly["BRENT"]

    models = []
    for label, frame in (("дневные", post_rets), ("месячные", monthly)):
        for regressors in (["BRENT_lag1"], ["RUB"], ["BRENT_lag1", "RUB"]):
            fit = A.explained_variance(frame, "KZT", regressors)
            models.append(
                {
                    "горизонт": label,
                    "модель": " + ".join(regressors),
                    "n": fit["n"],
                    "r2": round(fit["r2"], 4),
                    **{f"beta_{k}": round(v, 4) for k, v in fit["betas"].items()},
                }
            )
    models = pd.DataFrame(models)

    _save_table(horizons_all, "horizons_all.csv")
    _save_table(horizons_pre, "horizons_managed.csv")
    _save_table(horizons_post, "horizons_float.csv")
    _save_table(horizons_post_rank, "horizons_float_spearman.csv")
    _save_table(robust_oil, "robustness_brent.csv")
    _save_table(robust_rub, "robustness_rouble.csv")
    _save_table(regimes, "regime_comparison.csv")
    _save_table(brent_lag.set_index("lag"), "lead_lag_kzt_brent.csv")
    _save_table(jumps, "jump_contribution.csv")
    _save_table(models.set_index(["горизонт", "модель"]), "oil_vs_rouble_models.csv")

    print("[4/4] графики")
    plots.fig_long_history(prices)
    plots.fig_horizon(horizons_post)
    plots.fig_regimes(horizons_pre, horizons_post)
    plots.fig_long_jumps(rets["KZT"])
    plots.fig_brent_scatter(monthly)
    plots.fig_robustness(robust_oil, robust_rub)

    kzt = rets["KZT"].dropna()
    top3 = kzt.nlargest(3)
    summary = {
        "период": [str(prices.index[0].date()), str(prices.index[-1].date())],
        "торговых_дней": int(len(prices)),
        "валют_в_выборке": int(wide.shape[1]),
        "курс_начало": round(float(prices["KZT"].iloc[0]), 2),
        "курс_конец": round(float(prices["KZT"].iloc[-1]), 2),
        "corr_нефть_дневные": round(float(horizons_post.loc["дневные", "BRENT"]), 3),
        "corr_нефть_квартальные": round(float(horizons_post.loc["квартальные", "BRENT"]), 3),
        "corr_рубль_дневные": round(float(horizons_post.loc["дневные", "RUB"]), 3),
        "corr_рубль_квартальные": round(float(horizons_post.loc["квартальные", "RUB"]), 3),
        "corr_нефть_до_плавания_квартальные": round(float(horizons_pre.loc["квартальные", "BRENT"]), 3),
        "corr_рубль_до_плавания_дневные": round(float(horizons_pre.loc["дневные", "RUB"]), 3),
        "лучший_лаг_нефти": int(brent_lag.loc[brent_lag["corr"].idxmin(), "lag"]),
        "волатильность_управляемый_%": round(float(regimes.loc["управляемый курс", "волатильность_год_%"]), 1),
        "волатильность_плавающий_%": round(float(regimes.loc["плавающий курс", "волатильность_год_%"]), 1),
        "волатильность_управляемый_без_скачков_%": round(vol_pre["без_3_скачков_%"], 1),
        "дней_без_движения_управляемый_%": round(float(regimes.loc["управляемый курс", "дней_без_движения_%"]), 1),
        "дней_без_движения_плавающий_%": round(float(regimes.loc["плавающий курс", "дней_без_движения_%"]), 1),
        "вклад_3_худших_дней_%": round(float(jumps.loc[3, "доля_ослабления_%"]), 1),
        "вклад_10_худших_дней_%": round(float(jumps.loc[10, "доля_ослабления_%"]), 1),
        "три_худших_дня": {d.date().isoformat(): round(float(v) * 100, 1) for d, v in top3.items()},
        "r2_месячные_нефть": float(models.query("горизонт == 'месячные' and модель == 'BRENT_lag1'")["r2"].iloc[0]),
        "r2_месячные_рубль": float(models.query("горизонт == 'месячные' and модель == 'RUB'")["r2"].iloc[0]),
        "r2_месячные_вместе": float(models.query("горизонт == 'месячные' and модель == 'BRENT_lag1 + RUB'")["r2"].iloc[0]),
        "нефть_квартальные_спирмен": round(float(horizons_post_rank.loc["квартальные", "BRENT"]), 3),
        "нефть_квартальные_без_экстремумов": round(float(robust_oil.loc["квартальные", "без движений >40%"]), 3),
        "рубль_квартальные_спирмен": round(float(horizons_post_rank.loc["квартальные", "RUB"]), 3),
        "рубль_квартальные_без_экстремумов": round(float(robust_rub.loc["квартальные", "без движений >40%"]), 3),
    }

    (config.REPORTS_DIR / "summary_long.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nКлючевые цифры длинного ряда:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогнать расчёты и собрать графики")
    parser.add_argument("--only", choices=["short", "long"], default=None)
    args = parser.parse_args()

    if args.only != "long":
        run()
    if args.only != "short":
        print()
        run_long()


if __name__ == "__main__":
    main()
