"""Приведение курсов к одной базе и расчёт доходностей.

Нацбанк котирует всё в тенге: столько-то KZT за доллар, за рубль, за юань.
Сравнивать такие ряды между собой бессмысленно — в каждом из них уже сидит
сам тенге, и любые две пары будут коррелировать просто потому, что у них
общий знаменатель.

Поэтому пересчитываем всё в доллары:

    X за 1 USD = (KZT за 1 USD) / (KZT за 1 X)

Тенге при этом остаётся как есть — это и есть KZT за 1 USD. Дальше работаем
с логарифмическими доходностями: плюс означает ослабление валюты к доллару.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# Максимальный разрыв между соседними торговыми днями, который мы готовы
# считать "одним шагом". Через 25-дневную дыру в архиве Нацбанка доходность
# считать нельзя — это будет не дневное движение, а месячное.
MAX_GAP_DAYS = 5


def to_usd_base(
    wide: pd.DataFrame,
    currencies: list[str] | None = None,
    base: str = config.BASE,
) -> pd.DataFrame:
    """Перевести котировки из тенге в доллар: сколько единиц валюты за 1 USD."""
    if base not in wide.columns:
        raise KeyError(f"в данных нет базовой валюты {base}")

    currencies = currencies or config.PEERS
    base_price = wide[base]  # KZT за 1 USD

    out = pd.DataFrame(index=wide.index)
    out["KZT"] = base_price
    for code in currencies:
        if code == base:
            continue
        if code not in wide.columns:
            print(f"  пропускаю {code}: нет в данных")
            continue
        out[code] = base_price / wide[code]

    return out


def log_returns(prices: pd.DataFrame, max_gap_days: int = MAX_GAP_DAYS) -> pd.DataFrame:
    """Дневные лог-доходности. Через дыры в календаре доходность не тянем."""
    rets = np.log(prices).diff()

    gap = prices.index.to_series().diff().dt.days
    too_far = gap > max_gap_days
    if too_far.any():
        dropped = [d.date().isoformat() for d in prices.index[too_far.to_numpy()]]
        print(f"  обнуляю доходность через разрывы календаря: {dropped}")
        rets.loc[too_far.to_numpy()] = np.nan

    return rets.iloc[1:]


def to_trading_date(df: pd.DataFrame) -> pd.DataFrame:
    """Сдвинуть индекс с даты публикации курса на дату торгов.

    Это не догадка, а наблюдение из самих данных: если посмотреть, по каким
    дням недели курс вообще меняется, окажется, что понедельник почти всегда
    повторяет пятницу, зато суббота — нет. То есть курс, объявленный на
    дату D, посчитан по торгам дня D-1. Без этой поправки календарные
    выводы (день недели, число месяца) считаются не по тем дням.
    """
    out = df.copy()
    out.index = out.index - pd.Timedelta(days=1)
    out.index.name = "trade_date"
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Календарные признаки — понадобятся для внутримесячной сезонности."""
    out = df.copy()
    out["weekday"] = out.index.dayofweek  # 0 = понедельник
    out["day_of_month"] = out.index.day
    out["month"] = out.index.month
    out["year"] = out.index.year
    # Налоговая неделя в РК: основные платежи приходятся на 20-25 число,
    # под них экспортёры продают валютную выручку.
    out["tax_week"] = out["day_of_month"].between(20, 25)
    return out


def attach_brent(prices: pd.DataFrame, brent: pd.Series, name: str = "BRENT") -> pd.DataFrame:
    """Добавить нефть в панель, выровняв её по торговым дням курсов.

    Нефть намеренно кладётся как есть, без сдвигов: правильное выравнивание
    во времени — это результат, а не предпосылка, и его считает
    analysis.lead_lag_profile().
    """
    out = prices.copy()
    out[name] = brent.reindex(out.index)
    filled = out[name].notna().mean()
    print(f"  нефть покрывает {filled:.1%} торговых дней")
    return out


def lagged(df: pd.DataFrame, column: str, lag: int) -> pd.Series:
    """Сдвинутая копия колонки: lag=1 означает «значение предыдущего дня»."""
    return df[column].shift(lag).rename(f"{column}_lag{lag}")


def regime(index: pd.DatetimeIndex, float_date=None) -> pd.Series:
    """Разметить наблюдения на две эпохи курсовой политики.

    20 августа 2015 года Нацбанк отпустил тенге в свободное плавание.
    До этой даты курс был решением, после — ценой, и мерить их одной
    линейкой бессмысленно.
    """
    float_date = pd.Timestamp(float_date or config.FLOAT_DATE)
    labels = pd.Series("плавающий курс", index=index)
    labels[index < float_date] = "управляемый курс"
    return labels


def split_by_regime(prices: pd.DataFrame, float_date=None) -> dict[str, pd.DataFrame]:
    """Разрезать панель цен на две эпохи курсовой политики.

    Резать надо именно цены, а не доходности: если поделить уже посчитанные
    доходности, то в эпоху плавания попадёт доходность самого дня перехода —
    те самые +30% 20 августа 2015 года. А это не поведение нового режима,
    а последнее действие старого. Разрезав цены, мы теряем первую доходность
    каждого куска, и день перехода честно не достаётся никому.
    """
    float_date = pd.Timestamp(float_date or config.FLOAT_DATE)
    return {
        "управляемый курс": prices[prices.index < float_date],
        "плавающий курс": prices[prices.index >= float_date],
    }


def resample_prices(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Последнее значение в периоде — так считается доходность за неделю/месяц."""
    if freq == "D":
        return prices
    return prices.resample(freq).last().dropna(how="all")


def rolling_corr(rets: pd.DataFrame, a: str, b: str, window: int = 90) -> pd.Series:
    """Скользящая корреляция двух валют."""
    return rets[a].rolling(window, min_periods=window // 2).corr(rets[b])


def summary_table(rets: pd.DataFrame) -> pd.DataFrame:
    """Базовая статистика доходностей в годовом выражении."""
    stats = pd.DataFrame(
        {
            "наблюдений": rets.count(),
            "среднее_год_%": rets.mean() * 252 * 100,
            "волатильность_год_%": rets.std() * np.sqrt(252) * 100,
            "асимметрия": rets.skew(),
            "эксцесс": rets.kurt(),
            "макс_дневное_падение_%": rets.max() * 100,
            "макс_дневной_рост_%": rets.min() * 100,
        }
    )
    return stats.round(2)
