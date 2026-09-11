"""Аналитическая часть: с кем ходит тенге, насколько он несимметричен
и есть ли у него внутримесячный календарь.

Всё считается на лог-доходностях в базе USD (см. features.py).
Знак: плюс = валюта ослабла к доллару.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# 1. С кем ходит тенге
# --------------------------------------------------------------------------

def corr_with(rets: pd.DataFrame, target: str = "KZT", peers: list[str] | None = None) -> pd.Series:
    """Корреляция дневных доходностей target с каждой из peers."""
    peers = peers or [c for c in rets.columns if c != target]
    pairs = rets[[target] + peers].dropna()
    return pairs.corr()[target].drop(target).sort_values(ascending=False)


def corr_by_year(rets: pd.DataFrame, target: str, peers: list[str]) -> pd.DataFrame:
    """Та же корреляция, но по годам — чтобы увидеть смену режима."""
    out = {}
    for year, chunk in rets.groupby(rets.index.year):
        if len(chunk) < 60:  # неполный год не считаем
            continue
        out[year] = corr_with(chunk, target, peers)
    return pd.DataFrame(out)


def lead_lag_profile(
    rets: pd.DataFrame, target: str, other: str, max_lag: int = 5
) -> pd.DataFrame:
    """corr(target_t, other_{t-k}) для k от -max_lag до +max_lag.

    Официальный курс тенге Нацбанк устанавливает по итогам вчерашних торгов
    на KASE, а кросс-курсы берёт с мирового рынка. Из-за этого сдвиг на день
    вполне возможен — и его лучше измерить, чем игнорировать.
    """
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = rets[other].shift(lag)
        pair = pd.concat([rets[target], shifted], axis=1).dropna()
        rows.append({"lag": lag, "corr": pair.iloc[:, 0].corr(pair.iloc[:, 1]), "n": len(pair)})
    return pd.DataFrame(rows)


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """МНК с константой. Возвращает коэффициенты и R^2."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, r2


def explained_variance(rets: pd.DataFrame, target: str, regressors: list[str]) -> dict:
    """Сколько дисперсии тенге объясняет набор валют."""
    data = rets[[target] + regressors].dropna()
    beta, r2 = ols(data[target].to_numpy(), data[regressors].to_numpy())
    return {
        "regressors": regressors,
        "n": len(data),
        "r2": r2,
        "const": beta[0],
        "betas": dict(zip(regressors, beta[1:])),
    }


def stepwise_r2(rets: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    """Жадный отбор: какая валюта добавляет к объяснённой дисперсии больше всех."""
    chosen: list[str] = []
    rows = []
    remaining = list(candidates)
    while remaining:
        best, best_r2 = None, -np.inf
        for cand in remaining:
            r2 = explained_variance(rets, target, chosen + [cand])["r2"]
            if r2 > best_r2:
                best, best_r2 = cand, r2
        prev = rows[-1]["r2"] if rows else 0.0
        rows.append({"добавили": best, "r2": best_r2, "прирост_r2": best_r2 - prev})
        chosen.append(best)
        remaining.remove(best)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Асимметрия: одинаково ли валюта слабеет и крепнет
# --------------------------------------------------------------------------

def asymmetry_stats(rets: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Сравнить дни ослабления и дни укрепления."""
    cols = cols or list(rets.columns)
    rows = {}
    for col in cols:
        s = rets[col].dropna()
        down = s[s > 0]   # валюта ослабла
        up = s[s < 0]     # валюта укрепилась
        rows[col] = {
            "дней_ослабления_%": 100 * len(down) / len(s),
            "средн_ослабление_%": 100 * down.mean(),
            "средн_укрепление_%": -100 * up.mean(),
            "отношение_силы": down.mean() / -up.mean(),
            "асимметрия": s.skew(),
            "худший_день_%": 100 * s.max(),
            "лучший_день_%": -100 * s.min(),
        }
    return pd.DataFrame(rows).T.round(3)


def drawdown_profile(prices: pd.Series) -> pd.DataFrame:
    """Просадка валюты к доллару и скорость возврата."""
    peak = prices.cummin()          # минимум цены = самый крепкий курс
    loss = prices / peak - 1.0      # насколько валюта слабее своего максимума
    return pd.DataFrame({"price": prices, "best": peak, "below_best": loss})


def ratchet_ratio(prices: pd.Series, window: int = 60) -> pd.DataFrame:
    """Сколько дней валюта проводит вблизи своего годового минимума слабости.

    Идея простая: если курс двигается симметрично, он примерно поровну
    времени проводит в верхней и нижней половине своего диапазона.
    Валюта-храповик залипает в слабой половине.
    """
    roll_min = prices.rolling(window, min_periods=window).min()
    roll_max = prices.rolling(window, min_periods=window).max()
    pos = (prices - roll_min) / (roll_max - roll_min)
    return pd.DataFrame({"position_in_range": pos})


# --------------------------------------------------------------------------
# 3. Внутримесячный календарь
# --------------------------------------------------------------------------

def intramonth_profile(rets: pd.Series) -> pd.DataFrame:
    """Средняя доходность по числам месяца."""
    df = pd.DataFrame({"ret": rets.dropna()})
    df["day"] = df.index.day
    g = df.groupby("day")["ret"]
    out = pd.DataFrame(
        {
            "средн_%": g.mean() * 100,
            "медиана_%": g.median() * 100,
            "наблюдений": g.size(),
            "доля_ослабления_%": g.apply(lambda s: 100 * (s > 0).mean()),
        }
    )
    return out.round(4)


def permutation_test(
    values: pd.Series, mask: pd.Series, n_iter: int = 20000, seed: int = 7
) -> dict:
    """Перестановочный тест разницы средних между двумя группами дней.

    Обычный t-тест на финансовых рядах врёт из-за толстых хвостов, а scipy
    ради одной функции тянуть не хочется. Перемешиваем метки групп и
    смотрим, часто ли случайность даёт разницу не меньше наблюдаемой.
    """
    v = values.dropna()
    m = mask.reindex(v.index).astype(bool)
    observed = v[m].mean() - v[~m].mean()

    rng = np.random.default_rng(seed)
    arr = v.to_numpy()
    k = int(m.sum())
    diffs = np.empty(n_iter)
    total = arr.sum()
    for i in range(n_iter):
        idx = rng.choice(len(arr), size=k, replace=False)
        group = arr[idx].sum()
        diffs[i] = group / k - (total - group) / (len(arr) - k)

    p_value = float((np.abs(diffs) >= abs(observed)).mean())
    return {
        "observed_diff": float(observed),
        "group_mean": float(v[m].mean()),
        "rest_mean": float(v[~m].mean()),
        "n_group": k,
        "n_rest": int(len(v) - k),
        "p_value": p_value,
    }


def weekday_profile(rets: pd.Series) -> pd.DataFrame:
    names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    df = pd.DataFrame({"ret": rets.dropna()})
    df["wd"] = df.index.dayofweek
    g = df.groupby("wd")["ret"]
    out = pd.DataFrame({"средн_%": g.mean() * 100, "наблюдений": g.size()})
    out.index = [names[i] for i in out.index]
    return out.round(4)

# --------------------------------------------------------------------------
# 4. Горизонт и режим: то, ради чего нужен длинный ряд
# --------------------------------------------------------------------------

HORIZONS = [("D", "дневные"), ("W", "недельные"), ("ME", "месячные"), ("QE", "квартальные")]


def _returns_at(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Доходности на заданном горизонте.

    На дневном горизонте обязательно через features.log_returns: он зануляет
    доходность через длинные праздники и дыры в архиве. Иначе один Наурыз
    подмешивает в дневную выборку недельное движение, и дневная корреляция
    оказывается завышенной. На недельном и выше ресемплинг такие разрывы
    поглощает сам.
    """
    from src import features

    if freq == "D":
        return features.log_returns(prices).dropna()
    return np.log(features.resample_prices(prices, freq)).diff().dropna()


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    """Ранговая корреляция Спирмена, посчитанная руками.

    Тащить scipy ради одной функции не хочется, а корреляция Спирмена —
    это обычный Пирсон на рангах. Нужна она здесь не для красоты: Пирсон
    на месячных доходностях очень легко утаскивается парой катастрофических
    месяцев, а ранги на величину движения не смотрят.
    """
    pair = pd.concat([a, b], axis=1).dropna()
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))


def horizon_profile(
    prices: pd.DataFrame,
    target: str,
    others: list[str],
    horizons: list[tuple[str, str]] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Корреляция target с каждой из others на разных горизонтах.

    Смысл упражнения простой: связь может существовать, но не на том
    масштабе, на котором её ищут. Нефть влияет на тенге через торговый
    баланс и бюджет — это медленный канал, и в дневных доходностях его
    почти не видно.
    """
    horizons = horizons or HORIZONS
    rows = {}
    for freq, label in horizons:
        rets = _returns_at(prices[[target] + others], freq)
        if method == "spearman":
            row = {code: rank_corr(rets[target], rets[code]) for code in others}
        else:
            row = {code: rets[target].corr(rets[code]) for code in others}
        row["наблюдений"] = len(rets)
        rows[label] = row
    return pd.DataFrame(rows).T


def robustness_profile(
    prices: pd.DataFrame,
    target: str,
    other: str,
    horizons: list[tuple[str, str]] | None = None,
    extreme: float = 0.4,
    drop_year: int = 2020,
) -> pd.DataFrame:
    """Одна и та же связь, посчитанная четырьмя способами.

    Если корреляция держится только на Пирсоне и разваливается на рангах
    или после удаления пары аномальных периодов — значит, это не устойчивая
    связь, а несколько совместных катастроф.
    """
    horizons = horizons or HORIZONS
    rows = {}
    for freq, label in horizons:
        rets = _returns_at(prices[[target, other]], freq)
        calm = rets[rets[other].abs() < extreme]
        without_year = rets[rets.index.year != drop_year]
        rows[label] = {
            "Пирсон": rets[target].corr(rets[other]),
            "Спирмен": rank_corr(rets[target], rets[other]),
            f"без движений >{extreme:.0%}": calm[target].corr(calm[other]),
            f"без {drop_year} года": without_year[target].corr(without_year[other]),
            "наблюдений": len(rets),
        }
    return pd.DataFrame(rows).T


def regime_table(frames: dict[str, pd.DataFrame], target: str, others: list[str]) -> pd.DataFrame:
    """Корреляции и волатильность по эпохам курсовой политики.

    На вход идут уже нарезанные куски (features.split_by_regime), а не общий
    ряд с метками: иначе доходность дня перехода попадает в новый режим
    и портит обе цифры сразу.
    """
    rows = {}
    for name, rets in frames.items():
        data = rets[[target] + others].dropna()
        series = rets[target].dropna()
        row = {f"corr {code}": data[target].corr(data[code]) for code in others}
        row["волатильность_год_%"] = series.std() * np.sqrt(TRADING_DAYS) * 100
        row["дней_без_движения_%"] = 100 * float((series.abs() < 1e-9).mean())
        row["наблюдений"] = len(data)
        rows[name] = row
    return pd.DataFrame(rows).T


def jump_contribution(rets: pd.Series, tops: tuple[int, ...] = (1, 3, 5, 10)) -> pd.DataFrame:
    """Сколько от всего ослабления валюты дают несколько худших дней."""
    series = rets.dropna()
    total = series.sum()
    rows = []
    for n in tops:
        worst = series.nlargest(n)
        rows.append(
            {
                "дней": n,
                "доля_дней_%": 100 * n / len(series),
                "вклад_%": 100 * worst.sum(),
                "доля_ослабления_%": 100 * worst.sum() / total if total else np.nan,
                "даты": ", ".join(d.date().isoformat() for d in worst.index),
            }
        )
    return pd.DataFrame(rows).set_index("дней")


def excluding_jumps_volatility(rets: pd.Series, n: int = 3) -> dict:
    """Волатильность с разовыми скачками и без них.

    До 2015 года у тенге почти не было волатильности в обычном смысле:
    курс стоял неделями, а потом разом переставлялся решением регулятора.
    Среднеквадратичное отклонение в таком режиме описывает не рынок,
    а два-три дня из тысячи.
    """
    series = rets.dropna()
    worst = series.nlargest(n)
    return {
        "волатильность_год_%": float(series.std() * np.sqrt(TRADING_DAYS) * 100),
        f"без_{n}_скачков_%": float(series.drop(worst.index).std() * np.sqrt(TRADING_DAYS) * 100),
        "скачки": {d.date().isoformat(): round(float(v) * 100, 1) for d, v in worst.items()},
    }
