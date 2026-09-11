"""Очистка сырых курсов: типы, нормировка на кратность, широкая таблица.

Что пришлось разгребать в данных Нацбанка:

1. Курс публикуется не за одну единицу валюты, а за `quant` штук
   (10 драмов, 100 узбекских сумов, 1000 вьетнамских донгов). Без деления
   на quant сравнение валют превращается в сравнение масштабов.
2. В выходные и праздники официальный курс не устанавливается, а просто
   повторяет предыдущий рабочий день. Если оставить такие дни, в
   доходностях появятся искусственные нули, и корреляции поедут вниз.
3. На даты вне своего архива эндпоинт отдаёт не ошибку, а СЕГОДНЯШНИЙ
   снимок курсов. Выглядит это как совершенно нормальная строка данных —
   и молча портит начало ряда.
4. Список котируемых валют менялся: с 2021 по июль 2026 их было 39,
   потом стало 48. Поэтому широкая таблица дырявая справа.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import config


def load_raw(path=None) -> pd.DataFrame:
    path = path or config.RAW_RATES_CSV
    df = pd.read_csv(path, dtype=str)
    print(f"Прочитал {len(df)} строк из {path.name}")
    return df


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Привести сырые строки к типам и посчитать курс за одну единицу."""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["quant"] = pd.to_numeric(df["quant"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    bad = df["value"].isna() | df["quant"].isna() | (df["quant"] <= 0)
    if bad.any():
        print(f"  выбрасываю {int(bad.sum())} строк с битыми числами")
        df = df[~bad]

    # KZT за одну единицу иностранной валюты
    df["rate_kzt"] = df["value"] / df["quant"]

    dupes = df.duplicated(subset=["date", "code"], keep="last")
    if dupes.any():
        print(f"  выбрасываю {int(dupes.sum())} дублей date+code")
        df = df[~dupes]

    return df[["date", "code", "name_ru", "rate_kzt"]].sort_values(["date", "code"])


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Дата в индексе, валюты в колонках."""
    wide = long.pivot(index="date", columns="code", values="rate_kzt").sort_index()
    wide.columns.name = None
    return wide


def drop_non_trading_days(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Убрать дни, в которые курс не изменился ни по одной валюте.

    Ловим выходные и праздники не по календарю (казахстанские праздники
    плавают и переносятся), а по самому признаку: нулевое изменение сразу
    по четырём десяткам валют — событие, которого на живом рынке не бывает.
    """
    changed = wide.diff().abs().sum(axis=1, skipna=True)
    carried = changed.eq(0)
    carried.iloc[0] = False  # для первого дня diff не определён
    dropped = wide.index[carried]
    print(f"  убираю {len(dropped)} дней без изменения курсов (выходные/праздники)")
    return wide.loc[~carried], dropped


def drop_snapshot_echoes(wide: pd.DataFrame, lookback: int = 10, min_match: int = 20) -> pd.DataFrame:
    """Убрать строки-заглушки: копии актуального снимка курсов.

    Если попросить курс на дату, которой в архиве нет, сервис иногда молча
    отдаёт последний опубликованный снимок вместо ошибки. Внутри выборки
    такая строка выглядит нормально — но совпадает с одним из последних
    дней выборки до шестого знака сразу по двум десяткам валют. На живом
    рынке такого не бывает.
    """
    tail = wide.tail(lookback)
    if tail.empty:
        return wide

    flagged = pd.Series(False, index=wide.index)
    for _, snapshot in tail.iterrows():
        values = snapshot.dropna()
        if len(values) < min_match:
            continue
        same = wide[values.index].eq(values).sum(axis=1).ge(min_match)
        same.loc[wide.index >= tail.index[0]] = False  # сам хвост не трогаем
        flagged |= same

    if flagged.any():
        dates = [d.date().isoformat() for d in wide.index[flagged]]
        print(f"  убираю {int(flagged.sum())} строк-заглушек: {dates}")
    return wide.loc[~flagged]


def report_gaps(wide: pd.DataFrame, min_len: int = 7) -> list[tuple[str, str, int]]:
    """Найти дыры в календаре.

    Длинные выходные уже отфильтрованы, так что всё, что длиннее недели, —
    это дыра в самом архиве Нацбанка.
    """
    idx = wide.index
    gaps = []
    for prev, cur in zip(idx[:-1], idx[1:]):
        delta = (cur - prev).days
        if delta > min_len:
            gaps.append((prev.date().isoformat(), cur.date().isoformat(), delta - 1))
    for start, end, length in gaps:
        print(f"  дыра в данных: {start} -> {end} ({length} дней)")
    return gaps


def coverage(wide: pd.DataFrame) -> pd.Series:
    """Доля дней, в которые валюта вообще котировалась."""
    return wide.notna().mean().sort_values(ascending=False)


def build(raw_path=None, out_path=None) -> pd.DataFrame:
    raw = load_raw(raw_path)
    long = tidy(raw)
    wide = to_wide(long)
    print(f"  широкая таблица: {wide.shape[0]} дней x {wide.shape[1]} валют")

    wide, _ = drop_non_trading_days(wide)
    wide = drop_snapshot_echoes(wide)
    report_gaps(wide)

    out_path = out_path or config.CLEAN_RATES_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, float_format="%.6f")
    print(f"Сохранил {out_path.name}: {wide.shape[0]} торговых дней, {wide.shape[1]} валют")
    return wide


def load_clean(path=None) -> pd.DataFrame:
    path = path or config.CLEAN_RATES_CSV
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


# --------------------------------------------------------------------------
# Длинный ряд из Excel-выгрузки Нацбанка (2005-2026) плюс нефть
# --------------------------------------------------------------------------

def tidy_long(raw: pd.DataFrame) -> pd.DataFrame:
    """То же приведение к типам, но для выгрузки архива.

    Отличие от XML-эндпоинта одно, зато важное: за двадцать лет менялась
    кратность котировки. Узбекский сум в 2013 году публиковался за одну
    единицу, а в 2016-м — уже за сто. Кратность лежит в каждой строке
    отдельно, поэтому делить нужно построчно, а не по словарю валют.
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["quant"] = pd.to_numeric(df["quant"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    bad = df["value"].isna() | df["quant"].isna() | (df["quant"] <= 0)
    if bad.any():
        print(f"  выбрасываю {int(bad.sum())} строк с битыми числами")
        df = df[~bad]

    df["rate_kzt"] = df["value"] / df["quant"]
    df = df[~df.duplicated(subset=["date", "code"], keep="last")]
    return df[["date", "code", "rate_kzt"]].sort_values(["date", "code"])


def load_brent(path=None) -> pd.Series:
    path = path or config.RAW_BRENT_CSV
    brent = pd.read_csv(path, parse_dates=["date"]).set_index("date")["brent_usd"]
    print(f"Прочитал нефть: {len(brent)} дней, {brent.index.min().date()} .. {brent.index.max().date()}")
    return brent


def build_long(raw_path=None, out_path=None, min_coverage: float = 0.9) -> pd.DataFrame:
    """Собрать чистый длинный ряд курсов.

    `min_coverage` отсекает валюты, которые Нацбанк начал котировать уже
    посреди выборки: грузинский лари, армянский драм и компания появляются
    только с 2015 года, и в анализе двадцатилетних режимов от них один шум.
    """
    raw = pd.read_csv(raw_path or config.RAW_LONG_CSV)
    print(f"Прочитал {len(raw)} строк длинного ряда")

    long = tidy_long(raw)
    wide = to_wide(long)
    print(f"  широкая таблица: {wide.shape[0]} дней x {wide.shape[1]} валют")

    coverage_share = wide.notna().mean()
    thin = coverage_share[coverage_share < min_coverage]
    if len(thin):
        print(f"  убираю валюты с неполной историей: {sorted(thin.index)}")
        wide = wide.drop(columns=thin.index)

    wide, _ = drop_non_trading_days(wide)
    report_gaps(wide)

    out_path = out_path or config.CLEAN_LONG_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path, float_format="%.6f")
    print(f"Сохранил {out_path.name}: {wide.shape[0]} торговых дней, {wide.shape[1]} валют, "
          f"{wide.index.min().date()} .. {wide.index.max().date()}")
    return wide


def load_clean_long(path=None) -> pd.DataFrame:
    path = path or config.CLEAN_LONG_CSV
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def main() -> None:
    parser = argparse.ArgumentParser(description="Очистить сырые курсы Нацбанка")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--long", action="store_true", help="чистить длинный ряд из архива")
    args = parser.parse_args()
    if args.long:
        build_long(args.raw, args.out)
    else:
        build(args.raw, args.out)


if __name__ == "__main__":
    main()
