"""Дневная цена нефти Brent.

Это единственный источник в проекте, внешний по отношению к Казахстану.
Без него никак: половина разговоров про тенге — про нефть, а сама РК цену
на Brent не публикует. Беру серию DCOILBRENTEU (Brent, Europe, спот,
долларов за баррель) из открытой базы FRED — она ведётся с 1987 года
и отдаётся обычным CSV без ключей и регистраций.

Запуск:
    python -m src.fetch_brent
"""

from __future__ import annotations

import argparse
import io
import time

import pandas as pd
import requests

from src import config

# Перед FRED стоит защита от ботов с неожиданной логикой: она пропускает
# запросы с дефолтным User-Agent библиотеки, но вешает намертво и подставной
# браузерный, и свой собственный. Поэтому заголовок здесь не переопределяется
# вовсе — это не небрежность, а единственный вариант, который работает.
TIMEOUT = 30
RETRIES = 3


def download() -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(config.BRENT_URL, timeout=TIMEOUT)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2.0 * attempt)
    else:
        raise SystemExit(f"FRED не ответил за {RETRIES} попытки: {last_error}")

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", "brent_usd"]

    # В праздники FRED ставит точку вместо числа — это не ноль и не пропуск
    # в данных, просто рынок не работал.
    df["brent_usd"] = pd.to_numeric(df["brent_usd"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["brent_usd"])
    print(f"  {before} строк, из них {before - len(df)} нерабочих дней выброшено")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать дневные котировки Brent")
    parser.add_argument("--start", default=None, help="обрезать историю слева, YYYY-MM-DD")
    args = parser.parse_args()

    df = download()
    if args.start:
        df = df[df["date"] >= args.start]

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_BRENT_CSV, index=False)
    print(f"Готово: {len(df)} дней, {df['date'].min()} .. {df['date'].max()}")
    print(f"Сохранил {config.RAW_BRENT_CSV.name}")


if __name__ == "__main__":
    main()
