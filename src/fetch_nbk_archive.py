"""Длинная история курсов из Excel-выгрузки Нацбанка.

XML-эндпоинт (fetch_nbk.py) удобен, но его архив начинается с мая 2021 года,
а самое интересное в истории тенге случилось раньше: девальвация февраля 2009,
девальвация февраля 2014 и переход к свободному плаванию 20 августа 2015.

Excel-выгрузка с сайта Нацбанка отдаёт те же официальные курсы с 1999 года.
Платить за это приходится тем, что валюты нужно перечислять по внутренним id
формы, а ответ приходит в виде .xlsx.

Запуск:
    python -m src.fetch_nbk_archive
    python -m src.fetch_nbk_archive --start 2000-01-01 --chunk-years 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import time

import openpyxl
import pandas as pd
import requests

from src import config

USER_AGENT = "tenge-fx-analysis/0.1 (educational data project)"
TIMEOUT = 120
RETRIES = 3


def _chunks(start: dt.date, end: dt.date, years: int):
    """Резать период кусками: одним запросом на двадцать лет сервис давится."""
    cursor = start
    while cursor <= end:
        stop = min(dt.date(cursor.year + years, 1, 1) - dt.timedelta(days=1), end)
        yield cursor, stop
        cursor = stop + dt.timedelta(days=1)


def download_chunk(
    session: requests.Session, start: dt.date, end: dt.date, codes: list[str]
) -> bytes | None:
    params = [("beginDate", start.strftime("%d.%m.%Y")), ("endDate", end.strftime("%d.%m.%Y"))]
    params += [("rates[]", config.NBK_RATE_IDS[c]) for c in codes]

    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.get(
                config.NBK_ARCHIVE_URL,
                params=params,
                timeout=TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
        except requests.RequestException:
            time.sleep(2.0 * attempt)
            continue

        # Сервис на ошибку отвечает html-страницей с кодом 200, а не ошибкой.
        if resp.content[:2] != b"PK":
            time.sleep(2.0 * attempt)
            continue
        return resp.content

    return None


def parse_workbook(blob: bytes) -> pd.DataFrame:
    """Разобрать .xlsx в длинную таблицу date / code / quant / value.

    Колонки идут парами: CODE_quant и CODE. Кратность лежит в каждой строке
    отдельно, и это не формальность — за двадцать лет она менялась: узбекский
    сум в 2013 году котировался за 1 единицу, а в 2016-м уже за 100.
    """
    sheet = openpyxl.load_workbook(io.BytesIO(blob), read_only=True).active
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))

    pairs = []
    for i, name in enumerate(header):
        if i == 0 or name is None or str(name).endswith("_quant"):
            continue
        quant_col = header.index(f"{name}_quant") if f"{name}_quant" in header else None
        pairs.append((str(name), i, quant_col))

    records = []
    for row in rows:
        day = row[0]
        if not day:
            continue
        for code, value_col, quant_col in pairs:
            value = row[value_col]
            if value is None:
                continue
            records.append(
                {
                    "date": dt.datetime.strptime(str(day), "%d.%m.%Y").date().isoformat(),
                    "code": code,
                    "quant": row[quant_col] if quant_col is not None else 1,
                    "value": value,
                }
            )
    return pd.DataFrame(records)


def fetch_range(start: dt.date, end: dt.date, chunk_years: int = 4) -> pd.DataFrame:
    codes = list(config.NBK_RATE_IDS)
    session = requests.Session()
    frames = []

    for chunk_start, chunk_end in _chunks(start, end, chunk_years):
        blob = download_chunk(session, chunk_start, chunk_end, codes)
        if blob is None:
            print(f"  {chunk_start} .. {chunk_end}: не отдалось")
            continue
        frame = parse_workbook(blob)
        frames.append(frame)
        print(f"  {chunk_start} .. {chunk_end}: {len(frame)} строк")
        time.sleep(1.0)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "code"], keep="last")
    return df.sort_values(["date", "code"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать длинную историю курсов Нацбанка")
    parser.add_argument("--start", default=config.ARCHIVE_START.isoformat())
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--chunk-years", type=int, default=4)
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    print(f"Тяну курсы {start} .. {end} по {len(config.NBK_RATE_IDS)} валютам")

    df = fetch_range(start, end, args.chunk_years)
    if df.empty:
        raise SystemExit("Нацбанк не отдал ни одной строки")

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_LONG_CSV, index=False)
    print(f"Готово: {len(df)} строк, {df['code'].nunique()} валют, "
          f"{df['date'].min()} .. {df['date'].max()}")
    print(f"Сохранил {config.RAW_LONG_CSV.name}")


if __name__ == "__main__":
    main()
