"""Загрузка официальных курсов валют с сайта Нацбанка РК.

Эндпоинт отдаёт ровно один день за запрос, поэтому за пять лет истории
приходится сделать почти две тысячи обращений. Чтобы не дёргать чужой
сервер при каждом перезапуске, ответы кладём в data/cache/ и оттуда же
читаем при повторных прогонах.

Запуск:
    python -m src.fetch_nbk                      # с DEFAULT_START по сегодня
    python -m src.fetch_nbk --start 2022-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from src import config

USER_AGENT = "tenge-fx-analysis/0.1 (educational data project)"
TIMEOUT = 25
RETRIES = 4
WORKERS = 6

# Эндпоинт иногда отдаёт пустой ответ на вполне рабочую дату — с первого
# раза повезло не всегда. Отличаем "данных нет" от "сервер моргнул" по
# наличию вот этой подстроки.
_NO_DATA_MARKER = "информации нет"


def _cache_path(day: dt.date) -> Path:
    return config.CACHE_DIR / f"{day.year}" / f"{day.isoformat()}.xml"


def _read_cache(day: dt.date) -> str | None:
    path = _cache_path(day)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _write_cache(day: dt.date, text: str) -> None:
    path = _cache_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def download_day(session: requests.Session, day: dt.date) -> str | None:
    """Вернуть XML за один день. None — если сервер так и не ответил."""
    cached = _read_cache(day)
    if cached is not None:
        return cached

    params = {"fdate": day.strftime("%d.%m.%Y")}
    for attempt in range(1, RETRIES + 1):
        try:
            resp = session.get(
                config.NBK_RATES_URL,
                params=params,
                timeout=TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
        except requests.RequestException:
            time.sleep(1.5 * attempt)
            continue

        text = resp.text
        if _NO_DATA_MARKER in text:
            # Честное "данных нет" — кэшируем, чтобы не спрашивать заново.
            _write_cache(day, text)
            return text
        if "<item>" in text:
            _write_cache(day, text)
            return text

        # Пустой ответ без маркера — похоже на сбой, пробуем ещё раз.
        time.sleep(1.0 * attempt)

    return None


def parse_day_xml(xml_text: str, day: dt.date) -> list[dict]:
    """Разобрать XML одного дня в список записей по валютам."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    rows: list[dict] = []
    for item in root.findall(".//item"):
        code = (item.findtext("title") or "").strip()
        value = (item.findtext("description") or "").strip()
        if not code or not value:
            continue
        rows.append(
            {
                "date": day.isoformat(),
                "code": code,
                "name_ru": (item.findtext("fullname") or "").strip(),
                "quant": (item.findtext("quant") or "1").strip(),
                "value": value,
                "change": (item.findtext("change") or "").strip(),
            }
        )
    return rows


def daterange(start: dt.date, end: dt.date):
    day = start
    while day <= end:
        yield day
        day += dt.timedelta(days=1)


def fetch_range(start: dt.date, end: dt.date, workers: int = WORKERS) -> pd.DataFrame:
    days = list(daterange(start, end))
    print(f"Загружаю {len(days)} дней: {start} .. {end}")

    session = requests.Session()
    done = 0
    failed: list[dt.date] = []
    rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for day, xml_text in zip(days, pool.map(lambda d: download_day(session, d), days)):
            done += 1
            if xml_text is None:
                failed.append(day)
            else:
                rows.extend(parse_day_xml(xml_text, day))
            if done % 200 == 0:
                print(f"  {done}/{len(days)} дней, строк: {len(rows)}")

    if failed:
        print(f"Не удалось получить {len(failed)} дней, первые: {failed[:5]}", file=sys.stderr)

    df = pd.DataFrame(rows)
    print(f"Готово: {len(df)} строк, {df['code'].nunique() if len(df) else 0} валют")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать курсы Нацбанка РК")
    parser.add_argument("--start", default=config.DEFAULT_START.isoformat())
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    df = fetch_range(start, end, workers=args.workers)
    if df.empty:
        raise SystemExit("Нацбанк не отдал ни одной строки — проверь соединение")

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_RATES_CSV, index=False)
    print(f"Сохранил {config.RAW_RATES_CSV.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
