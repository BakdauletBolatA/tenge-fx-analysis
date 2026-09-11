"""Пути, константы и справочники проекта."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_RATES_CSV = RAW_DIR / "nbk_rates_raw.csv"
CLEAN_RATES_CSV = PROCESSED_DIR / "rates_daily.csv"

# Открытый XML-эндпоинт Нацбанка РК: один запрос = один календарный день.
NBK_RATES_URL = "https://nationalbank.kz/rss/get_rates.cfm"

# Архив эндпоинта не бездонный: на момент сборки проекта самая ранняя дата,
# на которую он отдаёт данные, — начало мая 2021 года. Ставим с запасом,
# пустые дни загрузчик просто пометит как отсутствующие.
DEFAULT_START = dt.date(2021, 5, 1)

# Валюты, которые нас интересуют в анализе. Нацбанк котирует 48 штук,
# но тащить в корреляции иранский риал смысла нет.
#
# Соседи и основные торговые партнёры — то, с чем тенге теоретически может
# ходить в одну ногу.
PEERS_REGION = ["RUB", "CNY", "UZS", "KGS", "AMD", "GEL", "AZN", "TRY", "BYN"]

# Резервные / сырьевые валюты — контрольная группа.
PEERS_GLOBAL = ["EUR", "GBP", "JPY", "CHF", "CAD", "NOK", "AUD", "BRL", "ZAR", "MXN", "INR"]

PEERS = PEERS_REGION + PEERS_GLOBAL

# Базовая валюта, через которую сравниваем всех со всеми.
BASE = "USD"

# Даты, которые стоит подписать на графиках: без них временной ряд
# читается как случайный шум.
EVENTS = {
    dt.date(2022, 2, 24): "Начало войны в Украине",
    dt.date(2022, 6, 29): "Рубль на максимуме (50 за доллар)",
    dt.date(2023, 8, 14): "ЦБ РФ поднимает ставку до 12%",
    dt.date(2024, 11, 27): "Рубль пробивает 110 за доллар",
}

# Оформление графиков — держим в одном месте, чтобы не плодить магию по файлам.
FIG_DPI = 140
FIG_SIZE_WIDE = (11, 5.0)
FIG_SIZE_SQUARE = (7.5, 6.5)

COLOR_KZT = "#1f5f8b"
COLOR_RUB = "#b3452c"
COLOR_MUTED = "#9aa0a6"
COLOR_ACCENT = "#c8942a"
