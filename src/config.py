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
RAW_LONG_CSV = RAW_DIR / "nbk_rates_long.csv"
RAW_BRENT_CSV = RAW_DIR / "brent_fred.csv"

CLEAN_RATES_CSV = PROCESSED_DIR / "rates_daily.csv"
CLEAN_LONG_CSV = PROCESSED_DIR / "rates_daily_long.csv"

# У Нацбанка два разных способа отдать курсы, и оба нужны.
#
# 1. XML-эндпоинт: один запрос = один календарный день, отдаёт сразу все
#    котируемые валюты, но архив начинается только с мая 2021 года.
NBK_RATES_URL = "https://nationalbank.kz/rss/get_rates.cfm"

# 2. Excel-выгрузка с сайта: валюты приходится выбирать по внутренним id,
#    зато история тянется с 1999 года — то есть захватывает все три
#    девальвации тенге (2009, 2014 и переход к плаванию в августе 2015).
NBK_ARCHIVE_URL = (
    "https://nationalbank.kz/ru/exchangerates/"
    "ezhednevnye-oficialnye-rynochnye-kursy-valyut/excel"
)

# Внутренние id валют в форме выгрузки. Сняты с самой формы: порядок
# чекбоксов совпадает с порядком строк в таблице курсов на странице.
NBK_RATE_IDS = {
    "USD": 5, "EUR": 6, "RUB": 16, "CNY": 8, "GBP": 2, "JPY": 26, "CHF": 23,
    "CAD": 7, "NOK": 14, "AUD": 1, "TRY": 41, "UZS": 20, "KGS": 10, "GEL": 52,
    "AMD": 51, "AZN": 48, "BYN": 38, "BRL": 46, "ZAR": 40, "MXN": 54, "INR": 49,
}

# Цена нефти Brent, дневные данные. Единственный источник в проекте,
# внешний по отношению к Казахстану — но без него "тенге идёт за нефтью"
# проверить нечем, а сама РК цену на Brent не публикует.
BRENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
BRENT_SERIES = "DCOILBRENTEU"

# Архив эндпоинта не бездонный: на момент сборки проекта самая ранняя дата,
# на которую он отдаёт данные, — начало мая 2021 года. Ставим с запасом,
# пустые дни загрузчик просто пометит как отсутствующие.
DEFAULT_START = dt.date(2021, 5, 1)

# Для длинного ряда берём двадцать с лишним лет: этого хватает, чтобы
# сравнить эпоху управляемого курса с эпохой свободного плавания.
ARCHIVE_START = dt.date(2005, 1, 1)

# 20 августа 2015 года Нацбанк отпустил тенге в свободное плавание и перешёл
# к инфляционному таргетированию. Это главный разлом всей выборки: до него
# курс был решением, после — ценой.
FLOAT_DATE = dt.date(2015, 8, 20)

# Разовые девальвации — не плавные движения рынка, а объявленные решения.
DEVALUATIONS = {
    dt.date(2009, 2, 4): "Девальвация 2009 года",
    dt.date(2014, 2, 11): "Девальвация 2014 года",
    dt.date(2015, 8, 20): "Переход к свободному плаванию",
}

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
