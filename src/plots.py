"""Графики к отчёту. Каждая функция возвращает путь к сохранённому файлу."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import analysis as A
from src import config


def _setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": config.FIG_DPI,
            "savefig.dpi": config.FIG_DPI,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def _save(fig, name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.relative_to(config.PROJECT_ROOT)}")
    return path


def _annotate_events(ax, ylim_frac: float = 0.96) -> None:
    lo, hi = ax.get_ylim()
    left, right = ax.get_xlim()
    # подписи ставим лесенкой: соседние события иначе наезжают друг на друга
    ladder = [0.0, 0.22, 0.44]
    for step, (day, label) in enumerate(config.EVENTS.items()):
        y = lo + (hi - lo) * (ylim_frac - ladder[step % len(ladder)])
        stamp = mdates.date2num(pd.Timestamp(day))
        if not left <= stamp <= right:
            continue
        ax.axvline(stamp, color=config.COLOR_MUTED, lw=0.8, ls="--", zorder=0)
        ax.annotate(
            label,
            xy=(stamp, y),
            rotation=90,
            fontsize=7,
            color="#5f6368",
            ha="right",
            va="top",
        )


# --------------------------------------------------------------------------

def fig_levels(px: pd.DataFrame, name="01_levels_kzt_vs_rub.png"):
    """Тенге и рубль к доллару, оба приведены к 100 на старте."""
    _setup()
    fig, ax = plt.subplots(figsize=config.FIG_SIZE_WIDE)

    for code, color, width in (("KZT", config.COLOR_KZT, 1.8), ("RUB", config.COLOR_RUB, 1.3)):
        series = 100 * px[code] / px[code].iloc[0]
        ax.plot(series.index, series, color=color, lw=width, label=code)

    ax.axhline(100, color="black", lw=0.7, alpha=0.5)
    ax.set_ylabel("индекс, старт = 100\n(выше = валюта слабее)")
    ax.set_title("Тенге и рубль к доллару: одна дорога или разные?")
    ax.legend(loc="upper left")
    _annotate_events(ax)

    final = {c: 100 * px[c].iloc[-1] / px[c].iloc[0] for c in ("KZT", "RUB")}
    ax.annotate(
        f"итог: KZT {final['KZT']:.0f}, RUB {final['RUB']:.0f}",
        xy=(0.99, 0.04),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="#5f6368",
    )
    return _save(fig, name)


def fig_corr_ranking(corr: pd.Series, name="02_corr_ranking.png"):
    """С какой валютой тенге ходит в одну ногу."""
    _setup()
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    data = corr.sort_values()
    colors = [config.COLOR_RUB if c == "RUB" else config.COLOR_MUTED for c in data.index]
    ax.barh(data.index, data.to_numpy(), color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("корреляция дневных доходностей с тенге")
    ax.set_title("Ближайший попутчик тенге — рубль, и с большим отрывом")

    for code, value in data.items():
        ax.annotate(
            f"{value:+.2f}",
            xy=(value, code),
            xytext=(4 if value >= 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color="#3c4043",
        )
    ax.margins(x=0.12)
    return _save(fig, name)


def fig_rolling_corr(rets: pd.DataFrame, window=90, name="03_rolling_corr.png"):
    """Связь с рублём не постоянна — она включается и выключается."""
    _setup()
    series = rets["KZT"].rolling(window, min_periods=window // 2).corr(rets["RUB"])

    fig, ax = plt.subplots(figsize=config.FIG_SIZE_WIDE)
    ax.plot(series.index, series, color=config.COLOR_KZT, lw=1.4)
    ax.fill_between(series.index, 0, series, where=series > 0, color=config.COLOR_KZT, alpha=0.18)
    ax.fill_between(series.index, 0, series, where=series < 0, color=config.COLOR_RUB, alpha=0.18)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(series.mean(), color=config.COLOR_ACCENT, lw=1.0, ls=":", label=f"среднее {series.mean():.2f}")
    ax.set_ylabel(f"скользящая корреляция, окно {window} дней")
    ax.set_title("Тенге то держится за рубль, то отпускает его")
    ax.legend(loc="upper right")
    _annotate_events(ax)
    return _save(fig, name)


def fig_conditional_corr(rets: pd.DataFrame, name="04_conditional_corr.png"):
    """Корреляция растёт вместе с размером движения рубля."""
    _setup()
    buckets = [(0.0, 0.005, "< 0,5%"), (0.005, 0.01, "0,5-1%"), (0.01, 0.02, "1-2%"), (0.02, 1.0, "> 2%")]
    pair = rets[["KZT", "RUB"]].dropna()

    labels, values, counts = [], [], []
    for lo, hi, label in buckets:
        sub = pair[pair["RUB"].abs().between(lo, hi, inclusive="left")]
        labels.append(label)
        values.append(sub["KZT"].corr(sub["RUB"]))
        counts.append(len(sub))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(labels, values, color=config.COLOR_KZT, width=0.6)
    bars[-1].set_color(config.COLOR_RUB)
    ax.set_xlabel("размер дневного движения рубля к доллару")
    ax.set_ylabel("корреляция тенге с рублём")
    ax.set_title("Тенге замечает рубль только тогда, когда рубль летит")

    for bar, value, n in zip(bars, values, counts):
        ax.annotate(
            f"{value:+.2f}\nn={n}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.margins(y=0.22)
    return _save(fig, name)


def fig_scatter(rets: pd.DataFrame, name="05_scatter_kzt_rub.png"):
    """Облако дневных доходностей: связь держится на хвостах."""
    _setup()
    pair = rets[["KZT", "RUB"]].dropna() * 100
    calm = pair[pair["RUB"].abs() < 2]
    storm = pair[pair["RUB"].abs() >= 2]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(calm["RUB"], calm["KZT"], s=11, alpha=0.35, color=config.COLOR_MUTED, label="обычные дни")
    ax.scatter(storm["RUB"], storm["KZT"], s=26, alpha=0.8, color=config.COLOR_RUB, label="рубль двигался больше 2%")

    slope, intercept = np.polyfit(pair["RUB"], pair["KZT"], 1)
    grid = np.linspace(pair["RUB"].min(), pair["RUB"].max(), 50)
    ax.plot(grid, slope * grid + intercept, color=config.COLOR_KZT, lw=1.6,
            label=f"наклон {slope:.2f}")

    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("дневное изменение рубля к доллару, %")
    ax.set_ylabel("дневное изменение тенге к доллару, %")
    ax.set_title("Рубль падает на 10% — тенге на 1,4%")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, name)


def fig_corr_by_year(table: pd.DataFrame, name="06_corr_by_year.png"):
    """Как связь менялась по годам."""
    _setup()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    im = ax.imshow(table.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iat[i, j]
            ax.annotate(f"{value:+.2f}", xy=(j, i), ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 0.3 else "#202124")

    ax.set_title("Корреляция с тенге по годам: 2026-й выбивается из ряда")
    ax.set_xlabel("крайние годы неполные: 2021-й с мая, 2026-й по сентябрь", fontsize=8, color="#5f6368")
    fig.colorbar(im, ax=ax, shrink=0.85, label="корреляция")
    return _save(fig, name)


def fig_tail_contribution(rets: pd.DataFrame, top_n=10, name="07_tail_contribution.png"):
    """Весь путь тенге вниз укладывается в несколько дней."""
    _setup()
    series = rets["KZT"].dropna()
    worst = series.nlargest(top_n).index

    full = series.cumsum() * 100
    without = series.drop(worst).cumsum() * 100

    fig, ax = plt.subplots(figsize=config.FIG_SIZE_WIDE)
    ax.plot(full.index, full, color=config.COLOR_KZT, lw=1.6, label="как было")
    ax.plot(without.index, without, color=config.COLOR_ACCENT, lw=1.6, ls="--",
            label=f"без {top_n} худших дней")
    ax.axhline(0, color="black", lw=0.8)
    ax.scatter(worst, full.loc[worst], color=config.COLOR_RUB, s=28, zorder=5,
               label="худшие дни")

    ax.set_ylabel("накопленное ослабление тенге к доллару, %")
    ax.set_title(f"Убери {top_n} дней из {len(series)} — и тенге за пять лет только крепчал")
    ax.legend(loc="upper left")
    return _save(fig, name)


def fig_return_distribution(rets: pd.DataFrame, name="08_return_distribution.png"):
    """Распределение дневных изменений: длинный правый хвост."""
    _setup()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    kzt = rets["KZT"].dropna() * 100
    bins = np.linspace(-3.5, 6.5, 81)
    ax.hist(kzt, bins=bins, color=config.COLOR_KZT, alpha=0.85)
    ax.axvline(0, color="black", lw=0.9)
    ax.axvline(kzt.mean(), color=config.COLOR_ACCENT, lw=1.2, ls=":",
               label=f"среднее {kzt.mean():+.3f}%")

    ax.set_yscale("log")
    ax.set_xlabel("дневное изменение курса тенге к доллару, %\n(вправо = тенге слабеет)")
    ax.set_ylabel("число дней (лог. шкала)")
    ax.set_title(f"Вверх по лестнице, вниз на лифте: асимметрия {kzt.skew():+.2f}")
    ax.annotate(
        f"дней ослабления: {(kzt > 0).mean() * 100:.0f}%\n"
        f"худший день: {kzt.max():+.1f}%\nлучший день: {kzt.min():+.1f}%",
        xy=(0.98, 0.92),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color="#3c4043",
    )
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, name)


def fig_intramonth(rets_trade: pd.DataFrame, name="09_intramonth.png"):
    """Проверка народной приметы про налоговую неделю."""
    _setup()
    profile = A.intramonth_profile(rets_trade["KZT"])

    fig, ax = plt.subplots(figsize=config.FIG_SIZE_WIDE)
    colors = [config.COLOR_ACCENT if 20 <= d <= 25 else config.COLOR_MUTED for d in profile.index]
    ax.bar(profile.index, profile["средн_%"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(profile["средн_%"].mean(), color=config.COLOR_KZT, lw=1.0, ls=":",
               label="среднее по всем дням")

    ax.set_xlabel("число месяца (дата торгов, не дата публикации курса)")
    ax.set_ylabel("среднее дневное изменение тенге, %")
    ax.set_title("Налоговая неделя 20-25 числа никак не выделяется")
    ax.set_xticks(range(1, 32))
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, name)
