"""Графики к отчёту. Каждая функция возвращает путь к сохранённому файлу."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.ticker

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


# --------------------------------------------------------------------------
# Длинный ряд: режимы курсовой политики и нефть
# --------------------------------------------------------------------------

def fig_long_history(prices: pd.DataFrame, name="10_long_history.png"):
    """Двадцать лет курса: две эпохи и три ступеньки."""
    _setup()
    fig, ax = plt.subplots(figsize=(11, 5.4))

    series = prices["KZT"]
    float_date = pd.Timestamp(config.FLOAT_DATE)

    ax.axvspan(series.index[0], float_date, color=config.COLOR_MUTED, alpha=0.12, zorder=0)
    ax.plot(series.index, series, color=config.COLOR_KZT, lw=1.5)
    ax.set_yscale("log")
    ax.set_yticks([100, 150, 200, 300, 400, 500])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("тенге за доллар (лог. шкала)")
    ax.set_title("Тенге за двадцать лет: курс переставляли, а не торговали")

    lo, hi = ax.get_ylim()
    for step, (day, label) in enumerate(config.DEVALUATIONS.items()):
        stamp = pd.Timestamp(day)
        if stamp not in series.index:
            nearest = series.index[series.index.searchsorted(stamp)]
        else:
            nearest = stamp
        ax.scatter([nearest], [series.loc[nearest]], color=config.COLOR_RUB, s=40, zorder=5)
        ax.annotate(
            label,
            xy=(mdates.date2num(nearest), series.loc[nearest]),
            xytext=(-10, 26 + 14 * (step % 2)),
            textcoords="offset points",
            fontsize=8,
            color="#3c4043",
            ha="right",
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": config.COLOR_MUTED},
        )

    ax.annotate("управляемый курс", xy=(0.11, 0.93), xycoords="axes fraction",
                fontsize=9, color="#5f6368", ha="center")
    ax.annotate("свободное плавание", xy=(0.74, 0.93), xycoords="axes fraction",
                fontsize=9, color="#5f6368", ha="center")
    return _save(fig, name)


def fig_horizon(profile: pd.DataFrame, name="11_horizon_profile.png"):
    """Рубль виден на днях, нефть — на кварталах."""
    _setup()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    x = range(len(profile))
    ax.plot(x, profile["RUB"], marker="o", lw=2.0, color=config.COLOR_RUB, label="рубль")
    ax.plot(x, profile["BRENT"], marker="o", lw=2.0, color=config.COLOR_ACCENT, label="нефть Brent")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(list(x), profile.index)
    ax.set_ylabel("корреляция с тенге")
    ax.set_xlabel("на каком горизонте считаем доходность")
    ax.set_title("Спор «рубль или нефть» решается выбором горизонта")

    for i, (rub, brent) in enumerate(zip(profile["RUB"], profile["BRENT"])):
        ax.annotate(f"{rub:+.2f}", xy=(i, rub), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=config.COLOR_RUB)
        ax.annotate(f"{brent:+.2f}", xy=(i, brent), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=9, color=config.COLOR_ACCENT)
    ax.legend(loc="center right")
    ax.margins(y=0.2)
    return _save(fig, name)


def fig_regimes(pre: pd.DataFrame, post: pd.DataFrame, name="12_regimes.png"):
    """До и после плавания — две разные валюты."""
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, (profile, title) in zip(
        axes,
        [(pre, "Управляемый курс\n2005 — август 2015"), (post, "Свободное плавание\nс августа 2015")],
    ):
        ax.plot(range(len(profile)), profile["RUB"], marker="o", lw=2.0,
                color=config.COLOR_RUB, label="рубль")
        ax.plot(range(len(profile)), profile["BRENT"], marker="o", lw=2.0,
                color=config.COLOR_ACCENT, label="нефть Brent")
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xticks(range(len(profile)), profile.index, fontsize=8)
        ax.set_ylim(-0.62, 0.62)
        ax.set_title(title, fontsize=11)

    axes[0].set_ylabel("корреляция с тенге")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Плавающий курс не ослабил связи, а создал их",
                 fontsize=13, fontweight="bold", y=1.09)
    fig.tight_layout()
    return _save(fig, name)


def fig_long_jumps(rets: pd.Series, name="13_long_jumps.png"):
    """Три дня из пяти тысяч решают почти половину."""
    _setup()
    series = rets.dropna()
    worst = series.nlargest(3)

    full = series.cumsum() * 100
    without = series.drop(worst.index).cumsum() * 100

    fig, ax = plt.subplots(figsize=config.FIG_SIZE_WIDE)
    ax.plot(full.index, full, color=config.COLOR_KZT, lw=1.6, label="как было")
    ax.plot(without.index, without, color=config.COLOR_ACCENT, lw=1.6, ls="--",
            label="без трёх дней девальваций")
    ax.scatter(worst.index, full.loc[worst.index], color=config.COLOR_RUB, s=40, zorder=5)
    ax.axhline(0, color="black", lw=0.8)

    # подписи разводим по разные стороны от линии, иначе они ложатся прямо на неё
    offsets = [(10, -30), (-12, 16), (12, -34)]
    for (day, value), offset in zip(worst.sort_index().items(), offsets):
        ax.annotate(
            f"{day.date()}\n{value * 100:+.0f}%",
            xy=(mdates.date2num(day), full.loc[day]),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] > 0 else "right",
            fontsize=8,
            color="#3c4043",
        )

    ax.set_ylabel("накопленное ослабление тенге к доллару, %")
    ax.set_title(f"Три дня из {len(series)} дают почти половину всего ослабления")
    ax.legend(loc="upper left")
    return _save(fig, name)


def fig_brent_scatter(monthly: pd.DataFrame, name="14_brent_monthly.png"):
    """Месячные изменения: нефть вверх — тенге крепче."""
    _setup()
    data = monthly[["KZT", "BRENT"]].dropna() * 100

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(data["BRENT"], data["KZT"], s=26, alpha=0.7, color=config.COLOR_KZT)

    slope, intercept = np.polyfit(data["BRENT"], data["KZT"], 1)
    grid = np.linspace(data["BRENT"].min(), data["BRENT"].max(), 50)
    corr = data["KZT"].corr(data["BRENT"])
    ax.plot(grid, slope * grid + intercept, color=config.COLOR_ACCENT, lw=1.8,
            label=f"наклон {slope:.2f}, r = {corr:+.2f}")

    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("изменение цены Brent за месяц, %")
    ax.set_ylabel("изменение курса тенге к доллару за месяц, %")
    ax.set_title("На месячном горизонте нефть видно невооружённым глазом")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, name)


def fig_robustness(oil: pd.DataFrame, rouble: pd.DataFrame, name="15_robustness.png"):
    """Проверка: устойчива ли связь или держится на нескольких катастрофах."""
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)

    columns = [c for c in oil.columns if c != "наблюдений"]
    styles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D"]

    for ax, table, title, color in (
        (axes[0], oil, "Нефть Brent", config.COLOR_ACCENT),
        (axes[1], rouble, "Рубль", config.COLOR_RUB),
    ):
        for column, style, marker in zip(columns, styles, markers):
            ax.plot(
                range(len(table)),
                table[column].abs(),
                style,
                marker=marker,
                ms=5,
                lw=1.6,
                color=color,
                alpha=1.0 if column == "Пирсон" else 0.55,
                label=column,
            )
        ax.set_xticks(range(len(table)), table.index, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 0.62)

    axes[0].set_ylabel("сила связи с тенге\n(корреляция по модулю)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(
        "Связь с рублём выдерживает любую проверку, связь с нефтью — нет",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout()
    return _save(fig, name)
