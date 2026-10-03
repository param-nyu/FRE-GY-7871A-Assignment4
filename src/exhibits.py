"""Figures for the report.

Four charts, each earning its place by showing something a table cannot:

1. The sentiment trend the brief asks for, with the event marked.
2. The two query families plotted against each other, which is where the
   divergence between existential fear and the commercial AI trade shows up.
3. Cumulative basket performance against the benchmark.
4. The media-bias comparison, US vs non-US vs scientific/tech.

All styling is local to this module so the notebook and the PDF produce
byte-identical figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from .config import EVENT_DATE, OUTPUT_DIR  # noqa: E402

PALETTE = {
    "risk": "#c1443c",       # existential-risk sentiment
    "company": "#2b6cb0",    # commercial AI sentiment
    "news": "#6b46c1",
    "basket": "#1a7f5a",
    "benchmark": "#8a8a8a",
    "us": "#c1443c",
    "nonus": "#2b6cb0",
    "tech": "#1a7f5a",
    "grid": "#d8d8d8",
}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.autolayout": True,
})


def _mark_event(ax) -> None:
    """Vertical rule at the Hubinger post, on every time-series chart."""
    event = pd.Timestamp(EVENT_DATE)
    ax.axvline(event, color="#333333", linestyle="--", linewidth=1.0, zorder=1)
    ax.annotate(
        "Hubinger post",
        xy=(event, ax.get_ylim()[1]),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=7.5,
        color="#333333",
    )


def _format_dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for label in ax.get_xticklabels():
        label.set_rotation(0)


def figure1_sentiment_trend(
    sentiment: pd.DataFrame, output_dir: Path = OUTPUT_DIR
) -> Path:
    """Daily sentiment ratio for both channels, with volume beneath."""
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 3.9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    frame = sentiment.sort_values("date")

    if "x_ratio" in frame:
        ax.plot(frame["date"], frame["x_ratio"], marker="o", markersize=3.2,
                color=PALETTE["risk"], label="X / Twitter", linewidth=1.5)
    if "news_ratio" in frame:
        ax.plot(frame["date"], frame["news_ratio"], marker="s", markersize=3.2,
                color=PALETTE["news"], label="News headlines", linewidth=1.5)

    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_ylabel("Sentiment ratio\n(pos - neg) / total")
    ax.set_title("Figure 1. Daily AI sentiment, 8-30 September 2026")
    ax.legend(loc="best", fontsize=8)
    _mark_event(ax)

    if "x_n" in frame:
        ax2.bar(frame["date"], frame["x_n"], color=PALETTE["risk"], alpha=0.45,
                width=0.7, label="Tweets")
    if "news_n" in frame:
        ax2.bar(frame["date"], frame["news_n"], color=PALETTE["news"], alpha=0.55,
                width=0.4, label="Articles")
    ax2.set_ylabel("Volume")
    ax2.legend(loc="upper right", fontsize=7.5, ncol=2)
    _format_dates(ax2)

    path = output_dir / "figure1_sentiment_trend.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure2_family_divergence(
    risk: pd.DataFrame | None,
    company: pd.DataFrame | None,
    output_dir: Path = OUTPUT_DIR,
) -> Path | None:
    """Existential-risk talk against commercial AI talk.

    The interesting case is divergence: fear about AI rising while sentiment
    toward the AI *trade* holds up. Blending the two families into one index
    would hide exactly this.
    """
    if risk is None or company is None:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(risk["date"], risk["x_ratio"], marker="o", markersize=3.2,
            color=PALETTE["risk"], label="AI-risk talk", linewidth=1.5)
    ax.plot(company["date"], company["x_ratio"], marker="s", markersize=3.2,
            color=PALETTE["company"], label="AI-company talk", linewidth=1.5)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_ylabel("Sentiment ratio")
    ax.set_title("Figure 2. Existential-risk sentiment vs. AI-trade sentiment")
    ax.legend(loc="best", fontsize=8)
    _mark_event(ax)
    _format_dates(ax)

    path = output_dir / "figure2_family_divergence.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure3_cumulative_returns(
    panel: pd.DataFrame,
    basket_tickers: list[str],
    benchmark: str = "SPY",
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Cumulative basket return against the benchmark across the window."""
    window = panel.sort_values("date")
    basket = (
        window[window["ticker"].isin(basket_tickers)]
        .groupby("date")["ret"].mean()
    )
    bench = window[window["ticker"] == benchmark].set_index("date")["ret"]

    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.plot(basket.index, (1 + basket.fillna(0)).cumprod() - 1,
            color=PALETTE["basket"], linewidth=1.8, label="AI basket (EW)")
    ax.plot(bench.index, (1 + bench.fillna(0)).cumprod() - 1,
            color=PALETTE["benchmark"], linewidth=1.5, linestyle="--",
            label=f"{benchmark}")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Cumulative return")
    ax.set_title("Figure 3. AI basket vs. benchmark")
    ax.legend(loc="best", fontsize=8)
    _mark_event(ax)
    _format_dates(ax)

    path = output_dir / "figure3_cumulative_returns.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure4_media_bias(
    by_bucket: pd.DataFrame,
    by_region: pd.DataFrame | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Negativity by outlet type, and by author region where available."""
    has_region = by_region is not None and len(by_region) > 0
    fig, axes = plt.subplots(1, 2 if has_region else 1,
                             figsize=(7.2 if has_region else 4.2, 2.6))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    buckets = by_bucket[by_bucket["bucket"] != "Other"]
    colors = [PALETTE["us"], PALETTE["nonus"], PALETTE["tech"]][: len(buckets)]
    bars = ax.bar(buckets["bucket"], buckets["neg_share"], color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.0f%%",
                 labels=[f"{v:.0%}" for v in buckets["neg_share"]], fontsize=8)
    ax.set_ylabel("Share of headlines negative")
    ax.set_ylim(0, max(buckets["neg_share"].max() * 1.25, 0.1))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("News outlets")
    ax.tick_params(axis="x", labelsize=8)

    if has_region:
        ax2 = axes[1]
        region = by_region[by_region["region"] != "Unknown"]
        bars2 = ax2.bar(region["region"], region["neg_share"],
                        color=[PALETTE["us"], PALETTE["nonus"]][: len(region)],
                        width=0.5)
        ax2.bar_label(bars2, labels=[f"{v:.0%}" for v in region["neg_share"]],
                      fontsize=8)
        ax2.set_ylim(0, max(region["neg_share"].max() * 1.25, 0.1))
        ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax2.set_title("X users, by profile region")
        ax2.tick_params(axis="x", labelsize=8)

    fig.suptitle("Figure 4. Negativity bias across sources", fontweight="bold",
                 fontsize=10, y=1.02)
    path = output_dir / "figure4_media_bias.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_all(results: dict, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    """Every figure the report needs, keyed by name."""
    sentiment = results["sentiment"]
    figures: dict[str, Path] = {}

    figures["trend"] = figure1_sentiment_trend(sentiment["aligned"], output_dir)

    divergence = figure2_family_divergence(
        sentiment.get("x_daily_ai_risk"),
        sentiment.get("x_daily_ai_company"),
        output_dir,
    )
    if divergence:
        figures["divergence"] = divergence

    figures["returns"] = figure3_cumulative_returns(
        results["window_panel"], results["basket_tickers"], output_dir=output_dir
    )
    figures["bias"] = figure4_media_bias(
        results["bias"]["by_bucket"], results["bias"].get("by_region"), output_dir
    )
    return figures
