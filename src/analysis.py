"""End-to-end analysis: from cached raw data to the three exhibits.

Kept separate from the scripts so that the notebook, the PDF builder and the
command line all run the *same* code path. A result that appears in the report
but not in the notebook would be indefensible, and the only way to guarantee
that cannot happen is to have one implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loader import basket_aggregate, build_market_panel
from .granger_causality import (
    adf_test,
    align_to_trading_days,
    granger_test,
    panel_granger_test,
    run_var_granger,
)
from .media_bias import (
    daily_negativity,
    daily_sentiment_by_region,
    negativity_asymmetry,
    negativity_by_bucket,
    negativity_by_outlet,
    negativity_vs_fundamentals,
    sentiment_by_region,
)
from .nlp_sentiment import bucket_outlet, daily_sentiment, score_news, score_tweets
from .table_generator import table1_summary, table2_granger, table3_media_bias

LM_DICT = config.LEXICON_DIR / "LoughranMcDonald_MasterDictionary.csv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-name panel and the equal-weighted basket, from cache."""
    panel = pd.read_csv(config.INTERIM_DIR / "market_panel.csv", parse_dates=["date"])
    basket = basket_aggregate(panel, config.AI_BASKET)
    return panel, basket


def load_tweets(refresh: bool = False) -> pd.DataFrame | None:
    """Scored tweets, or ``None`` if the social channel was never harvested.

    Scoring 27,000 posts with VADER is the slowest step in the pipeline, so
    the scored frame is cached. ``refresh=True`` forces a rescore, which is
    what you want after changing :data:`FINANCE_PATCH` or the classifier.

    Returning ``None`` rather than raising lets the rest of the pipeline run
    on the news channel alone, which is what happens when the twitterapi.io
    balance is exhausted mid-harvest.
    """
    cache = config.INTERIM_DIR / "tweets_scored.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache)

    shard_dir = config.SOCIAL_DIR / "raw"
    shards = sorted(shard_dir.glob("*.csv")) if shard_dir.exists() else []
    frames = [pd.read_csv(s) for s in shards if s.stat().st_size > 0]
    if not frames:
        return None
    tweets = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id", "family"])
    scored = score_tweets(tweets)
    scored.to_csv(cache, index=False)
    return scored


def load_news(refresh: bool = False) -> pd.DataFrame:
    """Scored headlines, with outlet buckets recomputed from the current taxonomy.

    The ``bucket`` column cached in ``news.csv`` was written at fetch time. The
    taxonomy in :mod:`config` has been revised since (the curated lists were
    extended and a country-code TLD fallback added, taking classified coverage
    from 29% to 61% of the corpus), so the column is recomputed here rather
    than trusted. Re-deriving is cheap; silently analysing a stale taxonomy is
    not.
    """
    cache = config.INTERIM_DIR / "news_scored.csv"
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache)
        # Buckets are re-derived even from cache: the taxonomy is the thing
        # most likely to have changed between runs, and it is cheap to redo.
        cached["bucket"] = cached["domain"].map(bucket_outlet)
        return cached

    news = pd.read_csv(config.NEWS_DIR / "news.csv")
    news["bucket"] = news["domain"].map(bucket_outlet)
    scored = score_news(news, LM_DICT)
    scored.to_csv(cache, index=False)
    return scored


# ---------------------------------------------------------------------------
# Sentiment series
# ---------------------------------------------------------------------------
def build_sentiment_series(
    tweets: pd.DataFrame | None,
    news: pd.DataFrame,
    trading_days: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Daily sentiment, both channels, both raw and trading-day aligned."""
    out: dict[str, pd.DataFrame] = {}

    if tweets is not None and len(tweets):
        out["x_daily"] = daily_sentiment(tweets, prefix="x")
        for family, group in tweets.groupby("family"):
            out[f"x_daily_{family}"] = daily_sentiment(group, prefix="x")

    out["news_daily"] = daily_sentiment(news, prefix="news")
    neg = daily_negativity(news, prefix="news")
    out["news_daily"] = out["news_daily"].merge(
        neg[["date", "news_neg_share"]], on="date", how="left"
    )

    combined = out["news_daily"]
    if "x_daily" in out:
        combined = out["x_daily"].merge(out["news_daily"], on="date", how="outer")
    combined = combined.sort_values("date").reset_index(drop=True)
    out["combined"] = combined
    out["aligned"] = align_to_trading_days(combined, trading_days)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_granger(
    panel: pd.DataFrame,
    basket: pd.DataFrame,
    aligned: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Panel and single-series Granger tests over every available driver."""
    drivers = [c for c in ("x_ratio", "news_ratio", "news_neg_share") if c in aligned]
    targets = ("basket_excess_ret", "basket_vol")

    merged = basket.merge(aligned, on="date", how="inner")

    single_rows, panel_rows, diagnostics = [], [], []
    for series in list(targets) + drivers:
        if series in merged:
            diagnostics.append(adf_test(merged[series], series))

    for target in targets:
        for driver in drivers:
            if target not in merged or driver not in merged:
                continue
            f_rows = granger_test(merged, target, driver, config.MAX_LAG)
            wald = run_var_granger(merged, target, driver, config.MAX_LAG)
            for _, row in f_rows.iterrows():
                rec = row.to_dict()
                rec["wald_stat"] = wald["wald_stat"] if row["lag"] == wald["lag"] else np.nan
                rec["wald_p"] = wald["p_value"] if row["lag"] == wald["lag"] else np.nan
                single_rows.append(rec)

    firm_targets = {"basket_excess_ret": "excess_ret", "basket_vol": "vol_proxy"}
    firm_panel = panel[panel["ticker"].isin(config.AI_BASKET)]
    for target, column in firm_targets.items():
        for driver in drivers:
            result = panel_granger_test(
                firm_panel, aligned, column, driver, max_lag=min(2, config.MAX_LAG)
            )
            result["target"] = target
            panel_rows.append(result)

    return {
        "single": pd.DataFrame(single_rows),
        "panel": pd.DataFrame(panel_rows),
        "diagnostics": pd.DataFrame(diagnostics),
    }


def run_media_bias(
    news: pd.DataFrame,
    basket: pd.DataFrame,
    tweets: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    daily = daily_negativity(news)
    out = {
        "by_bucket": negativity_by_bucket(news),
        "by_outlet": negativity_by_outlet(news),
        "asymmetry": negativity_asymmetry(news, basket),
        "fundamentals": negativity_vs_fundamentals(daily, basket),
        "daily": daily,
    }
    # The social-channel analogue of the outlet split: US vs non-US authors
    # talking about the same event on the same platform.
    if tweets is not None and "region" in tweets:
        out["by_region"] = sentiment_by_region(tweets)
        out["daily_by_region"] = daily_sentiment_by_region(tweets)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_all(output_dir: Path | None = None) -> dict:
    """Run the whole study and write every exhibit. Returns the results dict."""
    output_dir = Path(output_dir or config.OUTPUT_DIR)

    panel, basket = load_market()
    tweets = load_tweets()
    news = load_news()

    in_window = basket[
        basket["date"].between(config.SAMPLE_START, config.SAMPLE_END)
    ]
    trading_days = in_window["date"]

    sentiment = build_sentiment_series(tweets, news, trading_days)
    aligned = sentiment["aligned"]

    granger = run_granger(panel, basket, aligned)
    bias = run_media_bias(news, basket, tweets)

    window_panel = panel[panel["date"].between(config.SAMPLE_START, config.SAMPLE_END)]
    exhibits = [
        table1_summary(
            window_panel, aligned, config.AI_BASKET,
            config.SEMICONDUCTOR, config.HYPERSCALER,
        ),
        table2_granger(granger["single"], granger["panel"]),
        table3_media_bias(bias["by_bucket"], bias["asymmetry"], bias["fundamentals"]),
    ]
    for exhibit in exhibits:
        exhibit.save(output_dir)

    aligned.to_csv(output_dir / "sentiment_aligned.csv", index=False)
    sentiment["combined"].to_csv(output_dir / "sentiment_daily.csv", index=False)
    granger["diagnostics"].to_csv(output_dir / "adf_diagnostics.csv", index=False)
    bias["by_outlet"].to_csv(output_dir / "negativity_by_outlet.csv", index=False)

    if "by_region" in bias:
        bias["by_region"].to_csv(output_dir / "sentiment_by_region.csv", index=False)

    return {
        "panel": panel, "basket": basket, "window_panel": window_panel,
        "basket_tickers": config.AI_BASKET,
        "tweets": tweets, "news": news, "sentiment": sentiment,
        "granger": granger, "bias": bias, "exhibits": exhibits,
    }
