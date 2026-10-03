"""Media negativity bias, after Sacerdote, Sehgal & Cook (2020).

The COVID-19 study ("Why Is All COVID-19 News Bad News?", NBER w28110) found
that 91% of US major-outlet stories about COVID were negative, against 54% for
non-US English-language outlets and 65% for scientific journals, and -- the
part that makes it a *bias* finding rather than a tone finding -- that this
negativity was essentially uncorrelated with the underlying case trend.

This module applies the same three tests to AI-risk coverage:

1. :func:`negativity_by_bucket` -- the headline comparison. Negativity share
   across US Mainstream / Non-US / Scientific-Tech, with a two-proportion
   z-test against the US bucket.
2. :func:`negativity_vs_fundamentals` -- the "uncorrelated with the data"
   test. Sacerdote's benchmark was case counts; the analogue here is the AI
   basket's own realised return, which is the observable fundamental the
   coverage is ostensibly about. If negativity tracks neither returns nor
   volatility, tone is not being driven by what is happening in the sector.
3. :func:`negativity_asymmetry` -- the 5.5x finding. Sacerdote showed negative
   case-count stories outnumbered positive ones even while cases were
   *falling*. Here: the ratio of negative to positive coverage on days when
   the basket actually rose.

A caution carried into the report: the original study classified full article
text, while this one classifies headlines. Headlines are systematically more
negative than the bodies they head, so the *levels* here are not comparable to
Sacerdote's 91%. The *differences between buckets* are the finding, since the
headline bias applies to every bucket alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BUCKET_ORDER = ("US Mainstream", "Non-US", "Scientific/Tech", "Other")


def _two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-sided z-test for equality of two proportions (pooled variance)."""
    if min(n1, n2) == 0:
        return np.nan, np.nan
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def negativity_by_bucket(
    news: pd.DataFrame,
    label_col: str = "label",
    bucket_col: str = "bucket",
    reference: str = "US Mainstream",
    min_articles: int = 20,
) -> pd.DataFrame:
    """Negativity share by outlet bucket, tested against the US baseline.

    ``neg_share`` is negatives over *all* articles in the bucket, which is the
    quantity Sacerdote reports. ``neg_of_polar`` excludes neutrals and is given
    alongside it, because a bucket that simply writes more neutral headlines
    would otherwise look less negative than it is.

    Buckets thinner than ``min_articles`` are reported but not tested; a
    z-test on a handful of headlines would be noise dressed as a result.
    """
    rows = []
    ref = news[news[bucket_col] == reference]
    ref_neg, ref_n = int((ref[label_col] == "negative").sum()), len(ref)

    for bucket in [b for b in BUCKET_ORDER if b in set(news[bucket_col])]:
        sub = news[news[bucket_col] == bucket]
        n = len(sub)
        n_neg = int((sub[label_col] == "negative").sum())
        n_pos = int((sub[label_col] == "positive").sum())
        n_neu = n - n_neg - n_pos
        polar = n_neg + n_pos

        row = {
            "bucket": bucket,
            "n_articles": n,
            "n_outlets": sub["domain"].nunique() if "domain" in sub else np.nan,
            "neg_share": n_neg / n if n else np.nan,
            "pos_share": n_pos / n if n else np.nan,
            "neu_share": n_neu / n if n else np.nan,
            "neg_of_polar": n_neg / polar if polar else np.nan,
            "mean_sentiment": sub["sentiment"].mean() if "sentiment" in sub else np.nan,
            "neg_pos_ratio": (n_neg / n_pos) if n_pos else np.nan,
        }
        if bucket != reference and n >= min_articles and ref_n >= min_articles:
            z, p = _two_proportion_z(ref_neg, ref_n, n_neg, n)
            row["z_vs_us"] = z
            row["p_vs_us"] = p
        else:
            row["z_vs_us"] = np.nan
            row["p_vs_us"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def negativity_by_outlet(
    news: pd.DataFrame,
    label_col: str = "label",
    min_articles: int = 10,
) -> pd.DataFrame:
    """Per-outlet negativity, for the "which outlet is most negative" result.

    Sacerdote named the New York Times the most negative outlet in their
    sample; this is the equivalent ranking, restricted to outlets with enough
    coverage to rank meaningfully.
    """
    grouped = news.groupby(["domain", "bucket"], dropna=False)
    out = grouped.agg(
        n_articles=(label_col, "size"),
        n_negative=(label_col, lambda s: int((s == "negative").sum())),
        mean_sentiment=("sentiment", "mean"),
    ).reset_index()
    out = out[out["n_articles"] >= min_articles].copy()
    out["neg_share"] = out["n_negative"] / out["n_articles"]
    return out.sort_values("neg_share", ascending=False).reset_index(drop=True)


def negativity_vs_fundamentals(
    daily_news: pd.DataFrame,
    basket: pd.DataFrame,
    neg_col: str = "news_neg_share",
    fundamentals: tuple[str, ...] = ("basket_excess_ret", "basket_ret", "basket_vol"),
) -> pd.DataFrame:
    """Correlate daily negativity against what the sector actually did.

    This is the heart of the bias claim. If coverage tone responded to
    fundamentals, negativity would fall on days the basket rallied. Both
    Pearson and Spearman are reported -- with ~16 trading days a single
    outlier can drive Pearson, and the rank correlation is the more honest
    summary at this sample size.
    """
    merged = daily_news.merge(basket, on="date", how="inner")
    rows = []
    for col in fundamentals:
        if col not in merged:
            continue
        pair = merged[[neg_col, col]].dropna()
        row = {"fundamental": col, "n": len(pair), "pearson_r": np.nan,
               "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
        if len(pair) >= 5 and pair[neg_col].nunique() > 1 and pair[col].nunique() > 1:
            r, rp = stats.pearsonr(pair[neg_col], pair[col])
            rho, sp = stats.spearmanr(pair[neg_col], pair[col])
            row.update({"pearson_r": float(r), "pearson_p": float(rp),
                        "spearman_rho": float(rho), "spearman_p": float(sp)})
        rows.append(row)
    return pd.DataFrame(rows)


def negativity_asymmetry(
    news: pd.DataFrame,
    basket: pd.DataFrame,
    label_col: str = "label",
    return_col: str = "basket_excess_ret",
) -> pd.DataFrame:
    """Negative-to-positive coverage ratio on up days versus down days.

    Sacerdote's 5.5x result was that negative case-count stories outnumbered
    positive ones *even while cases fell*. The analogue: if the negative-to-
    positive ratio stays well above 1 on days the basket rose, coverage tone
    is not tracking the sector's fortunes.
    """
    frame = news.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    merged = frame.merge(
        basket[["date", return_col]].assign(date=pd.to_datetime(basket["date"])),
        on="date",
        how="inner",
    )
    merged["direction"] = np.where(merged[return_col] >= 0, "Up days", "Down days")

    rows = []
    for direction, group in merged.groupby("direction"):
        n_neg = int((group[label_col] == "negative").sum())
        n_pos = int((group[label_col] == "positive").sum())
        rows.append({
            "direction": direction,
            "n_days": group["date"].nunique(),
            "n_articles": len(group),
            "n_negative": n_neg,
            "n_positive": n_pos,
            "neg_pos_ratio": (n_neg / n_pos) if n_pos else np.nan,
            "neg_share": n_neg / len(group) if len(group) else np.nan,
            "mean_sentiment": group["sentiment"].mean(),
        })
    return pd.DataFrame(rows).sort_values("direction").reset_index(drop=True)


def sentiment_by_region(
    tweets: pd.DataFrame,
    label_col: str = "label",
    region_col: str = "region",
    reference: str = "US",
    min_tweets: int = 50,
) -> pd.DataFrame:
    """US vs non-US sentiment on the social channel.

    The lecture asked for this split alongside the news-outlet one, and it is
    the sharper test of the two: the news comparison comes from different
    *organisations* with different editorial incentives, whereas this compares
    individuals talking about the same event on the same platform, so a gap
    here is harder to attribute to anything but audience.

    ``Unknown`` is reported as its own row rather than dropped. Profile
    location is self-reported free text and missing for a large minority of
    accounts; hiding that would overstate how much of the sample the split
    actually covers.
    """
    rows = []
    ref = tweets[tweets[region_col] == reference]
    ref_neg, ref_n = int((ref[label_col] == "negative").sum()), len(ref)

    for region in ("US", "Non-US", "Unknown"):
        sub = tweets[tweets[region_col] == region]
        n = len(sub)
        if n == 0:
            continue
        n_neg = int((sub[label_col] == "negative").sum())
        n_pos = int((sub[label_col] == "positive").sum())
        row = {
            "region": region,
            "n_tweets": n,
            "share_of_sample": n / len(tweets),
            "n_authors": sub["author"].nunique() if "author" in sub else np.nan,
            "neg_share": n_neg / n,
            "pos_share": n_pos / n,
            "sentiment_ratio": (n_pos - n_neg) / n,
            "mean_sentiment": sub["sentiment"].mean(),
        }
        if region != reference and n >= min_tweets and ref_n >= min_tweets:
            z, p = _two_proportion_z(ref_neg, ref_n, n_neg, n)
            row["z_vs_us"], row["p_vs_us"] = z, p
        else:
            row["z_vs_us"] = row["p_vs_us"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def daily_sentiment_by_region(
    tweets: pd.DataFrame,
    region_col: str = "region",
    label_col: str = "label",
) -> pd.DataFrame:
    """Daily sentiment ratio split by author region, for the trend chart."""
    frame = tweets.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["signed"] = frame[label_col].map(
        {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    )
    out = (
        frame.pivot_table(index="date", columns=region_col, values="signed", aggfunc="mean")
        .add_prefix("ratio_")
    )
    counts = (
        frame.pivot_table(index="date", columns=region_col, values="signed", aggfunc="size")
        .add_prefix("n_")
    )
    return out.join(counts).reset_index()


def daily_negativity(
    news: pd.DataFrame,
    label_col: str = "label",
    prefix: str = "news",
) -> pd.DataFrame:
    """Daily negativity share overall and per bucket, for the time-series plot."""
    frame = news.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    overall = frame.groupby("date").agg(
        **{
            f"{prefix}_n": (label_col, "size"),
            f"{prefix}_neg": (label_col, lambda s: int((s == "negative").sum())),
            f"{prefix}_pos": (label_col, lambda s: int((s == "positive").sum())),
        }
    )
    overall[f"{prefix}_neg_share"] = overall[f"{prefix}_neg"] / overall[f"{prefix}_n"]

    per_bucket = (
        frame.assign(is_neg=(frame[label_col] == "negative").astype(float))
        .pivot_table(index="date", columns="bucket", values="is_neg", aggfunc="mean")
        .add_prefix("neg_share_")
    )
    return overall.join(per_bucket).reset_index()
