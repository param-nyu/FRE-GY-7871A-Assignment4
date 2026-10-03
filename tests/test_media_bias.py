"""Tests for the Sacerdote-style negativity bias measures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.media_bias import (
    _two_proportion_z,
    negativity_asymmetry,
    negativity_by_bucket,
    negativity_vs_fundamentals,
)


def make_news(bucket: str, n_neg: int, n_pos: int, n_neu: int = 0) -> pd.DataFrame:
    labels = ["negative"] * n_neg + ["positive"] * n_pos + ["neutral"] * n_neu
    return pd.DataFrame({
        "bucket": bucket,
        "label": labels,
        "sentiment": [-0.5] * n_neg + [0.5] * n_pos + [0.0] * n_neu,
        "domain": [f"{bucket.lower()}-outlet.com"] * len(labels),
        "date": pd.Timestamp("2026-09-08"),
    })


class TestTwoProportionZ:
    def test_identical_proportions_give_zero(self):
        z, p = _two_proportion_z(50, 100, 50, 100)
        assert z == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_large_gap_is_significant(self):
        z, p = _two_proportion_z(90, 100, 50, 100)
        assert z > 0
        assert p < 0.01

    def test_empty_sample_is_nan(self):
        z, p = _two_proportion_z(0, 0, 10, 20)
        assert np.isnan(z) and np.isnan(p)


class TestNegativityByBucket:
    @pytest.fixture
    def news(self) -> pd.DataFrame:
        # Deliberately mirrors the COVID study's shape: US most negative.
        return pd.concat([
            make_news("US Mainstream", n_neg=91, n_pos=9),
            make_news("Non-US", n_neg=54, n_pos=46),
            make_news("Scientific/Tech", n_neg=65, n_pos=35),
        ], ignore_index=True)

    def test_shares_are_computed_correctly(self, news):
        out = negativity_by_bucket(news)
        us = out[out["bucket"] == "US Mainstream"].iloc[0]
        assert us["neg_share"] == pytest.approx(0.91)

    def test_us_vs_non_us_gap_is_significant(self, news):
        out = negativity_by_bucket(news)
        non_us = out[out["bucket"] == "Non-US"].iloc[0]
        assert non_us["p_vs_us"] < 0.01

    def test_reference_bucket_has_no_self_test(self, news):
        out = negativity_by_bucket(news)
        us = out[out["bucket"] == "US Mainstream"].iloc[0]
        assert np.isnan(us["p_vs_us"])

    def test_thin_buckets_are_reported_but_not_tested(self):
        news = pd.concat([
            make_news("US Mainstream", 50, 50),
            make_news("Non-US", 3, 1),
        ], ignore_index=True)
        out = negativity_by_bucket(news, min_articles=20)
        thin = out[out["bucket"] == "Non-US"].iloc[0]
        assert thin["n_articles"] == 4
        assert np.isnan(thin["p_vs_us"])

    def test_neg_of_polar_excludes_neutrals(self):
        news = make_news("US Mainstream", n_neg=30, n_pos=10, n_neu=60)
        out = negativity_by_bucket(news, reference="US Mainstream")
        row = out.iloc[0]
        assert row["neg_share"] == pytest.approx(0.30)
        assert row["neg_of_polar"] == pytest.approx(0.75)

    def test_shares_sum_to_one(self):
        news = make_news("US Mainstream", 30, 10, 60)
        row = negativity_by_bucket(news).iloc[0]
        assert row["neg_share"] + row["pos_share"] + row["neu_share"] == pytest.approx(1.0)


class TestAsymmetry:
    def test_splits_on_up_and_down_days(self):
        news = pd.DataFrame({
            "date": pd.to_datetime(["2026-09-08"] * 4 + ["2026-09-09"] * 4),
            "label": ["negative"] * 3 + ["positive"] + ["negative"] * 2 + ["positive"] * 2,
            "sentiment": [-0.5] * 3 + [0.5] + [-0.5] * 2 + [0.5] * 2,
        })
        basket = pd.DataFrame({
            "date": pd.to_datetime(["2026-09-08", "2026-09-09"]),
            "basket_excess_ret": [0.02, -0.01],
        })
        out = negativity_asymmetry(news, basket)
        up = out[out["direction"] == "Up days"].iloc[0]
        assert up["neg_pos_ratio"] == pytest.approx(3.0)


class TestNegativityVsFundamentals:
    def test_detects_a_real_relationship(self):
        n = 30
        daily = pd.DataFrame({
            "date": pd.date_range("2026-09-01", periods=n),
            "news_neg_share": np.linspace(0.2, 0.9, n),
        })
        basket = pd.DataFrame({
            "date": daily["date"],
            "basket_excess_ret": -np.linspace(0.2, 0.9, n),
        })
        out = negativity_vs_fundamentals(daily, basket, fundamentals=("basket_excess_ret",))
        assert out["pearson_r"].iloc[0] == pytest.approx(-1.0, abs=1e-6)

    def test_reports_nan_on_tiny_sample(self):
        daily = pd.DataFrame({
            "date": pd.date_range("2026-09-01", periods=3),
            "news_neg_share": [0.1, 0.2, 0.3],
        })
        basket = pd.DataFrame({"date": daily["date"], "basket_excess_ret": [0.0, 0.1, 0.2]})
        out = negativity_vs_fundamentals(daily, basket, fundamentals=("basket_excess_ret",))
        assert np.isnan(out["pearson_r"].iloc[0])
