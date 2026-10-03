"""Tests for the harvest filters, relevance scoring and outlet taxonomy."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.nlp_sentiment import (
    _day_range,
    _domain_of,
    _tweet_passes_filters,
    _unix,
    bucket_outlet,
    daily_sentiment,
    relevance_score,
    relevance_weight,
)


def make_tweet(**overrides) -> dict:
    """A tweet that passes every filter, so each test can break exactly one."""
    tweet = {
        "text": "AI safety is important",
        "viewCount": 500,
        "isReply": False,
        "author": {"followers": 100, "userName": "someone"},
    }
    tweet.update(overrides)
    return tweet


class TestQualityFilters:
    def test_baseline_tweet_passes(self):
        assert _tweet_passes_filters(make_tweet())

    def test_retweets_excluded(self):
        assert not _tweet_passes_filters(make_tweet(retweeted_tweet={"id": "1"}))

    def test_low_view_count_excluded(self):
        assert not _tweet_passes_filters(make_tweet(viewCount=99))
        assert _tweet_passes_filters(make_tweet(viewCount=100))  # boundary is inclusive

    def test_low_follower_count_excluded(self):
        assert not _tweet_passes_filters(make_tweet(author={"followers": 9}))
        assert _tweet_passes_filters(make_tweet(author={"followers": 10}))

    def test_empty_text_excluded(self):
        assert not _tweet_passes_filters(make_tweet(text="   "))

    def test_missing_fields_do_not_crash(self):
        assert not _tweet_passes_filters({"text": "hi"})


class TestDateHandling:
    def test_day_range_is_inclusive(self):
        days = _day_range("2026-09-08", "2026-09-30")
        assert len(days) == 23
        assert days[0] == dt.date(2026, 9, 8)
        assert days[-1] == dt.date(2026, 9, 30)

    def test_unix_bounds_span_exactly_one_day(self):
        day = dt.date(2026, 9, 8)
        assert _unix(day, end_of_day=True) - _unix(day) == 86_400

    def test_unix_is_utc_midnight(self):
        ts = _unix(dt.date(2026, 9, 8))
        assert dt.datetime.fromtimestamp(ts, dt.timezone.utc).hour == 0


class TestOutletTaxonomy:
    @pytest.mark.parametrize(
        "domain,expected",
        [
            ("nytimes.com", "US Mainstream"),
            ("www.nytimes.com", "US Mainstream"),
            ("bbc.co.uk", "Non-US"),
            ("theguardian.com", "Non-US"),
            ("techcrunch.com", "Scientific/Tech"),
            ("nature.com", "Scientific/Tech"),
            ("some-random-blog.xyz", "Other"),
        ],
    )
    def test_bucketing(self, domain, expected):
        assert bucket_outlet(domain) == expected

    def test_subdomains_resolve_to_parent(self):
        assert bucket_outlet("edition.cnn.com") == "US Mainstream"

    def test_domain_extraction_strips_www(self):
        assert _domain_of("https://www.nytimes.com/2026/09/08/ai.html") == "nytimes.com"

    def test_unknown_domains_are_kept_not_dropped(self):
        """Falling through to Other matters: a silent drop would bias the split."""
        assert bucket_outlet("") == "Other"


class TestRelevanceScoring:
    def test_headline_mention_scores_high(self):
        assert relevance_score("Nvidia stock surges on AI demand") >= 70

    def test_body_only_mention_scores_lower(self):
        headline = relevance_score("Nvidia earnings beat")
        body_only = relevance_score("Markets rose today", "Nvidia was mentioned")
        assert body_only < headline

    def test_off_topic_scores_zero(self):
        assert relevance_score("Local bakery wins award") == 0

    def test_entity_plus_risk_gets_bonus(self):
        both = relevance_score("Nvidia faces AI safety scrutiny")
        entity_only = relevance_score("Nvidia reports revenue")
        assert both > entity_only

    def test_score_is_bounded(self):
        crowded = "Nvidia Meta Amazon Google AI safety existential extinction doom"
        assert 0 <= relevance_score(crowded) <= 100

    def test_weight_is_full_above_threshold(self):
        assert relevance_weight(90) == 1.0
        assert relevance_weight(100) == 1.0

    def test_weight_is_zero_below_minimum(self):
        assert relevance_weight(10) == 0.0

    def test_weight_tapers_in_between(self):
        mid = relevance_weight(60)
        assert 0.0 < mid < 1.0


class TestDailyAggregation:
    @pytest.fixture
    def scored(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": ["2026-09-08"] * 4 + ["2026-09-09"] * 2,
            "sentiment": [0.5, 0.6, -0.4, 0.0, -0.8, -0.2],
            "label": ["positive", "positive", "negative", "neutral",
                      "negative", "negative"],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        })

    def test_ratio_per_day(self, scored):
        out = daily_sentiment(scored, prefix="x")
        day1 = out[out["date"] == pd.Timestamp("2026-09-08")].iloc[0]
        assert day1["x_ratio"] == pytest.approx((2 - 1) / 4)
        day2 = out[out["date"] == pd.Timestamp("2026-09-09")].iloc[0]
        assert day2["x_ratio"] == pytest.approx(-1.0)

    def test_counts_sum_to_total(self, scored):
        out = daily_sentiment(scored, prefix="x")
        assert (out["x_pos"] + out["x_neg"] + out["x_neu"] == out["x_n"]).all()

    def test_output_is_sorted_by_date(self, scored):
        out = daily_sentiment(scored, prefix="x")
        assert out["date"].is_monotonic_increasing
