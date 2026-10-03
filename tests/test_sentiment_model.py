"""Tests for the sentiment scorers."""

from __future__ import annotations

import pytest

from src.sentiment_model import (
    FINANCE_PATCH,
    LoughranMcDonaldScorer,
    VaderScorer,
    classify,
    clean_text,
    sentiment_ratio,
)

LM_DICT = "data/lexicons/LoughranMcDonald_MasterDictionary.csv"


@pytest.fixture(scope="module")
def vader() -> VaderScorer:
    return VaderScorer()


class TestCleanText:
    def test_strips_urls_and_mentions(self):
        out = clean_text("@elonmusk AI is risky https://t.co/abc123")
        assert "@elonmusk" not in out
        assert "http" not in out
        assert "AI is risky" in out

    def test_preserves_case_and_punctuation(self):
        # VADER reads caps and "!" as intensity; normalising them loses signal.
        out = clean_text("AI is TERRIFYING!!!")
        assert "TERRIFYING" in out
        assert "!!!" in out

    def test_handles_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(float("nan")) == ""


class TestFinancePatch:
    def test_repairs_the_canonical_failure(self, vader):
        """Unpatched VADER scores this negative; it is unambiguously bullish.

        This is the regression test for the whole patch: "crushed" is violent
        in ordinary English and triumphant in markets.
        """
        assert vader.score("Nvidia earnings crushed it, bullish") > 0.3

    def test_ai_risk_terms_score_negative(self, vader):
        assert vader.score("my p(doom) just went up, extinction risk is real") < 0
        assert vader.score("this model is misaligned and dangerous") < 0

    def test_ambiguous_position_words_are_not_patched(self):
        """"short"/"long" flip sign with context, so they must stay unpatched."""
        assert "short" not in FINANCE_PATCH
        assert "long" not in FINANCE_PATCH

    def test_patch_actually_reaches_the_analyzer(self, vader):
        assert vader._analyzer.lexicon["bullish"] == FINANCE_PATCH["bullish"]

    def test_scores_stay_in_range(self, vader):
        for text in ["AI will kill us all", "amazing breakthrough!!!", ""]:
            assert -1.0 <= vader.score(text) <= 1.0


class TestLoughranMcDonald:
    @pytest.fixture(scope="class")
    def lm(self) -> LoughranMcDonaldScorer:
        return LoughranMcDonaldScorer(LM_DICT)

    def test_positive_and_negative_text(self, lm):
        assert lm.score("Revenue growth was strong and profitable") > 0
        assert lm.score("The company faces litigation and losses") < 0

    def test_negation_flips_polarity(self, lm):
        """Raw LM counts miss "not profitable"; the negation window fixes it."""
        plain = lm.score("The company was profitable")
        negated = lm.score("The company was not profitable")
        assert plain > 0
        assert negated < plain

    def test_neutral_text_scores_zero(self, lm):
        assert lm.score("The meeting is scheduled for Tuesday at noon") == 0.0

    def test_empty_text_is_safe(self, lm):
        assert lm.score("") == 0.0
        assert lm.counts("") == (0, 0)


class TestClassifyAndRatio:
    @pytest.mark.parametrize(
        "score,expected",
        [(0.9, "positive"), (0.05, "positive"), (0.0, "neutral"),
         (-0.04, "neutral"), (-0.05, "negative"), (-0.9, "negative")],
    )
    def test_classify_boundaries(self, score, expected):
        assert classify(score) == expected

    def test_ratio_matches_definition(self):
        # (3 - 1) / 5
        labels = ["positive"] * 3 + ["negative"] + ["neutral"]
        assert sentiment_ratio(labels) == pytest.approx(0.4)

    def test_neutrals_stay_in_denominator(self):
        """Dropping neutrals would make these two days look identical."""
        assert sentiment_ratio(["positive", "negative", "neutral"]) == pytest.approx(0.0)
        dense = ["positive"] * 2 + ["negative"]
        assert sentiment_ratio(dense) == pytest.approx(1 / 3)

    def test_ratio_is_bounded(self):
        assert sentiment_ratio(["positive"] * 10) == 1.0
        assert sentiment_ratio(["negative"] * 10) == -1.0

    def test_empty_is_nan(self):
        import math

        assert math.isnan(sentiment_ratio([]))
