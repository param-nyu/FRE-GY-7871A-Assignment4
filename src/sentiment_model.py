"""Sentiment scoring, with a domain patch for financial and AI-risk language.

Two scorers, deliberately matched to two different kinds of text:

``VaderScorer``
    For tweets. VADER handles the things that matter in informal text --
    negation, intensifiers, emoji, capitalisation, punctuation -- which a bag-
    of-words lexicon cannot.

``LoughranMcDonaldScorer``
    For news. LM is the finance-standard word list and is what the companion
    10-K assignment scores with, so the news channel stays comparable to it.

**Why the patch exists.** Off-the-shelf VADER mis-signs financial text. The
canonical failure, reproduced in ``tests/test_sentiment_model.py``:

    "Nvidia earnings crushed it, bullish"  ->  compound = -0.42

"crushed" is violent in ordinary English and triumphant in markets, and
"bullish" is not in VADER's lexicon at all. :data:`FINANCE_PATCH` repairs the
terms that actually recur in this corpus. Only unambiguous terms are patched:
"short" and "long" are left alone precisely because their sign depends on
whether they are being used as verbs, adjectives or position descriptions.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# VADER's lexicon is on a roughly [-4, +4] valence scale.
FINANCE_PATCH: dict[str, float] = {
    # --- market direction ---
    "bullish": 2.6, "bearish": -2.6, "rally": 2.0, "rallied": 2.0,
    "surge": 2.2, "surged": 2.2, "soar": 2.5, "soared": 2.5,
    "plunge": -2.5, "plunged": -2.5, "crash": -3.0, "crashed": -3.0,
    "selloff": -2.0, "sell-off": -2.0, "tanked": -2.5, "slump": -2.0,
    "breakout": 2.0, "squeeze": 1.5, "melt-up": 2.2,
    # --- earnings / analyst language ---
    "beat": 2.0, "blowout": 2.5, "upgrade": 2.0, "upgraded": 2.0,
    "downgrade": -2.0, "downgraded": -2.0, "outperform": 2.0,
    "underperform": -2.0, "overvalued": -2.0, "undervalued": 1.8,
    "guidance": 0.0, "capex": 0.0,
    # "crushed"/"smashed" are violent in general English, triumphant in markets
    "crushed": 2.2, "smashed": 2.0, "crushing": 1.8,
    # --- retail / social register ---
    "moon": 2.0, "mooning": 2.2, "tendies": 2.5, "rekt": -3.0,
    "bagholder": -2.3, "bagholding": -2.3, "dip": -0.8, "bubble": -2.0,
    "moat": 1.5, "layoffs": -2.5, "bagholders": -2.3,
    # --- AI-risk register, central to this event study ---
    "doom": -3.0, "doomer": -2.0, "doomers": -2.0, "p(doom)": -2.5,
    "extinction": -3.0, "existential": -1.8, "misaligned": -2.0,
    "alignment": 0.5, "aligned": 1.0, "superintelligence": -0.5,
    "agi": 0.0, "accelerationist": 0.3, "singularity": -0.3,
}

_WORD_RE = re.compile(r"[a-z']+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str, drop_mentions: bool = True) -> str:
    """Strip URLs, @-handles and excess whitespace; keep emoji and casing.

    Casing and punctuation survive on purpose: VADER reads ALL-CAPS and "!!!"
    as intensity signals, so normalising them away would discard signal.
    """
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    if drop_mentions:
        text = _MENTION_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# VADER
# ---------------------------------------------------------------------------
class VaderScorer:
    """VADER compound score in [-1, +1], with :data:`FINANCE_PATCH` applied."""

    def __init__(self, patch: dict[str, float] | None = None) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()
        self.patch = FINANCE_PATCH if patch is None else patch
        self._analyzer.lexicon.update(self.patch)

    def score(self, text: str) -> float:
        cleaned = clean_text(text)
        if not cleaned:
            return 0.0
        return float(self._analyzer.polarity_scores(cleaned)["compound"])

    def score_many(self, texts) -> pd.Series:
        return pd.Series([self.score(t) for t in texts], dtype=float)


# ---------------------------------------------------------------------------
# Loughran-McDonald
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_lm(path: str) -> tuple[frozenset[str], frozenset[str]]:
    """Positive and negative word sets from the LM master dictionary.

    A word is in a category when its column holds the (non-zero) year it was
    first assigned, which is LM's own encoding.
    """
    frame = pd.read_csv(path)
    frame["Word"] = frame["Word"].astype(str).str.lower()
    pos = frozenset(frame.loc[frame["Positive"] > 0, "Word"])
    neg = frozenset(frame.loc[frame["Negative"] > 0, "Word"])
    return pos, neg


class LoughranMcDonaldScorer:
    """Finance-domain tone, with a short negation window.

    The score is the signed proportion of sentiment words,

    .. math:: \\text{tone} = \\frac{n_{pos} - n_{neg}}{n_{pos} + n_{neg}}

    which lands in [-1, +1] and is zero when a document carries no sentiment
    words at all. A negator within :data:`NEGATION_WINDOW` tokens flips the
    polarity of the word it governs -- LM's raw word counts famously miss
    "not profitable", and the fix is cheap.
    """

    NEGATORS = frozenset({
        "not", "no", "never", "none", "neither", "nor", "cannot",
        "can't", "won't", "don't", "doesn't", "didn't", "isn't",
        "aren't", "wasn't", "weren't", "without", "lacks", "lacking",
    })
    NEGATION_WINDOW = 3

    def __init__(self, dictionary_path: str | Path) -> None:
        self.positive, self.negative = _load_lm(str(dictionary_path))

    def counts(self, text: str) -> tuple[int, int]:
        """(positive, negative) word counts after negation flipping."""
        tokens = _WORD_RE.findall(clean_text(text).lower())
        n_pos = n_neg = 0
        for i, tok in enumerate(tokens):
            is_pos, is_neg = tok in self.positive, tok in self.negative
            if not (is_pos or is_neg):
                continue
            window = tokens[max(0, i - self.NEGATION_WINDOW):i]
            if any(w in self.NEGATORS for w in window):
                is_pos, is_neg = is_neg, is_pos
            n_pos += is_pos
            n_neg += is_neg
        return n_pos, n_neg

    def score(self, text: str) -> float:
        n_pos, n_neg = self.counts(text)
        total = n_pos + n_neg
        return 0.0 if total == 0 else (n_pos - n_neg) / total

    def score_many(self, texts) -> pd.Series:
        return pd.Series([self.score(t) for t in texts], dtype=float)


# ---------------------------------------------------------------------------
# Classification and aggregation
# ---------------------------------------------------------------------------
# VADER's conventional neutral band. Documents inside it count toward the
# denominator of the sentiment ratio but toward neither numerator term.
NEUTRAL_BAND = 0.05


def classify(score: float, band: float = NEUTRAL_BAND) -> str:
    """Map a continuous score to ``positive`` / ``negative`` / ``neutral``."""
    if score >= band:
        return "positive"
    if score <= -band:
        return "negative"
    return "neutral"


def sentiment_ratio(labels) -> float:
    """The brief's daily ratio :math:`(Pos - Neg) / Total`, in [-1, +1].

    Neutrals stay in ``Total``. That is the definition used in the retail
    study, and it matters: dropping them would make a day of 2 positive and 1
    negative tweet look identical to a day of 200 and 100.
    """
    labels = list(labels)
    if not labels:
        return float("nan")
    n_pos = sum(1 for x in labels if x == "positive")
    n_neg = sum(1 for x in labels if x == "negative")
    return (n_pos - n_neg) / len(labels)
