"""Central configuration for the Assignment 4 pipeline.

This file defines the sample: which equities, which window, which sentiment
channels, and how news outlets are bucketed for the media-bias replication.
Nothing here reaches the network; the loaders in :mod:`data_loader` and
:mod:`nlp_sentiment` read these constants and do the fetching.

Credentials are read from the environment, never hard-coded. Put them in
``.env`` (git-ignored) and they are loaded on import:

    TWITTERAPI_IO_KEY=...        # https://twitterapi.io, $0.15 per 1k tweets
    ALPHAVANTAGE_KEY=...         # https://www.alphavantage.co/support/#api-key
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRICE_DIR = DATA / "prices"
SOCIAL_DIR = DATA / "social"
NEWS_DIR = DATA / "news"
INTERIM_DIR = DATA / "interim"
LEXICON_DIR = DATA / "lexicons"
OUTPUT_DIR = ROOT / "outputs"

for _d in (PRICE_DIR, SOCIAL_DIR, NEWS_DIR, INTERIM_DIR, LEXICON_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader, so the repo has no hard dependency on python-dotenv.

    Existing environment variables win, which keeps CI and shell overrides
    working without editing the file.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

TWITTERAPI_IO_KEY = os.environ.get("TWITTERAPI_IO_KEY", "")
ALPHAVANTAGE_KEY = os.environ.get("ALPHAVANTAGE_KEY", "")

# A contact string is sent as the User-Agent on every outbound request, the
# same courtesy the SEC requires and GDELT asks for.
USER_AGENT = os.environ.get(
    "RESEARCH_USER_AGENT", "NYU FRE-GY-7871A academic research"
)


# ---------------------------------------------------------------------------
# Event and sample window  (Section 1 of the brief)
# ---------------------------------------------------------------------------
# Evan Hubinger (@EvanHub), Anthropic, posted the "we really do earnestly
# believe AI could kill all humans ... >10% within the next decade" thread.
# This is day 0 of the event study.
EVENT_DATE = "2026-09-08"

SAMPLE_START = "2026-09-08"  # inclusive
SAMPLE_END = "2026-09-30"    # inclusive

# Prices are pulled from well before the window so that returns on the first
# sample day have a prior close, and so the pre-event baseline in Table 1 has
# something to compare against.
PRICE_START = "2026-07-01"
PRICE_END = "2026-10-02"


# ---------------------------------------------------------------------------
# The AI equity basket  (Section 2 of the brief)
# ---------------------------------------------------------------------------
# Eight large-cap names with direct AI revenue exposure, split into the two
# economically distinct roles the report leans on: firms that *sell* AI compute
# and firms that *spend* on it. The brief names Nvidia, Meta, Amazon and Google
# explicitly; the rest complete the hyperscaler/semiconductor pair.
AI_BASKET = ["NVDA", "META", "AMZN", "GOOGL", "MSFT", "AAPL", "AMD", "AVGO"]

SEMICONDUCTOR = ["NVDA", "AMD", "AVGO"]          # AI compute suppliers
HYPERSCALER = ["META", "AMZN", "GOOGL", "MSFT", "AAPL"]  # AI capex spenders

BENCHMARK = "SPY"                 # market proxy for excess returns
RISK_FREE_SERIES = "DGS3MO"       # FRED 3-month Treasury constant maturity

ALL_TICKERS = AI_BASKET + [BENCHMARK]


# ---------------------------------------------------------------------------
# Social-media channel  (twitterapi.io)
# ---------------------------------------------------------------------------
TWITTER_SEARCH_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

# Quality filters mandated by the brief and by the retail-sentiment study the
# lecture reviewed. Retweets are excluded in the query string itself; the two
# numeric floors are applied after the fact because the API does not expose
# them as search operators.
MIN_FOLLOWERS = 10
MIN_VIEWS = 100
EXCLUDE_REPLIES = False  # replies carry genuine sentiment; retweets do not

# Two query families. The risk family measures fear about AI itself; the
# company family measures sentiment attached to the basket. Keeping them
# separate lets the report show that the two move differently.
TWITTER_QUERIES: dict[str, str] = {
    "ai_risk": (
        '("AI risk" OR "AI safety" OR "AI doom" OR "existential risk" OR '
        'superintelligence OR "AI alignment" OR "kill all humans" OR "p(doom)" '
        'OR "AI extinction" OR doomer)'
    ),
    "ai_company": (
        '("AI bubble" OR "AI capex" OR "AI stocks" OR Nvidia OR OpenAI OR '
        'Anthropic OR "data center" OR "AI spending" OR hyperscaler)'
    ),
}

# Appended to every query: English only, no retweets.
TWITTER_QUERY_SUFFIX = "lang:en -is:retweet"

# Per (query family, day) page cap. 20 tweets per page, so 40 pages is at
# most 800 tweets per family per day: 2 families x 23 days x 800 = 36,800
# tweets, about 552,000 credits, inside the 650,000 ceiling below.
TWITTER_MAX_PAGES = 40

# Base delay between pages, doubled on each 429 retry. Measured page latency
# is ~1.2s and four back-to-back pages with no delay returned no 429s, so the
# original 2.0s was pure overhead -- it tripled harvest wall-time for nothing.
# A small courtesy gap remains, with _search_page's exponential backoff as the
# real protection against throttling.
TWITTER_PAGE_DELAY_SEC = 0.35

# Day-queries are fetched concurrently; each owns an independent cursor chain.
# Measured: 8 simultaneous requests all returned 200 and completed in the
# wall-time of a single sequential pair, so the API tolerates this comfortably.
TWITTER_MAX_WORKERS = 8

# --- Credit budget -------------------------------------------------------
# twitterapi.io bills in credits: 1,000,000 credits = $10, and tweets cost
# $0.15 per 1,000, so one tweet is 15 credits. Confirmed empirically against
# the dashboard: 35 calls returning 20 tweets each billed 10,500 credits,
# i.e. 300 credits per page.
CREDITS_PER_TWEET = 15

# Hard ceiling on what a harvest may spend, enforced in fetch_tweets. The
# account holds 1,000,000 credits. This ceiling leaves ~350,000 credits
# (about 23,000 tweets) in reserve after a full harvest, comfortably above
# the 100,000 the account holder asked to keep back for contingencies. It is
# a guard rail, not a target: the run stops cleanly and keeps its shards.
TWITTER_CREDIT_BUDGET = 650_000
TWITTER_MAX_TWEETS = TWITTER_CREDIT_BUDGET // CREDITS_PER_TWEET


# ---------------------------------------------------------------------------
# News channel  (Google News RSS)
# ---------------------------------------------------------------------------
# GDELT DOC 2.0 was the first choice -- it tags source country natively -- but
# it returned HTTP 429 continuously during this study, from two independent
# networks, so it is globally throttled rather than locally blocked. Google
# News RSS is the substitute: no key, no observed rate limit, honours
# ``after:``/``before:`` date operators, and critically exposes the publisher
# domain in each item's <source url="..."> attribute, which is what the outlet
# taxonomy below keys on.
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
NEWS_MIN_INTERVAL_SEC = 1.5  # courtesy pacing, not an enforced limit

NEWS_QUERIES: dict[str, str] = {
    "ai_risk": (
        '"AI safety" OR "AI risk" OR superintelligence OR "AI alignment" OR '
        '"existential risk" OR "AI extinction"'
    ),
    "ai_company": (
        '"artificial intelligence" AND (Nvidia OR OpenAI OR Anthropic OR '
        'Microsoft OR Google OR Meta OR Amazon)'
    ),
}


# ---------------------------------------------------------------------------
# Outlet taxonomy for the Sacerdote, Sehgal & Cook (2020) replication
# ---------------------------------------------------------------------------
# Three buckets, mirroring the COVID-19 study: major US outlets, non-US
# English-language outlets, and the scientific / technical trade press. Any
# domain not matched here is bucketed as "Other US" or "Other Non-US" using
# GDELT's own source-country tag, so nothing is silently dropped.
US_MAINSTREAM_DOMAINS = {
    "nytimes.com", "washingtonpost.com", "cnn.com", "nbcnews.com",
    "cbsnews.com", "abcnews.go.com", "usatoday.com", "foxnews.com",
    "msnbc.com", "politico.com", "thehill.com", "newsweek.com",
    "latimes.com", "nypost.com", "huffpost.com", "axios.com",
    "bostonglobe.com", "chicagotribune.com", "time.com", "forbes.com",
    "businessinsider.com", "cnbc.com", "wsj.com", "bloomberg.com",
    "marketwatch.com", "fortune.com", "barrons.com",
    # Added after auditing the unclassified tail: these are unambiguously
    # major US outlets and together account for a large share of what the
    # first pass dropped into "Other".
    "npr.org", "pbs.org", "apnews.com", "abcnews.com", "foxbusiness.com",
    "nbcbayarea.com", "aol.com", "yahoo.com", "finance.yahoo.com",
    "thedailybeast.com", "vox.com", "slate.com", "theatlantic.com",
    "newyorker.com", "latimes.com", "sfgate.com", "mercurynews.com",
    "seattletimes.com", "dallasnews.com", "miamiherald.com",
    "usnews.com", "thestreet.com", "fool.com", "seekingalpha.com",
    "benzinga.com", "investors.com", "pymnts.com", "investing.com",
    "tradingview.com", "simplywall.st", "zacks.com", "kiplinger.com",
    "nasdaq.com", "news.bloomberglaw.com", "aws.amazon.com",
}

NON_US_DOMAINS = {
    "bbc.com", "bbc.co.uk", "theguardian.com", "reuters.com", "ft.com",
    "aljazeera.com", "dw.com", "france24.com", "scmp.com", "straitstimes.com",
    "thehindu.com", "timesofindia.indiatimes.com", "economictimes.indiatimes.com",
    "japantimes.co.jp", "abc.net.au", "smh.com.au", "cbc.ca", "globeandmail.com",
    "independent.co.uk", "telegraph.co.uk", "thetimes.co.uk", "euronews.com",
    "rte.ie", "irishtimes.com", "channelnewsasia.com", "nikkei.com",
    # Added after auditing the unclassified tail.
    "business-standard.com", "ndtv.com", "firstpost.com", "livemint.com",
    "indianexpress.com", "moneycontrol.com", "hindustantimes.com",
    "mk.co.kr", "koreaherald.com", "aa.com.tr", "jpost.com", "haaretz.com",
    "afr.com", "news.com.au", "theage.com.au", "lemonde.fr", "lefigaro.fr",
    "spiegel.de", "handelsblatt.com", "elpais.com", "corriere.it",
    "thenews.com.pk", "dawn.com", "mezha.net", "kyivpost.com",
    "globaltimes.cn", "chinadaily.com.cn", "rt.com", "sputniknews.com",
    "thestar.com.my", "bangkokpost.com", "jakartapost.com", "qz.com",
}

SCIENTIFIC_TECH_DOMAINS = {
    "nature.com", "science.org", "sciencedaily.com", "newscientist.com",
    "scientificamerican.com", "arstechnica.com", "techcrunch.com",
    "theverge.com", "wired.com", "venturebeat.com", "zdnet.com",
    "ieee.org", "spectrum.ieee.org", "technologyreview.com", "arxiv.org",
    "theregister.com", "anandtech.com", "tomshardware.com", "engadget.com",
    "phys.org", "quantamagazine.org", "semianalysis.com",
    # Added after auditing the unclassified tail.
    "futurism.com", "quantumzeitgeist.com", "techtarget.com", "gizmodo.com",
    "thenextweb.com", "readwrite.com", "analyticsindiamag.com",
    "infoworld.com", "computerworld.com", "theinformation.com",
    "hackernoon.com", "towardsdatascience.com", "marktechpost.com",
    "unite.ai", "aibusiness.com", "siliconangle.com", "datacenterdynamics.com",
}

OUTLET_BUCKETS = {
    "US Mainstream": US_MAINSTREAM_DOMAINS,
    "Non-US": NON_US_DOMAINS,
    "Scientific/Tech": SCIENTIFIC_TECH_DOMAINS,
}

# Country-code TLDs, used as a fallback for the long tail. A curated list can
# never cover 850+ domains by hand, and leaving 70% of the corpus in "Other"
# would mean the bucket comparison rested on a minority of the data. A ccTLD
# is strong, auditable evidence of a non-US publisher.
#
# Deliberately excluded: .io, .ai, .co, .me, .tv, .cc, .ly and .to, which are
# sold as generic vanity domains and carry no reliable geographic signal --
# a US startup on a .ai domain is the norm, not the exception.
COUNTRY_CODE_TLDS = {
    "ac", "ad", "ae", "af", "ag", "al", "am", "ar", "at", "au", "az", "ba",
    "bd", "be", "bg", "bh", "br", "bs", "bw", "by", "ca", "ch", "cl", "cn",
    "cr", "cu", "cy", "cz", "de", "dk", "do", "dz", "ec", "ee", "eg", "es",
    "fi", "fj", "fr", "ge", "gh", "gr", "gt", "hk", "hn", "hr", "hu", "id",
    "ie", "il", "in", "iq", "ir", "is", "it", "jm", "jo", "jp", "ke", "kh",
    "kr", "kw", "kz", "lb", "lk", "lt", "lu", "lv", "ma", "mk", "mm", "mn",
    "mt", "mu", "mv", "mx", "my", "ng", "nl", "no", "np", "nz", "om", "pe",
    "ph", "pk", "pl", "pr", "ps", "pt", "py", "qa", "ro", "rs", "ru", "sa",
    "se", "sg", "si", "sk", "sn", "sv", "sy", "th", "tn", "tr", "tw", "tz",
    "ua", "ug", "uk", "uy", "uz", "ve", "vn", "za", "zm", "zw",
}


# ---------------------------------------------------------------------------
# Relevance scoring  (retail-sentiment study, 0-100)
# ---------------------------------------------------------------------------
# Stories scoring at or above this threshold drive the event sentiment score;
# everything below is down-weighted rather than discarded.
RELEVANCE_FULL_WEIGHT = 90
RELEVANCE_MIN = 25  # below this a story is dropped as off-topic


# ---------------------------------------------------------------------------
# Econometrics
# ---------------------------------------------------------------------------
MAX_LAG = 3          # the window is short; more than 3 lags is not estimable
SIGNIFICANCE = 0.05
