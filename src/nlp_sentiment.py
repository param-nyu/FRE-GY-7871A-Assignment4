"""Two-channel sentiment pipeline: X/Twitter and news.

The brief asks for social media *and* news, and the lecture's reasoning for
both is worth restating because it drives the design: news aims at objectivity
while Twitter is "pure emotion", so the two channels carry different
information and are kept as separate series throughout rather than blended
into one index.

Channel 1 -- X via `twitterapi.io <https://docs.twitterapi.io>`_
    ``GET /twitter/tweet/advanced_search``, ``x-api-key`` header, cursor
    pagination, 20 tweets per page. Note that the date operators are
    ``since_time``/``until_time`` as **Unix timestamps**; the documentation
    states plainly that the ``since:``/``until:`` date form is not supported,
    and it fails silently rather than erroring.

Channel 2 -- Google News RSS
    One request per (query family, day), publisher domain read from each
    item's ``<source url>`` attribute.

Both channels are fetched day by day rather than as one window-wide query.
That costs more requests but it is the only way to get an unbiased daily
series: a single window-wide call returns the API's idea of the top N items
and would silently over-weight whichever days had the loudest coverage.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from .config import (
    COUNTRY_CODE_TLDS,
    CREDITS_PER_TWEET,
    EXCLUDE_REPLIES,
    GOOGLE_NEWS_RSS,
    MIN_FOLLOWERS,
    MIN_VIEWS,
    NEWS_DIR,
    NEWS_MIN_INTERVAL_SEC,
    NEWS_QUERIES,
    OUTLET_BUCKETS,
    RELEVANCE_FULL_WEIGHT,
    RELEVANCE_MIN,
    SAMPLE_END,
    SAMPLE_START,
    SOCIAL_DIR,
    TWITTER_CREDIT_BUDGET,
    TWITTER_MAX_PAGES,
    TWITTER_MAX_TWEETS,
    TWITTER_MAX_WORKERS,
    TWITTER_PAGE_DELAY_SEC,
    TWITTER_QUERIES,
    TWITTER_QUERY_SUFFIX,
    TWITTER_SEARCH_URL,
    TWITTERAPI_IO_KEY,
    USER_AGENT,
)
from .sentiment_model import (
    LoughranMcDonaldScorer,
    VaderScorer,
    classify,
    sentiment_ratio,
)

UTC = dt.timezone.utc


def _day_range(start: str = SAMPLE_START, end: str = SAMPLE_END) -> list[dt.date]:
    """Every calendar day in the sample window, inclusive of both ends."""
    first = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    return [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]


def _unix(day: dt.date, end_of_day: bool = False) -> int:
    """Midnight UTC on ``day`` as a Unix timestamp."""
    moment = dt.datetime.combine(
        day + dt.timedelta(days=1 if end_of_day else 0), dt.time.min, tzinfo=UTC
    )
    return int(moment.timestamp())


# ---------------------------------------------------------------------------
# User geography
# ---------------------------------------------------------------------------
# X exposes only a free-text, self-reported profile location -- "Houston",
# "SLO, CA, USA", "L3", or blank. It cannot be geocoded reliably, so this is a
# deliberately conservative gazetteer match: a user is only labelled when the
# string is unambiguous, and everything else becomes "Unknown" rather than
# being forced into a bucket. Roughly a third of accounts leave it empty, so
# the US/non-US split is a sub-sample result and is reported as such.
_US_MARKERS = {
    "usa", "u.s.", "u.s.a", "united states", "america",
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming",
    "nyc", "sf", "bay area", "silicon valley", "san francisco", "los angeles",
    "chicago", "boston", "seattle", "austin", "denver", "atlanta", "houston",
    "dallas", "miami", "philadelphia", "phoenix", "san diego", "san jose",
    "brooklyn", "manhattan", "palo alto", "berkeley", "cambridge, ma", "d.c.",
    "dc", "washington dc",
}
# Two-letter US state codes, matched only as a standalone token.
_US_STATE_CODES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy",
}
_NON_US_MARKERS = {
    "united kingdom", "uk", "england", "scotland", "wales", "ireland",
    "london", "manchester", "edinburgh", "dublin", "cambridge, uk", "oxford",
    "canada", "toronto", "vancouver", "montreal", "ottawa",
    "australia", "sydney", "melbourne", "brisbane", "new zealand",
    "india", "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad",
    "china", "beijing", "shanghai", "hong kong", "shenzhen", "taiwan",
    "japan", "tokyo", "osaka", "korea", "seoul", "singapore",
    "germany", "berlin", "munich", "france", "paris", "spain", "madrid",
    "barcelona", "italy", "rome", "milan", "netherlands", "amsterdam",
    "sweden", "stockholm", "norway", "oslo", "denmark", "copenhagen",
    "switzerland", "zurich", "geneva", "austria", "vienna", "belgium",
    "brussels", "poland", "warsaw", "portugal", "lisbon", "finland",
    "brazil", "sao paulo", "mexico", "argentina", "chile", "colombia",
    "nigeria", "lagos", "kenya", "south africa", "egypt", "israel",
    "tel aviv", "dubai", "uae", "saudi", "turkey", "istanbul", "russia",
    "moscow", "ukraine", "pakistan", "bangladesh", "indonesia", "jakarta",
    "philippines", "manila", "vietnam", "thailand", "bangkok", "malaysia",
    "europe", "eu", "worldwide", "global",
}


def classify_region(location: str) -> str:
    """Label a free-text profile location as ``US``, ``Non-US`` or ``Unknown``.

    Non-US markers are checked first: strings like "Cambridge, UK" and
    "London, Ontario" contain tokens that would otherwise trip the US
    gazetteer, and a false "US" is more damaging to the comparison than an
    honest "Unknown".
    """
    text = (location or "").strip().lower()
    if not text:
        return "Unknown"

    for marker in _NON_US_MARKERS:
        if marker in text:
            return "Non-US"
    for marker in _US_MARKERS:
        if marker in text:
            return "US"

    tokens = {t.strip(" .,/|-") for t in text.replace(",", " ").split()}
    if tokens & _US_STATE_CODES:
        return "US"
    return "Unknown"


# ---------------------------------------------------------------------------
# Channel 1: X / Twitter
# ---------------------------------------------------------------------------
def _tweet_passes_filters(tweet: dict) -> bool:
    """Apply the brief's three quality filters to one tweet.

    Retweets are excluded in the query string, but the API still returns items
    carrying a ``retweeted_tweet`` payload, so the flag is re-checked here.
    The follower and view floors cannot be expressed as search operators and
    must be applied client-side.
    """
    if tweet.get("retweeted_tweet"):
        return False
    if EXCLUDE_REPLIES and tweet.get("isReply"):
        return False
    if (tweet.get("viewCount") or 0) < MIN_VIEWS:
        return False
    author = tweet.get("author") or {}
    if (author.get("followers") or 0) < MIN_FOLLOWERS:
        return False
    return bool((tweet.get("text") or "").strip())


def _parse_tweet(tweet: dict, family: str, day: dt.date) -> dict:
    author = tweet.get("author") or {}
    return {
        "date": day.isoformat(),
        "family": family,
        "id": tweet.get("id"),
        "created_at": tweet.get("createdAt"),
        "text": tweet.get("text") or "",
        "lang": tweet.get("lang"),
        "view_count": tweet.get("viewCount") or 0,
        "like_count": tweet.get("likeCount") or 0,
        "retweet_count": tweet.get("retweetCount") or 0,
        "reply_count": tweet.get("replyCount") or 0,
        "is_reply": bool(tweet.get("isReply")),
        "author": author.get("userName"),
        "author_followers": author.get("followers") or 0,
        "author_verified": bool(author.get("isBlueVerified")),
        "author_location": (author.get("location") or "").strip(),
        "region": classify_region(author.get("location") or ""),
    }


class CreditExhausted(RuntimeError):
    """twitterapi.io returned 402: the account balance is spent.

    Distinguished from every other failure because the remedy is different --
    no amount of retrying fixes it, and the partial harvest collected up to
    that point is still valid data worth keeping.
    """


def _search_page(
    session: requests.Session,
    query: str,
    cursor: str,
    attempts: int = 5,
) -> dict:
    """One page of advanced-search results, with backoff on 429.

    twitterapi.io throttles pagination well before the documented per-second
    ceiling -- the second page of a day's results reliably 429s if requested
    immediately after the first -- so a plain loop over cursors fails. Backoff
    is exponential and the final failure is raised, never swallowed, so a
    truncated harvest cannot masquerade as a quiet day.
    """
    delay = TWITTER_PAGE_DELAY_SEC
    for attempt in range(attempts):
        resp = session.get(
            TWITTER_SEARCH_URL,
            headers={"x-api-key": TWITTERAPI_IO_KEY},
            params={"query": query, "queryType": "Latest", "cursor": cursor},
            timeout=45,
        )
        if resp.status_code == 402:
            raise CreditExhausted(
                "twitterapi.io returned 402 Payment Required -- the account "
                "balance is exhausted. Top up at https://twitterapi.io "
                "(usage is $0.15 per 1,000 tweets). Days already harvested "
                "are cached and will not be re-fetched or re-charged."
            )
        if resp.status_code == 429:
            if attempt == attempts - 1:
                resp.raise_for_status()
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


def fetch_tweets_for_day(
    family: str,
    day: dt.date,
    session: requests.Session,
    max_pages: int = TWITTER_MAX_PAGES,
) -> tuple[list[dict], int]:
    """All qualifying tweets for one query family on one day.

    Returns ``(kept, seen)`` so the caller can report what the quality filters
    removed -- a number the report has to disclose.
    """
    query = (
        f"{TWITTER_QUERIES[family]} {TWITTER_QUERY_SUFFIX} "
        f"since_time:{_unix(day)} until_time:{_unix(day, end_of_day=True)}"
    )

    kept: list[dict] = []
    seen = 0
    cursor = ""
    for page in range(max_pages):
        if page:
            time.sleep(TWITTER_PAGE_DELAY_SEC)
        payload = _search_page(session, query, cursor)

        tweets = payload.get("tweets") or []
        seen += len(tweets)
        kept.extend(
            _parse_tweet(t, family, day) for t in tweets if _tweet_passes_filters(t)
        )

        if not payload.get("has_next_page") or not payload.get("next_cursor"):
            break
        cursor = payload["next_cursor"]

    return kept, seen


def fetch_tweets(
    start: str = SAMPLE_START,
    end: str = SAMPLE_END,
    cache_path: Path | None = None,
    refresh: bool = False,
    max_workers: int = TWITTER_MAX_WORKERS,
) -> pd.DataFrame:
    """Harvest the full window, both query families, with filter diagnostics.

    Day-queries are fetched **concurrently**. Each (family, day) owns an
    independent cursor chain and writes its own shard, so there is no shared
    mutable state between workers and the only coordination needed is the
    credit counter. Measured: eight concurrent requests complete in the time
    one sequential batch of two does, which takes the full harvest from
    roughly 45 minutes to under 10.

    Every (family, day) is checkpointed the moment it is fetched. Tweets cost
    money, so a crash two-thirds of the way through must never force a
    re-fetch of what was already paid for -- and must never silently lose it.
    """
    cache_path = Path(cache_path or SOCIAL_DIR / "tweets.csv")
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    if not TWITTERAPI_IO_KEY:
        raise RuntimeError(
            "TWITTERAPI_IO_KEY is not set. Add it to .env -- see "
            "https://twitterapi.io (usage-priced, $0.15 per 1,000 tweets)."
        )

    shard_dir = SOCIAL_DIR / "raw"
    shard_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        (family, day)
        for family in TWITTER_QUERIES
        for day in _day_range(start, end)
        if refresh or not (shard_dir / f"{family}_{day.isoformat()}.csv").exists()
    ]
    done = len(TWITTER_QUERIES) * len(_day_range(start, end)) - len(pending)
    print(
        f"[tweets] {done} shard(s) already cached; fetching {len(pending)} "
        f"with {max_workers} workers",
        flush=True,
    )

    state = {"billed": 0, "exhausted": None, "budget_stop": False}
    lock = threading.Lock()
    local = threading.local()

    def session_for_thread() -> requests.Session:
        if not hasattr(local, "session"):
            local.session = requests.Session()
            local.session.headers.update({"User-Agent": USER_AGENT})
        return local.session

    def worker(task: tuple[str, dt.date]) -> tuple[str, dt.date, int, int] | None:
        family, day = task
        with lock:
            if state["exhausted"] or state["budget_stop"]:
                return None
            if state["billed"] >= TWITTER_MAX_TWEETS:
                state["budget_stop"] = True
                return None
        try:
            kept, seen = fetch_tweets_for_day(family, day, session_for_thread())
        except CreditExhausted as exc:
            with lock:
                state["exhausted"] = exc
            return None
        with lock:
            state["billed"] += seen
            billed = state["billed"]
        pd.DataFrame(kept).to_csv(shard_dir / f"{family}_{day.isoformat()}.csv",
                                  index=False)
        return family, day, len(kept), seen, billed

    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(worker, task): task for task in pending}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                family, day, kept, seen, billed = result
                print(
                    f"[tweets] {family} {day} kept={kept:4d} seen={seen:4d} "
                    f"| billed={billed:,} (~{billed * CREDITS_PER_TWEET:,} cr)",
                    flush=True,
                )

    shards = sorted(shard_dir.glob("*.csv"))
    frames = [pd.read_csv(s) for s in shards if s.stat().st_size > 0]
    if not frames:
        if state["exhausted"]:
            raise state["exhausted"]
        raise RuntimeError("No tweets harvested and no cached shards found.")

    frame = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["id", "family"]
    )
    expected = len(TWITTER_QUERIES) * len(_day_range(start, end))
    print(
        f"[tweets] kept={len(frame):,} from {len(shards)}/{expected} "
        f"(family, day) shards"
    )
    print(
        f"[tweets] billed this run: {state['billed']:,} tweets "
        f"(~{state['billed'] * CREDITS_PER_TWEET:,} credits)"
    )
    if state["exhausted"]:
        print(f"[tweets] WARNING: harvest incomplete -- {state['exhausted']}",
              flush=True)
    if state["budget_stop"]:
        print(
            "[tweets] WARNING: harvest stopped on the local credit budget, "
            "not on account balance. Raise TWITTER_CREDIT_BUDGET to continue.",
            flush=True,
        )

    frame.to_csv(cache_path, index=False)
    return frame


# ---------------------------------------------------------------------------
# Channel 2: News
# ---------------------------------------------------------------------------
def _domain_of(url: str) -> str:
    """Bare registrable-ish domain, lowercased, ``www.`` stripped."""
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def bucket_outlet(domain: str) -> str:
    """Map a publisher domain to one of the three Sacerdote-style buckets.

    Resolution order, most to least reliable:

    1. The curated domain lists in :mod:`config`.
    2. A country-code TLD, which assigns ``Non-US``. This fallback exists
       because the curated lists alone left roughly 70% of a 2,977-article
       corpus in ``Other``, which would have meant the headline bias
       comparison rested on a minority of the data. Generic vanity ccTLDs
       (.io, .ai, .co and friends) are excluded from
       :data:`COUNTRY_CODE_TLDS` precisely because they carry no geographic
       signal.
    3. ``Other``, for everything unresolved -- overwhelmingly ``.com``
       domains too small or too obscure to place.

    Anything that lands in ``Other`` is kept in the corpus and reported, never
    silently dropped: a bucket that quietly swallowed the long tail would make
    the comparison meaningless.
    """
    domain = (domain or "").lower().strip()
    if not domain:
        return "Other"

    for bucket, domains in OUTLET_BUCKETS.items():
        if domain in domains or any(domain.endswith("." + d) for d in domains):
            return bucket

    tld = domain.rsplit(".", 1)[-1]
    if tld in COUNTRY_CODE_TLDS:
        return "Non-US"
    return "Other"


def fetch_news_for_day(
    family: str, day: dt.date, session: requests.Session
) -> list[dict]:
    """One day of headlines for one query family, via Google News RSS.

    Google's ``after``/``before`` operators are exclusive of the ``before``
    date, so the next day is passed to capture ``day`` itself.
    """
    query = (
        f"{NEWS_QUERIES[family]} "
        f"after:{day.isoformat()} before:{(day + dt.timedelta(days=1)).isoformat()}"
    )
    resp = session.get(
        GOOGLE_NEWS_RSS,
        params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        timeout=45,
    )
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    rows = []
    for item in root.findall(".//item"):
        source = item.find("source")
        publisher_url = source.attrib.get("url", "") if source is not None else ""
        domain = _domain_of(publisher_url)
        title = (item.findtext("title") or "").strip()
        # Google appends " - Publisher" to every headline; strip it so the
        # outlet name does not leak into the sentiment score.
        publisher = source.text if source is not None else ""
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(f" - {publisher}")]

        rows.append(
            {
                "date": day.isoformat(),
                "family": family,
                "title": title,
                "publisher": publisher,
                "domain": domain,
                "bucket": bucket_outlet(domain),
                "pub_date": item.findtext("pubDate"),
                "link": item.findtext("link"),
            }
        )
    return rows


def fetch_news(
    start: str = SAMPLE_START,
    end: str = SAMPLE_END,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Harvest headlines across the window for both query families."""
    cache_path = Path(cache_path or NEWS_DIR / "news.csv")
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rows: list[dict] = []
    for family in NEWS_QUERIES:
        for day in _day_range(start, end):
            try:
                day_rows = fetch_news_for_day(family, day, session)
            except (requests.RequestException, ET.ParseError) as exc:
                print(f"[news] {family} {day} FAILED: {exc}", flush=True)
                continue
            rows.extend(day_rows)
            print(f"[news] {family} {day} n={len(day_rows):3d}", flush=True)
            time.sleep(NEWS_MIN_INTERVAL_SEC)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["title", "domain", "family"])
    frame.to_csv(cache_path, index=False)
    return frame


# ---------------------------------------------------------------------------
# Relevance scoring  (0-100, per the retail-sentiment study)
# ---------------------------------------------------------------------------
# Terms that make a story *about* the AI basket or the existential-risk debate,
# as opposed to merely mentioning AI in passing.
ENTITY_TERMS = (
    "nvidia", "meta", "amazon", "google", "alphabet", "microsoft", "apple",
    "amd", "broadcom", "openai", "anthropic", "hyperscaler", "data center",
)
RISK_TERMS = (
    "ai safety", "ai risk", "existential", "superintelligence", "alignment",
    "extinction", "doom", "kill all humans", "catastrophic", "ai regulation",
)


def relevance_score(title: str, body: str = "") -> int:
    """How central the AI basket / existential-risk theme is, on 0-100.

    Mirrors the retail study's construct: a story whose *headline* turns on the
    subject scores at or above :data:`RELEVANCE_FULL_WEIGHT` and drives the
    event sentiment score, while a passing mention lower down is discounted
    rather than discarded.

    The weighting is deliberately simple and transparent -- headline presence
    dominates, breadth of distinct matched terms adds the rest -- because an
    opaque relevance model would be impossible to defend in a 6-page writeup.
    """
    title_l, body_l = (title or "").lower(), (body or "").lower()
    all_terms = ENTITY_TERMS + RISK_TERMS

    in_title = {t for t in all_terms if t in title_l}
    in_body = {t for t in all_terms if t in body_l} - in_title

    if not in_title and not in_body:
        return 0

    score = 0.0
    if in_title:
        score += 70.0                                   # headline presence
        score += min(len(in_title) - 1, 2) * 10.0       # breadth, capped
    if in_body:
        score += min(len(in_body), 3) * 10.0            # body-only mentions

    # A story matching both an entity and the risk theme is squarely on-topic.
    matched = in_title | in_body
    if matched & set(ENTITY_TERMS) and matched & set(RISK_TERMS):
        score += 10.0

    return int(min(score, 100.0))


def relevance_weight(score: int) -> float:
    """Convert a relevance score into an aggregation weight.

    Full weight at or above the threshold, linear taper below it, zero under
    :data:`RELEVANCE_MIN`.
    """
    if score >= RELEVANCE_FULL_WEIGHT:
        return 1.0
    if score < RELEVANCE_MIN:
        return 0.0
    return (score - RELEVANCE_MIN) / (RELEVANCE_FULL_WEIGHT - RELEVANCE_MIN)


# ---------------------------------------------------------------------------
# Scoring and daily aggregation
# ---------------------------------------------------------------------------
def score_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    """Attach VADER sentiment, label and relevance to each tweet."""
    scorer = VaderScorer()
    out = tweets.copy()
    out["sentiment"] = scorer.score_many(out["text"])
    out["label"] = out["sentiment"].map(classify)
    out["relevance"] = [relevance_score(t, t) for t in out["text"]]
    out["weight"] = out["relevance"].map(relevance_weight)
    return out


def score_news(news: pd.DataFrame, lm_dictionary: str | Path) -> pd.DataFrame:
    """Attach both VADER and Loughran-McDonald tone to each headline.

    Both are reported. VADER is the headline-level comparator against the
    social channel; LM is the finance-domain measure and is what the media-bias
    table uses, so that the negativity comparison rests on a lexicon built for
    this kind of text.
    """
    vader = VaderScorer()
    lm = LoughranMcDonaldScorer(lm_dictionary)

    out = news.copy()
    out["title"] = out["title"].fillna("")
    out["sentiment"] = vader.score_many(out["title"])
    out["label"] = out["sentiment"].map(classify)
    out["lm_tone"] = lm.score_many(out["title"])
    out["lm_label"] = out["lm_tone"].map(classify)
    out["relevance"] = [relevance_score(t) for t in out["title"]]
    out["weight"] = out["relevance"].map(relevance_weight)
    return out


def daily_sentiment(
    scored: pd.DataFrame,
    label_col: str = "label",
    prefix: str = "s",
    weighted: bool = True,
) -> pd.DataFrame:
    """Collapse a scored corpus to one row per day.

    Produces the brief's ratio :math:`(Pos - Neg)/Total` alongside the mean
    continuous score, the volume, and -- when ``weighted`` -- a
    relevance-weighted ratio that discounts marginal items.
    """
    frames = []
    for day, group in scored.groupby("date"):
        row = {
            "date": day,
            f"{prefix}_ratio": sentiment_ratio(group[label_col]),
            f"{prefix}_mean": group["sentiment"].mean(),
            f"{prefix}_n": len(group),
            f"{prefix}_pos": int((group[label_col] == "positive").sum()),
            f"{prefix}_neg": int((group[label_col] == "negative").sum()),
            f"{prefix}_neu": int((group[label_col] == "neutral").sum()),
        }
        if weighted and "weight" in group:
            w = group["weight"]
            total = w.sum()
            if total > 0:
                pos = w[group[label_col] == "positive"].sum()
                neg = w[group[label_col] == "negative"].sum()
                row[f"{prefix}_ratio_w"] = (pos - neg) / total
                row[f"{prefix}_mean_w"] = (group["sentiment"] * w).sum() / total
            else:
                row[f"{prefix}_ratio_w"] = float("nan")
                row[f"{prefix}_mean_w"] = float("nan")
        frames.append(row)

    out = pd.DataFrame(frames).sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    return out
