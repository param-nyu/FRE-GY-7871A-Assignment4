# Assignment 4 — Sentiment Around AI Risk and the AI Equity Complex

An event study around the **8 September 2026** post by Evan Hubinger
(Anthropic) stating a >10% personal probability that AI kills all humans
within a decade. Two sentiment channels (X and news) are tracked from
8–30 September 2026 and tested against an eight-name AI equity basket.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in the two keys
python scripts/01_get_market_data.py
python scripts/02_get_sentiment_data.py
python scripts/03_run_analysis.py
python build_report.py
```

The report lands at `outputs/Assignment4_Report.pdf`.

## Credentials

Both go in `.env`, which is git-ignored. Never commit it.

| Variable | Source | Cost |
|---|---|---|
| `TWITTERAPI_IO_KEY` | [twitterapi.io](https://twitterapi.io) | $0.15 / 1,000 tweets (~$5 for a full harvest) |
| `ALPHAVANTAGE_KEY` | [alphavantage.co](https://www.alphavantage.co/support/#api-key) | Free, no card |

### Credit budgeting

twitterapi.io bills 15 credits per tweet (1,000,000 credits = $10). The
harvest is capped in `config.TWITTER_CREDIT_BUDGET`; it stops cleanly and
keeps its shards rather than draining the account. Every `(family, day)` is
checkpointed to `data/social/raw/` the moment it is fetched, so an interrupted
run **never re-fetches or re-pays for** what it already has.

## Layout

```
.
├── src/
│   ├── config.py            sample definition, basket, filters, outlet taxonomy
│   ├── data_loader.py       Alpha Vantage OHLCV + FRED; excess returns, vol proxy
│   ├── sentiment_model.py   VADER (+ finance patch) and Loughran-McDonald scorers
│   ├── nlp_sentiment.py     X and news harvesting, filters, relevance, daily ratios
│   ├── granger_causality.py VAR, lag selection, panel and single-series tests
│   ├── media_bias.py        Sacerdote-style negativity comparisons
│   ├── table_generator.py   Tables 1–3 in Markdown, LaTeX and CSV
│   ├── exhibits.py          Figures 1–4
│   ├── analysis.py          end-to-end orchestration
│   └── report_text.py       data-derived report narrative
├── scripts/                 01 market, 02 sentiment, 03 analysis
├── tests/                   pytest suite
├── build_report.py          HTML + PDF
├── data/lexicons/           Loughran-McDonald dictionary (vendored)
└── outputs/                 tables, figures, report
```

## Design decisions worth knowing

**Why the basket splits semis from hyperscalers.** NVDA/AMD/AVGO *sell* AI
compute; MSFT/GOOGL/AMZN/META/AAPL *buy* it. An AI-risk shock that threatens
regulation should hit a seller's order book differently from a buyer's capex
plan, and pooling them averages that away.

**Why VADER is patched.** Unpatched, *"Nvidia earnings crushed it, bullish"*
scores **−0.42** — "crushed" is violent in ordinary English and "bullish" is
absent from the lexicon. Patched, it scores **+0.78**. This is a regression
test in `tests/test_sentiment_model.py`. Only unambiguous terms are patched;
`short` and `long` are deliberately left alone.

**Why the panel test is primary.** The mandated window is 17 trading days. A
per-name VAR spends most of its degrees of freedom on the intercept and
own-lags. Pooling the eight names gives ~130 firm-days. Standard errors are
clustered **by date**, because the sentiment series is common across firms —
firm-clustered errors would be far too small and would manufacture
significance.

**Why weekend sentiment rolls forward.** Sentiment exists every calendar day;
markets do not. Saturday sentiment cannot be traded until Monday, so it is
volume-weighted into the next trading day. Dropping it would discard ~3 in 10
observations of the driver.

**Why two query families.** `ai_risk` (doom, alignment, p(doom)) and
`ai_company` (Nvidia, capex, AI bubble) are kept as separate series. They can
move in opposite directions — "AI might kill us all *and* I'm long NVDA" — and
a blended index would net that out.

## Known substitutions

- **GDELT → Google News RSS.** GDELT DOC 2.0 was the intended news source
  (it tags source country natively) but returned HTTP 429 continuously from
  two independent networks, indicating a global throttle. Outlet nationality
  is therefore assigned from the curated domain lists in `config.py`.
- **yfinance → Alpha Vantage.** Yahoo hard-rate-limits shared egress IPs.
  Alpha Vantage's free tier caps at `outputsize=compact` (100 trading days),
  which still spans the window with room for prior closes.

## Tests

```bash
python -m pytest tests -q
```

The Granger tests are the ones worth reading: they run on constructed series
with a known answer — one where sentiment genuinely leads returns and one
where it cannot — plus a direction test, since a transposed implementation
would silently report the reverse hypothesis.
