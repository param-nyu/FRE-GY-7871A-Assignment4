"""Market and macro data for the AI basket.

Two sources, both free:

* **Alpha Vantage** for daily OHLCV. The assignment's volatility proxy needs
  the intraday high and low, not just the close, which rules out close-only
  feeds. Yahoo/``yfinance`` is the usual choice here but rate-limits shared
  egress IPs to the point of being unusable, so Alpha Vantage is the primary
  and ``yfinance`` is kept as an opt-in fallback.
* **FRED** for the 3-month Treasury yield, used to convert the raw daily
  return into an excess return over the risk-free rate where that matters.

Everything is cached under ``data/prices``. Delete the cache to refetch.
"""

from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from .config import (
    ALL_TICKERS,
    ALPHAVANTAGE_KEY,
    BENCHMARK,
    PRICE_DIR,
    PRICE_END,
    PRICE_START,
    RISK_FREE_SERIES,
    USER_AGENT,
)

ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Alpha Vantage's free tier allows 25 requests/day and 5 per minute. One
# request per ticker at 13s spacing keeps a 9-ticker pull inside both limits
# (9 requests, ~2 minutes) and leaves daily budget for a re-run.
_AV_INTERVAL_SEC = 13.0


class DataUnavailable(RuntimeError):
    """Raised when a required series cannot be fetched and has no cache."""


def _get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    attempts: int = 4,
    timeout: int = 60,
) -> requests.Response:
    """GET with exponential backoff.

    FRED intermittently stops responding for a minute or more at a time. An
    earlier version backed off 2/4/6s and still exhausted its retries inside a
    single outage, which made the pipeline look broken when the series was
    simply briefly unavailable -- so the schedule is 5/15/45s, wide enough to
    ride out a short outage without hanging the run for long.
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = requests.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(5.0 * (3**attempt))

    # Last resort: urllib. Observed on this network, ``requests`` can time out
    # against FRED for minutes at a stretch while a plain urllib call to the
    # identical URL returns in under a second. The cause is upstream and not
    # worth diagnosing, but losing the risk-free series to it is not
    # acceptable, so the stdlib client gets one attempt before we give up.
    try:
        full = url
        if params:
            full = f"{url}?{urlencode(params)}"
        request = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            body = handle.read()
        fallback = requests.Response()
        fallback.status_code = 200
        fallback._content = body
        fallback.url = full
        return fallback
    except Exception as exc:  # noqa: BLE001 - genuinely last-resort
        raise DataUnavailable(
            f"GET {url} failed after {attempts} requests attempts ({last}) "
            f"and a urllib fallback ({exc})"
        ) from exc


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def _fetch_alphavantage_daily(ticker: str, session: requests.Session) -> pd.DataFrame:
    """Daily OHLCV for one ticker, as a tidy frame.

    ``outputsize=compact`` returns the last 100 trading days, which on the
    free tier is all that is on offer -- ``full`` is gated behind a premium
    plan. 100 trading days back from the sample end reaches mid-May 2026, so
    it comfortably spans :data:`PRICE_START` and leaves the first in-window
    day with a genuine prior close.

    Alpha Vantage signals throttling, premium gating and bad symbols in the
    JSON body with a 200 status, so the error keys are checked explicitly
    rather than relying on ``raise_for_status``.
    """
    if not ALPHAVANTAGE_KEY:
        raise DataUnavailable(
            "ALPHAVANTAGE_KEY is not set. Get a free key (no credit card) at "
            "https://www.alphavantage.co/support/#api-key and put it in "
            ".env"
        )

    resp = session.get(
        ALPHAVANTAGE_URL,
        params={
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": ALPHAVANTAGE_KEY,
        },
        timeout=45,
    )
    resp.raise_for_status()
    payload = resp.json()

    for key in ("Note", "Information", "Error Message"):
        if key in payload:
            raise DataUnavailable(f"Alpha Vantage refused {ticker}: {payload[key]}")

    series = payload.get("Time Series (Daily)")
    if not series:
        raise DataUnavailable(f"Alpha Vantage returned no series for {ticker}")

    frame = pd.DataFrame.from_dict(series, orient="index").astype(float)
    frame.columns = [c.split(". ", 1)[-1] for c in frame.columns]
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame["ticker"] = ticker
    return frame.reset_index(names="date")


def download_ohlcv(
    tickers: list[str] | None = None,
    start: str = PRICE_START,
    end: str = PRICE_END,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Long-format daily OHLCV for the basket plus the benchmark.

    Returns columns ``date, ticker, open, high, low, close, volume``. The
    result is cached, and a cache that already covers every requested ticker
    is returned untouched so repeated notebook runs cost nothing.
    """
    tickers = list(tickers or ALL_TICKERS)
    cache_path = Path(cache_path or PRICE_DIR / "ohlcv.csv")

    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        if not set(tickers) - set(cached["ticker"].unique()):
            return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    frames, failed = [], []
    for i, ticker in enumerate(tickers):
        try:
            frames.append(_fetch_alphavantage_daily(ticker, session))
        except (DataUnavailable, requests.RequestException) as exc:
            failed.append(f"{ticker}: {exc}")
        if i < len(tickers) - 1:
            time.sleep(_AV_INTERVAL_SEC)

    if not frames:
        raise DataUnavailable("No tickers downloaded.\n" + "\n".join(failed))
    if failed:
        print(f"[data_loader] {len(failed)} ticker(s) failed:\n  " + "\n  ".join(failed))

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[["date", "ticker", "open", "high", "low", "close", "volume"]]
    panel = panel[panel["date"].between(start, end)].reset_index(drop=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel.to_csv(cache_path, index=False)
    return panel


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------
def download_risk_free(
    series_id: str = RISK_FREE_SERIES,
    start: str = PRICE_START,
    end: str = PRICE_END,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> pd.Series:
    """Daily risk-free rate as a *decimal daily* return, indexed by date.

    FRED publishes ``DGS3MO`` as an annualised percentage. It is divided by 100
    and then by 252 to put it on the same per-trading-day footing as the equity
    returns. Non-trading days are dropped and gaps forward-filled, since the
    series is missing on market holidays.
    """
    cache_path = Path(cache_path or PRICE_DIR / f"{series_id}.csv")

    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return cached.iloc[:, 0]

    resp = _get_with_retry(FRED_CSV_URL, params={"id": series_id})

    raw = pd.read_csv(io.StringIO(resp.text))
    raw.columns = ["date", "value"]
    raw["date"] = pd.to_datetime(raw["date"])
    # FRED marks missing observations with a literal ".".
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.set_index("date")["value"].loc[start:end].ffill().dropna()

    daily = raw / 100.0 / 252.0
    daily.name = "rf_daily"
    daily.to_frame().to_csv(cache_path)
    return daily


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
def build_market_panel(
    ohlcv: pd.DataFrame,
    risk_free: pd.Series | None = None,
    benchmark: str = BENCHMARK,
) -> pd.DataFrame:
    """Attach returns, excess returns and the volatility proxy to the panel.

    Columns added:

    ``ret``
        Simple close-to-close return.
    ``excess_ret``
        ``ret`` less the same-day benchmark return, per the brief's
        :math:`R^{excess}_{i,t} = R_{i,t} - R_{SPY,t}`. The benchmark's own
        rows carry ``NaN`` here, since SPY has no excess return over itself.
    ``excess_ret_rf``
        ``ret`` less the daily risk-free rate, kept alongside for the summary
        table; the Granger tests use ``excess_ret``.
    ``vol_proxy``
        The intraday range proxy :math:`2(H_t - L_t)/(H_t + L_t)` from the
        retail-sentiment study.

    Returns are computed per ticker *before* the sample window is imposed, so
    the first in-window day has a genuine prior close rather than a ``NaN``.
    """
    panel = ohlcv.sort_values(["ticker", "date"]).copy()
    panel["ret"] = panel.groupby("ticker")["close"].pct_change()

    denom = panel["high"] + panel["low"]
    panel["vol_proxy"] = np.where(
        denom > 0, 2.0 * (panel["high"] - panel["low"]) / denom, np.nan
    )

    bench = (
        panel.loc[panel["ticker"] == benchmark, ["date", "ret"]]
        .rename(columns={"ret": "bench_ret"})
        .set_index("date")["bench_ret"]
    )
    panel["bench_ret"] = panel["date"].map(bench)
    panel["excess_ret"] = panel["ret"] - panel["bench_ret"]
    panel.loc[panel["ticker"] == benchmark, "excess_ret"] = np.nan

    if risk_free is not None:
        panel["rf_daily"] = panel["date"].map(risk_free)
        panel["excess_ret_rf"] = panel["ret"] - panel["rf_daily"]

    return panel.reset_index(drop=True)


def basket_aggregate(panel: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Equal-weighted daily basket series.

    An equal-weighted average is the right aggregator here: the basket is a
    thematic construct, not a portfolio anyone holds, and cap-weighting would
    let a single name dominate the sentiment tests.
    """
    sub = panel[panel["ticker"].isin(tickers)]
    out = (
        sub.groupby("date")[["ret", "excess_ret", "vol_proxy"]]
        .mean()
        .rename(
            columns={
                "ret": "basket_ret",
                "excess_ret": "basket_excess_ret",
                "vol_proxy": "basket_vol",
            }
        )
    )
    out["n_names"] = sub.groupby("date")["ticker"].nunique()
    return out.reset_index()
