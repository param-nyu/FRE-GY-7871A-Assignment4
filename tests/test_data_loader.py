"""Tests for the derived market quantities.

These exercise the maths, not the network: the fetchers are covered by the
cache files they write, while the transformations below are where a silent
sign error would do real damage to the results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import basket_aggregate, build_market_panel


@pytest.fixture
def toy_ohlcv() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-09-08", "2026-09-09", "2026-09-10"])
    rows = []
    for ticker, closes in {
        "NVDA": [100.0, 110.0, 99.0],
        "AMD": [50.0, 52.0, 52.0],
        "SPY": [400.0, 404.0, 400.0],
    }.items():
        for date, close in zip(dates, closes):
            rows.append({
                "date": date, "ticker": ticker,
                "open": close, "high": close * 1.02, "low": close * 0.98,
                "close": close, "volume": 1_000.0,
            })
    return pd.DataFrame(rows)


class TestBuildMarketPanel:
    def test_returns_are_close_to_close(self, toy_ohlcv):
        panel = build_market_panel(toy_ohlcv)
        nvda = panel[panel["ticker"] == "NVDA"].sort_values("date")
        assert np.isnan(nvda["ret"].iloc[0])        # no prior close
        assert nvda["ret"].iloc[1] == pytest.approx(0.10)

    def test_excess_return_subtracts_same_day_benchmark(self, toy_ohlcv):
        panel = build_market_panel(toy_ohlcv)
        nvda = panel[panel["ticker"] == "NVDA"].sort_values("date")
        # NVDA +10%, SPY +1% on the same day.
        assert nvda["excess_ret"].iloc[1] == pytest.approx(0.10 - 0.01)

    def test_benchmark_has_no_excess_return_over_itself(self, toy_ohlcv):
        panel = build_market_panel(toy_ohlcv)
        spy = panel[panel["ticker"] == "SPY"]
        assert spy["excess_ret"].isna().all()

    def test_volatility_proxy_matches_formula(self, toy_ohlcv):
        """vol = 2(H-L)/(H+L); with H=1.02c and L=0.98c this is exactly 0.04.

        Compared elementwise rather than on ``.unique()``: floating-point
        division leaves the per-row values differing in the last bits, so
        ``unique()`` returns five not-quite-equal 0.04s and a length
        comparison against a single expected value fails for reasons that have
        nothing to do with the formula.
        """
        panel = build_market_panel(toy_ohlcv)
        values = panel["vol_proxy"].dropna()
        assert len(values) == len(panel)
        assert values.to_numpy() == pytest.approx(0.04)

    def test_volatility_proxy_is_non_negative(self, toy_ohlcv):
        panel = build_market_panel(toy_ohlcv)
        assert (panel["vol_proxy"].dropna() >= 0).all()

    def test_returns_computed_per_ticker_not_across(self, toy_ohlcv):
        """A naive global pct_change would leak AMD's close into NVDA's return."""
        panel = build_market_panel(toy_ohlcv)
        first_rows = panel.sort_values("date").groupby("ticker").head(1)
        assert first_rows["ret"].isna().all()

    def test_risk_free_excess_return(self, toy_ohlcv):
        rf = pd.Series(
            0.0001, index=pd.to_datetime(["2026-09-08", "2026-09-09", "2026-09-10"])
        )
        panel = build_market_panel(toy_ohlcv, rf)
        nvda = panel[panel["ticker"] == "NVDA"].sort_values("date")
        assert nvda["excess_ret_rf"].iloc[1] == pytest.approx(0.10 - 0.0001)


class TestBasketAggregate:
    def test_equal_weighted_mean(self, toy_ohlcv):
        panel = build_market_panel(toy_ohlcv)
        basket = basket_aggregate(panel, ["NVDA", "AMD"])
        day2 = basket[basket["date"] == pd.Timestamp("2026-09-09")].iloc[0]
        # NVDA +10%, AMD +4% -> mean 7%
        assert day2["basket_ret"] == pytest.approx(0.07)

    def test_excludes_benchmark(self, toy_ohlcv):
        panel = build_market_panel(toy_ohlcv)
        basket = basket_aggregate(panel, ["NVDA", "AMD"])
        assert (basket["n_names"] == 2).all()
