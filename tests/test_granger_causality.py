"""Tests for the econometric engine.

The important tests here are the two that use *constructed* data with a known
answer: one series built so that sentiment genuinely leads returns, and one
built so that it cannot. A Granger implementation that passes both is doing
what it claims; one that only ever returns small p-values is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.granger_causality import (
    adf_test,
    align_to_trading_days,
    granger_test,
    panel_granger_test,
    select_lag,
)

RNG = np.random.default_rng(20260908)


@pytest.fixture
def causal_frame() -> pd.DataFrame:
    """y_t depends on s_{t-1} by construction, so the test must reject H0."""
    n = 200
    s = RNG.normal(size=n)
    y = np.empty(n)
    y[0] = RNG.normal()
    for t in range(1, n):
        y[t] = 0.2 * y[t - 1] + 0.8 * s[t - 1] + 0.1 * RNG.normal()
    return pd.DataFrame({"date": pd.date_range("2026-01-01", periods=n), "y": y, "s": s})


@pytest.fixture
def independent_frame() -> pd.DataFrame:
    """Two independent series, so the test must usually fail to reject."""
    n = 200
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n),
        "y": RNG.normal(size=n),
        "s": RNG.normal(size=n),
    })


class TestGrangerTest:
    def test_detects_genuine_lead(self, causal_frame):
        out = granger_test(causal_frame, "y", "s", max_lag=2)
        assert not out.empty
        assert out["p_value"].min() < 0.01

    def test_does_not_fire_on_independent_series(self, independent_frame):
        out = granger_test(independent_frame, "y", "s", max_lag=2)
        assert out["p_value"].min() > 0.05

    def test_direction_matters(self, causal_frame):
        """s -> y is real; y -> s is not. A transposed implementation would
        report both, which is the classic way to misreport this test."""
        forward = granger_test(causal_frame, "y", "s", max_lag=2)["p_value"].min()
        reverse = granger_test(causal_frame, "s", "y", max_lag=2)["p_value"].min()
        assert forward < 0.01
        assert reverse > forward

    def test_reports_degrees_of_freedom(self, causal_frame):
        out = granger_test(causal_frame, "y", "s", max_lag=2)
        assert (out["df_num"] > 0).all()
        assert (out["df_denom"] > 0).all()

    def test_short_sample_degrades_gracefully(self):
        tiny = pd.DataFrame({"y": [1.0, 2.0, 3.0], "s": [1.0, 0.0, 1.0]})
        out = granger_test(tiny, "y", "s", max_lag=3)
        assert isinstance(out, pd.DataFrame)  # no exception


class TestPanelGranger:
    def test_detects_shared_lead_across_entities(self):
        """One sentiment series drives every firm, as in the real design."""
        days = pd.date_range("2026-01-01", periods=60, freq="B")
        s = RNG.normal(size=len(days))
        sentiment = pd.DataFrame({"date": days, "s": s})

        rows = []
        for ticker in list("ABCDEFGH"):
            y = np.empty(len(days))
            y[0] = RNG.normal() * 0.01
            for t in range(1, len(days)):
                y[t] = 0.1 * y[t - 1] + 0.02 * s[t - 1] + 0.002 * RNG.normal()
            rows.append(pd.DataFrame({"date": days, "ticker": ticker, "ret": y}))
        panel = pd.concat(rows, ignore_index=True)

        out = panel_granger_test(panel, sentiment, "ret", "s", max_lag=1)
        assert out["n_entities"] == 8
        assert out["p_value"] < 0.05

    def test_null_is_correctly_sized(self):
        """Under the null the test must reject at about its nominal rate.

        This replaced a single-draw assertion that the p-value exceeded 0.05.
        That version was wrong twice over: it drew from a module-level RNG, so
        the result depended on which tests ran before it, and a correctly
        sized test rejects 5% of the time *by construction* -- so the old
        assertion was guaranteed to fail roughly one run in twenty, and did.

        Checking the rejection *rate* over many replications is the question
        actually worth asking of a test statistic. A separate simulation at
        the study's own T=17 shows the rate rising to about 7%, which is why
        the report treats the panel p-values as mildly liberal.
        """
        rng = np.random.default_rng(31415)
        days = pd.date_range("2026-01-01", periods=60, freq="B")

        p_values = []
        for _ in range(120):
            sentiment = pd.DataFrame({"date": days, "s": rng.normal(size=len(days))})
            panel = pd.concat(
                [
                    pd.DataFrame({"date": days, "ticker": t,
                                  "ret": rng.normal(size=len(days)) * 0.01})
                    for t in list("ABCDEFGH")
                ],
                ignore_index=True,
            )
            out = panel_granger_test(panel, sentiment, "ret", "s", max_lag=1)
            if not np.isnan(out["p_value"]):
                p_values.append(out["p_value"])

        rejection_rate = np.mean(np.array(p_values) < 0.05)
        assert len(p_values) > 100
        # Generous band: 120 replications put the Monte Carlo standard error
        # on a 5% rate at about 2 points, so anything inside [0, 0.13] is
        # consistent with correct size while still catching a test that
        # rejects most of the time.
        assert rejection_rate < 0.13, f"over-rejects under the null: {rejection_rate:.3f}"

    def test_returns_shaped_result_on_tiny_sample(self):
        days = pd.date_range("2026-09-08", periods=3, freq="B")
        sentiment = pd.DataFrame({"date": days, "s": [0.1, -0.2, 0.0]})
        panel = pd.DataFrame({"date": list(days) * 2, "ticker": ["A"] * 3 + ["B"] * 3,
                              "ret": [0.01, -0.02, 0.0, 0.02, -0.01, 0.005]})
        out = panel_granger_test(panel, sentiment, "ret", "s", max_lag=1)
        assert set(out) >= {"f_stat", "p_value", "n_obs"}


class TestAlignment:
    def test_weekend_sentiment_rolls_into_next_trading_day(self):
        """The event is followed by a weekend, so this is load-bearing."""
        sentiment = pd.DataFrame({
            "date": pd.to_datetime(
                ["2026-09-11", "2026-09-12", "2026-09-13", "2026-09-14"]
            ),
            "x_ratio": [0.1, 0.5, 0.5, 0.2],
            "x_n": [10, 10, 10, 10],
        })
        trading = pd.to_datetime(["2026-09-11", "2026-09-14"])
        out = align_to_trading_days(sentiment, trading)

        assert len(out) == 2
        monday = out[out["date"] == pd.Timestamp("2026-09-14")].iloc[0]
        # Sat + Sun + Mon collapse into Monday.
        assert monday["n_calendar_days"] == 3
        assert monday["x_n"] == 30

    def test_volume_columns_are_summed_not_averaged(self):
        sentiment = pd.DataFrame({
            "date": pd.to_datetime(["2026-09-12", "2026-09-13"]),
            "x_ratio": [0.4, 0.6], "x_n": [100, 300],
        })
        out = align_to_trading_days(sentiment, pd.to_datetime(["2026-09-14"]))
        assert out["x_n"].iloc[0] == 400

    def test_value_columns_are_volume_weighted(self):
        sentiment = pd.DataFrame({
            "date": pd.to_datetime(["2026-09-12", "2026-09-13"]),
            "x_ratio": [0.0, 1.0], "x_n": [100, 300],
        })
        out = align_to_trading_days(sentiment, pd.to_datetime(["2026-09-14"]))
        # (0*100 + 1*300) / 400
        assert out["x_ratio"].iloc[0] == pytest.approx(0.75)

    def test_drop_mode_keeps_only_trading_days(self):
        sentiment = pd.DataFrame({
            "date": pd.to_datetime(["2026-09-11", "2026-09-12", "2026-09-14"]),
            "x_ratio": [0.1, 0.2, 0.3], "x_n": [1, 1, 1],
        })
        trading = pd.to_datetime(["2026-09-11", "2026-09-14"])
        out = align_to_trading_days(sentiment, trading, how="drop")
        assert len(out) == 2

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            align_to_trading_days(pd.DataFrame({"date": []}), [], how="nonsense")


class TestDiagnostics:
    def test_adf_flags_stationary_series(self):
        out = adf_test(pd.Series(RNG.normal(size=200)), "white noise")
        assert out["stationary"] is True

    def test_adf_flags_random_walk(self):
        out = adf_test(pd.Series(np.cumsum(RNG.normal(size=200))), "random walk")
        assert out["stationary"] is False

    def test_adf_handles_degenerate_input(self):
        out = adf_test(pd.Series([1.0, 1.0, 1.0]), "constant")
        assert np.isnan(out["adf_stat"])

    def test_lag_selection_respects_sample_length(self):
        """With 16 observations the feasible lag order must be small."""
        short = pd.DataFrame({"y": RNG.normal(size=16), "s": RNG.normal(size=16)})
        info = select_lag(short, max_lag=3)
        assert info["max_feasible_lag"] <= 3
        assert info["selected"] >= 1
