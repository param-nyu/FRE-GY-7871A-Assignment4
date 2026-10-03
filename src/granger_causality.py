"""Granger predictive-causality tests for sentiment on returns and volatility.

The specification the brief asks for:

.. math::

    A_{i,t} = \\alpha_i + \\sum_{k=1}^{p} \\beta_{i,k} A_{i,t-k}
            + \\sum_{k=1}^{p} \\gamma_{i,k} S_{t-k} + \\epsilon_{i,t}

with :math:`H_0: \\gamma_{i,1} = \\dots = \\gamma_{i,p} = 0`. Rejecting it says
lagged sentiment carries information about :math:`A_{i,t}` beyond the asset's
own history -- Granger's answer to the chicken-and-egg question of whether
sentiment moves price or price moves sentiment.

**A warning that governs how these results should be read.** The sample window
is 23 calendar days, about 16 trading days. A per-name VAR with even two lags
spends most of its degrees of freedom on the intercept and own-lags, so a
single-series test here has very little power, and any individual p-value
should be treated as suggestive at best. :func:`panel_granger_test` is the
primary specification for that reason: pooling the eight basket names gives
roughly 130 firm-days against the same shared sentiment series, and clusters
standard errors by date because :math:`S_t` is common across firms -- ignoring
that cross-sectional dependence would understate the standard errors badly and
manufacture significance.

Both specifications are reported. Where they disagree, the panel is the one to
believe.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

from .config import MAX_LAG


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def align_to_trading_days(
    sentiment: pd.DataFrame,
    trading_days: pd.Series | pd.Index,
    date_col: str = "date",
    how: str = "accumulate",
) -> pd.DataFrame:
    """Map calendar-day sentiment onto trading days.

    Sentiment exists every calendar day; markets do not. ``accumulate`` is the
    economically right default: sentiment generated on a Saturday cannot be
    traded until Monday, so weekend and holiday sentiment is averaged into the
    *next* trading day rather than discarded. Dropping it would throw away
    roughly three of every ten observations, and in this window the event
    itself is followed by a weekend.

    Volume columns (``*_n``, ``*_pos``, ``*_neg``, ``*_neu``) are summed; all
    other numeric columns are averaged, weighted by daily volume where a
    volume column is present.
    """
    if how not in {"accumulate", "drop"}:
        raise ValueError("how must be 'accumulate' or 'drop'")

    trading = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(trading_days)).unique()))
    frame = sentiment.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])

    if how == "drop":
        return frame[frame[date_col].isin(trading)].reset_index(drop=True)

    # searchsorted with side="left" maps each calendar day to the first
    # trading day on or after it.
    positions = trading.searchsorted(frame[date_col].values, side="left")
    keep = positions < len(trading)
    frame = frame.loc[keep].copy()
    frame["trade_date"] = trading[positions[keep]]

    count_cols = [c for c in frame.columns if c.endswith(("_n", "_pos", "_neg", "_neu"))]
    value_cols = [
        c
        for c in frame.select_dtypes("number").columns
        if c not in count_cols
    ]
    weight_col = next((c for c in count_cols if c.endswith("_n")), None)

    rows = []
    for trade_date, group in frame.groupby("trade_date"):
        row = {"date": trade_date}
        for col in count_cols:
            row[col] = group[col].sum()
        weights = group[weight_col] if weight_col else None
        for col in value_cols:
            vals = group[col]
            if weights is not None and weights.sum() > 0 and vals.notna().any():
                mask = vals.notna()
                row[col] = np.average(vals[mask], weights=weights[mask]) if mask.any() else np.nan
            else:
                row[col] = vals.mean()
        row["n_calendar_days"] = len(group)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def adf_test(series: pd.Series, name: str = "") -> dict:
    """Augmented Dickey-Fuller test.

    Granger causality on non-stationary series produces spurious rejections,
    so every series entering a VAR is tested first. Returns are stationary by
    construction; the volatility proxy and the sentiment ratio are not
    guaranteed to be, and in a 16-day sample the test itself is weak -- the
    result is reported rather than used to auto-difference.
    """
    clean = pd.Series(series).dropna()
    if len(clean) < 6 or clean.nunique() <= 1:
        return {"series": name, "adf_stat": np.nan, "p_value": np.nan,
                "stationary": np.nan, "n": len(clean)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, _, nobs, _, _ = adfuller(clean, autolag="AIC")
    return {
        "series": name,
        "adf_stat": float(stat),
        "p_value": float(pval),
        "stationary": bool(pval < 0.05),
        "n": int(nobs),
    }


def select_lag(data: pd.DataFrame, max_lag: int = MAX_LAG) -> dict:
    """Lag order by AIC and BIC from a VAR fit.

    The usable maximum is capped by sample length: a VAR on two variables with
    p lags needs 2p+1 parameters per equation, so p is bounded well below
    :data:`MAX_LAG` here. The cap is reported alongside the choice so the
    report can state that the lag order was constrained by the window, not
    chosen freely.
    """
    clean = data.dropna()
    n_obs, n_vars = clean.shape
    # Keep at least 5 residual degrees of freedom per equation.
    feasible = max(1, int((n_obs - 5 - 1) / max(n_vars, 1)))
    usable = int(min(max_lag, feasible))

    out = {"n_obs": int(n_obs), "max_feasible_lag": usable, "aic": np.nan,
           "bic": np.nan, "selected": 1}
    if usable < 1 or n_obs < 10:
        return out

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = VAR(clean).select_order(maxlags=usable)
        aic = int(sel.aic) if sel.aic else 1
        bic = int(sel.bic) if sel.bic else 1
        out.update({"aic": aic, "bic": bic, "selected": max(1, min(aic, usable))})
    except (ValueError, np.linalg.LinAlgError):
        pass
    return out


# ---------------------------------------------------------------------------
# Single-series Granger test
# ---------------------------------------------------------------------------
def granger_test(
    frame: pd.DataFrame,
    target: str,
    driver: str,
    max_lag: int = MAX_LAG,
) -> pd.DataFrame:
    """Does ``driver`` Granger-cause ``target``? One row per lag order.

    statsmodels' ``grangercausalitytests`` orders its input ``[target,
    driver]`` and tests whether the *second* column helps predict the first.
    Getting that order backwards silently tests the reverse hypothesis, which
    is the single easiest way to misreport this result, so the columns are
    assembled explicitly here rather than passed through positionally.
    """
    data = frame[[target, driver]].dropna()
    rows = []
    feasible = max(1, int((len(data) - 5) / 2))
    for lag in range(1, int(min(max_lag, feasible)) + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = grangercausalitytests(data, maxlag=[lag])
            f_stat, p_value, df_denom, df_num = result[lag][0]["ssr_ftest"]
        except (ValueError, np.linalg.LinAlgError, KeyError):
            continue
        rows.append({
            "target": target,
            "driver": driver,
            "lag": lag,
            "f_stat": float(f_stat),
            "p_value": float(p_value),
            "df_num": int(df_num),
            "df_denom": int(df_denom),
            "n_obs": int(len(data)),
        })
    return pd.DataFrame(rows)


def run_var_granger(
    frame: pd.DataFrame,
    target: str,
    driver: str,
    max_lag: int = MAX_LAG,
) -> dict:
    """Fit the bivariate VAR and return its Wald test for ``driver -> target``.

    The Wald statistic complements the F-test above: the F-test is computed
    equation-by-equation on OLS residuals, the Wald test comes from the fitted
    VAR system. Reporting both guards against a result that is an artefact of
    one estimator.
    """
    data = frame[[target, driver]].dropna()
    lag_info = select_lag(data, max_lag)
    p = lag_info["selected"]

    out = {"target": target, "driver": driver, "lag": p,
           "n_obs": lag_info["n_obs"], "wald_stat": np.nan, "p_value": np.nan,
           "max_feasible_lag": lag_info["max_feasible_lag"]}
    if len(data) < 10 or p < 1:
        return out

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = VAR(data).fit(p)
            test = fitted.test_causality(target, [driver], kind="wald")
        out["wald_stat"] = float(test.test_statistic)
        out["p_value"] = float(test.pvalue)
    except (ValueError, np.linalg.LinAlgError):
        pass
    return out


# ---------------------------------------------------------------------------
# Panel Granger test  (primary specification)
# ---------------------------------------------------------------------------
def panel_granger_test(
    panel: pd.DataFrame,
    sentiment: pd.DataFrame,
    target: str,
    driver: str,
    max_lag: int = 2,
    entity_col: str = "ticker",
    date_col: str = "date",
) -> dict:
    """Pooled Granger test with firm fixed effects and date-clustered errors.

    Estimates

    .. math::
        A_{i,t} = \\alpha_i + \\sum_k \\beta_k A_{i,t-k}
                + \\sum_k \\gamma_k S_{t-k} + \\epsilon_{i,t}

    and reports the joint F-test of :math:`\\gamma = 0`.

    Two choices matter and are deliberate. Firm fixed effects absorb
    level differences across the basket, so the test is identified from
    within-firm time variation. Standard errors are clustered by **date**
    rather than by firm: the sentiment series is identical across firms on a
    given day, so residuals are correlated in the cross-section, and
    firm-clustered or plain OLS errors would be far too small.
    """
    merged = panel.merge(
        sentiment[[date_col, driver]], on=date_col, how="inner"
    ).sort_values([entity_col, date_col])

    work = merged[[entity_col, date_col, target, driver]].copy()
    for lag in range(1, max_lag + 1):
        work[f"y_lag{lag}"] = work.groupby(entity_col)[target].shift(lag)
        work[f"s_lag{lag}"] = work.groupby(entity_col)[driver].shift(lag)
    work = work.dropna()

    sentiment_terms = [f"s_lag{k}" for k in range(1, max_lag + 1)]
    own_terms = [f"y_lag{k}" for k in range(1, max_lag + 1)]

    out = {
        "target": target, "driver": driver, "lag": max_lag,
        "n_obs": int(len(work)), "n_entities": int(work[entity_col].nunique()),
        "f_stat": np.nan, "p_value": np.nan, "r2": np.nan,
        "sum_gamma": np.nan,
    }
    if len(work) < 20 or work[entity_col].nunique() < 2:
        return out

    dummies = pd.get_dummies(work[entity_col], prefix="fe", drop_first=True, dtype=float)
    X = pd.concat([work[own_terms + sentiment_terms].astype(float), dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = work[target].astype(float)

    try:
        model = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": work[date_col]}
        )
        hypothesis = ", ".join(f"{t} = 0" for t in sentiment_terms)
        wald = model.f_test(hypothesis)
        out["f_stat"] = float(np.squeeze(wald.fvalue))
        out["p_value"] = float(np.squeeze(wald.pvalue))
        out["r2"] = float(model.rsquared)
        out["sum_gamma"] = float(sum(model.params[t] for t in sentiment_terms))
    except (ValueError, np.linalg.LinAlgError):
        pass
    return out


def run_all_tests(
    basket: pd.DataFrame,
    sentiment: pd.DataFrame,
    targets: tuple[str, ...] = ("basket_excess_ret", "basket_vol"),
    drivers: tuple[str, ...] = ("x_ratio", "news_ratio"),
    max_lag: int = MAX_LAG,
) -> pd.DataFrame:
    """Every (target, driver) pair, single-series F-test and VAR Wald test."""
    merged = basket.merge(sentiment, on="date", how="inner")
    rows = []
    for target in targets:
        for driver in drivers:
            if target not in merged or driver not in merged:
                continue
            f_rows = granger_test(merged, target, driver, max_lag)
            wald = run_var_granger(merged, target, driver, max_lag)
            for _, r in f_rows.iterrows():
                rec = r.to_dict()
                rec["wald_stat"] = wald["wald_stat"] if r["lag"] == wald["lag"] else np.nan
                rec["wald_p"] = wald["p_value"] if r["lag"] == wald["lag"] else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)
