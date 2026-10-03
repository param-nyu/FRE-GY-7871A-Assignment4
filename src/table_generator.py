"""Exhibit construction: the three required tables, in Markdown and LaTeX.

Every table is built from a DataFrame and emitted in both formats from the
same object, so the PDF and the notebook can never drift apart. Formatting
conventions follow the empirical-finance house style: three decimals for
returns and ratios, stars for significance, and the number of observations on
every panel so a reader can see what each statistic rests on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, SIGNIFICANCE


def stars(p: float) -> str:
    """Conventional significance markers."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _fmt(value, decimals: int = 3) -> str:
    """Format a cell, rendering missing values as an em dash rather than nan."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        # Counts arrive as floats because a NaN anywhere makes the whole column
        # float. Printing "62.000 outlets" is simply wrong, so integral values
        # are rendered as integers.
        if float(value).is_integer() and abs(value) >= 1:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"
    return str(value)


class Exhibit:
    """A titled table that can render itself as Markdown, LaTeX or CSV."""

    def __init__(
        self,
        frame: pd.DataFrame,
        number: int,
        title: str,
        notes: str = "",
        decimals: int = 3,
    ) -> None:
        self.frame = frame
        self.number = number
        self.title = title
        self.notes = notes
        self.decimals = decimals

    @property
    def label(self) -> str:
        return f"Table {self.number}"

    def _formatted(self) -> pd.DataFrame:
        out = self.frame.copy()
        for col in out.columns:
            out[col] = out[col].map(lambda v: _fmt(v, self.decimals))
        return out

    def to_markdown(self) -> str:
        body = self._formatted().to_markdown(index=False)
        parts = [f"**{self.label}. {self.title}**", "", body]
        if self.notes:
            parts += ["", f"*Notes.* {self.notes}"]
        return "\n".join(parts)

    def to_latex(self) -> str:
        body = self._formatted().to_latex(
            index=False, escape=True, column_format="l" + "r" * (self.frame.shape[1] - 1)
        )
        notes = (
            f"\n\\begin{{tablenotes}}\\small\\item {self.notes}\\end{{tablenotes}}"
            if self.notes
            else ""
        )
        return (
            f"\\begin{{table}}[htbp]\\centering\n"
            f"\\caption{{{self.title}}}\n\\label{{tab:{self.number}}}\n"
            f"{body}{notes}\n\\end{{table}}"
        )

    def save(self, output_dir: Path = OUTPUT_DIR, stem: str | None = None) -> dict:
        stem = stem or f"table{self.number}"
        paths = {
            "csv": output_dir / f"{stem}.csv",
            "md": output_dir / f"{stem}.md",
            "tex": output_dir / f"{stem}.tex",
        }
        self.frame.to_csv(paths["csv"], index=False)
        paths["md"].write_text(self.to_markdown())
        paths["tex"].write_text(self.to_latex())
        return paths


# ---------------------------------------------------------------------------
# Table 1: summary statistics
# ---------------------------------------------------------------------------
def table1_summary(
    panel: pd.DataFrame,
    sentiment: pd.DataFrame,
    basket_tickers: list[str],
    semis: list[str],
    hypers: list[str],
) -> Exhibit:
    """Per-name return/volatility moments plus the sentiment series.

    Panel A is the equity basket, Panel B the sentiment indicators. They are
    stacked into one exhibit because the brief asks for a single Table 1, and
    a ``panel`` column keeps them legible.
    """
    rows = []
    for ticker in basket_tickers:
        sub = panel[panel["ticker"] == ticker]
        if sub.empty:
            continue
        group = "Semis" if ticker in semis else ("Hyperscaler" if ticker in hypers else "")
        rows.append({
            "Panel": "A. AI basket",
            "Series": ticker,
            "Group": group,
            "N": int(sub["excess_ret"].notna().sum()),
            "Mean": sub["excess_ret"].mean(),
            "SD": sub["excess_ret"].std(),
            "Min": sub["excess_ret"].min(),
            "Max": sub["excess_ret"].max(),
            "Mean vol": sub["vol_proxy"].mean(),
        })

    if rows:
        eq = pd.DataFrame(rows)
        rows.append({
            "Panel": "A. AI basket", "Series": "Equal-weighted", "Group": "",
            "N": int(eq["N"].max()), "Mean": eq["Mean"].mean(), "SD": eq["SD"].mean(),
            "Min": eq["Min"].min(), "Max": eq["Max"].max(),
            "Mean vol": eq["Mean vol"].mean(),
        })

    for col, name in (
        ("x_ratio", "X sentiment ratio"),
        ("x_mean", "X mean score"),
        ("news_ratio", "News sentiment ratio"),
        ("news_neg_share", "News negativity share"),
    ):
        if col not in sentiment:
            continue
        series = sentiment[col].dropna()
        if series.empty:
            continue
        rows.append({
            "Panel": "B. Sentiment", "Series": name, "Group": "",
            "N": int(series.size), "Mean": series.mean(), "SD": series.std(),
            "Min": series.min(), "Max": series.max(), "Mean vol": np.nan,
        })

    return Exhibit(
        pd.DataFrame(rows),
        number=1,
        title="Summary statistics: AI equity basket and sentiment indicators",
        notes=(
            "Excess return is the daily simple return less the same-day SPY return. "
            "Mean vol is the intraday range proxy 2(H-L)/(H+L). Sample runs "
            "8-30 September 2026. Sentiment ratios are (positive - negative) / total "
            "with neutrals retained in the denominator."
        ),
    )


# ---------------------------------------------------------------------------
# Table 2: Granger causality
# ---------------------------------------------------------------------------
def table2_granger(
    single: pd.DataFrame,
    panel_results: pd.DataFrame,
) -> Exhibit:
    """Granger results, panel specification first, single-series beneath.

    Order is deliberate. The panel test is the one with enough degrees of
    freedom to be informative at this sample size, so it leads; the
    single-series tests follow as a robustness check rather than a headline.
    """
    rows = []
    for _, r in panel_results.iterrows():
        rows.append({
            "Specification": "Panel (firm FE, date-clustered)",
            "Target": r["target"],
            "Sentiment": r["driver"],
            "Lags": int(r["lag"]),
            "N": int(r["n_obs"]),
            "F": r["f_stat"],
            "p": r["p_value"],
            "Sig": stars(r["p_value"]),
            "Sum gamma": r.get("sum_gamma", np.nan),
        })
    # Only the lag the VAR actually selected is tabulated. Printing every lag
    # from 1 to MAX_LAG triples the table's length to show the same six
    # relationships three times each, and invites reading whichever lag
    # happens to look best -- which at 17 observations is exactly the
    # specification-search the short sample cannot support. The full lag grid
    # is still written to outputs/table2.csv for anyone who wants it.
    reported_lag = single["lag"].mode().iloc[0] if len(single) else None
    for _, r in single[single["lag"] == reported_lag].iterrows():
        rows.append({
            "Specification": "Bivariate VAR",
            "Target": r["target"],
            "Sentiment": r["driver"],
            "Lags": int(r["lag"]),
            "N": int(r["n_obs"]),
            "F": r["f_stat"],
            "p": r["p_value"],
            "Sig": stars(r["p_value"]),
            "Sum gamma": np.nan,
        })

    return Exhibit(
        pd.DataFrame(rows),
        number=2,
        title="Granger causality: does lagged sentiment predict returns and volatility?",
        notes=(
            "H0: all coefficients on lagged sentiment are jointly zero. The panel "
            "specification pools the eight basket names with firm fixed effects and "
            "clusters standard errors by date, since the sentiment series is common "
            "across firms on a given day. *** p<0.01, ** p<0.05, * p<0.10. The "
            "single-series tests are shown at the VAR-selected lag only, to avoid "
            "inviting a specification search across lags that 17 observations "
            "cannot support; the full lag grid is in outputs/table2.csv. They are "
            "severely underpowered at this sample length and should not be read as "
            "independent evidence."
        ),
    )


# ---------------------------------------------------------------------------
# Table 3: media negativity bias
# ---------------------------------------------------------------------------
def table3_media_bias(
    by_bucket: pd.DataFrame,
    asymmetry: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> Exhibit:
    """The Sacerdote-style negativity matrix, with the two supporting tests."""
    rows = []
    for _, r in by_bucket.iterrows():
        rows.append({
            "Panel": "A. Negativity by outlet type",
            "Group": r["bucket"],
            "N": int(r["n_articles"]),
            "Outlets": int(r["n_outlets"]) if not np.isnan(r["n_outlets"]) else np.nan,
            "Neg share": r["neg_share"],
            "Pos share": r["pos_share"],
            "Neg/Pos": r["neg_pos_ratio"],
            "Mean tone": r["mean_sentiment"],
            "p vs US": r["p_vs_us"],
            "Sig": stars(r["p_vs_us"]),
        })

    if asymmetry is not None:
        for _, r in asymmetry.iterrows():
            rows.append({
                "Panel": "B. Coverage on up vs down days",
                "Group": r["direction"],
                "N": int(r["n_articles"]),
                "Outlets": np.nan,
                "Neg share": r["neg_share"],
                "Pos share": np.nan,
                "Neg/Pos": r["neg_pos_ratio"],
                "Mean tone": r["mean_sentiment"],
                "p vs US": np.nan,
                "Sig": "",
            })

    if fundamentals is not None:
        for _, r in fundamentals.iterrows():
            rows.append({
                "Panel": "C. Negativity vs fundamentals",
                "Group": r["fundamental"],
                "N": int(r["n"]),
                "Outlets": np.nan,
                "Neg share": np.nan,
                "Pos share": np.nan,
                "Neg/Pos": r["pearson_r"],
                "Mean tone": r["spearman_rho"],
                "p vs US": r["pearson_p"],
                "Sig": stars(r["pearson_p"]),
            })

    return Exhibit(
        pd.DataFrame(rows),
        number=3,
        title="Media negativity bias: US mainstream vs non-US vs scientific/tech outlets",
        notes=(
            "Panel A compares negativity shares across outlet types; p-values are "
            "two-proportion z-tests against the US mainstream bucket. Panel B tests "
            "whether coverage stays negative on days the basket rose, the analogue of "
            "Sacerdote et al.'s finding that negative COVID stories outnumbered "
            "positive ones 5.5-to-1 even as cases fell. In Panel C the Neg/Pos and "
            "Mean tone columns hold the Pearson r and Spearman rho of daily negativity "
            "against each fundamental. Headlines are classified, not full article "
            "text, so levels are not directly comparable to the 91% US figure in the "
            "original study; the across-bucket differences are the finding."
        ),
    )


def write_all(exhibits: list[Exhibit], output_dir: Path = OUTPUT_DIR) -> Path:
    """Save every exhibit and a combined Markdown file of all of them."""
    sections = []
    for exhibit in exhibits:
        exhibit.save(output_dir)
        sections.append(exhibit.to_markdown())
    combined = output_dir / "all_tables.md"
    combined.write_text("\n\n---\n\n".join(sections))
    return combined
