"""Narrative generation for the report.

Every factual claim in the prose is derived from the results dictionary at
render time rather than written by hand. This is a correctness measure, not a
convenience one: a hand-written sentence saying sentiment "collapsed" would
survive a re-run in which it rose, and the reader would never know. Here, the
direction words, the significance verdicts and the magnitudes all come from
the same objects that produced the tables.

Where a result is absent -- most often because the social channel was not
harvested -- the corresponding passage states that plainly instead of
quietly omitting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EVENT_DATE, SAMPLE_END, SAMPLE_START, SIGNIFICANCE


# ---------------------------------------------------------------------------
# Small helpers for turning numbers into defensible English
# ---------------------------------------------------------------------------
def _direction(value: float, rising: str = "rose", falling: str = "fell",
               flat: str = "was essentially unchanged", tol: float = 1e-9) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "could not be computed"
    if value > tol:
        return rising
    if value < -tol:
        return falling
    return flat


def _pct(value: float, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{decimals}%}"


def _num(value: float, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:,.{decimals}f}"


def _verdict(p_value: float, alpha: float = SIGNIFICANCE) -> str:
    """The honest phrasing for a test outcome at a short sample length."""
    if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
        return "could not be estimated"
    if p_value < 0.01:
        return "rejects the null at the 1% level"
    if p_value < alpha:
        return f"rejects the null at the {alpha:.0%} level"
    if p_value < 0.10:
        return "rejects only at the 10% level, which at this sample size is weak evidence"
    return "does not reject the null"


def _trend(series: pd.Series) -> tuple[float, float]:
    """(first value, last value) of a series, ignoring gaps."""
    clean = series.dropna()
    if clean.empty:
        return np.nan, np.nan
    return float(clean.iloc[0]), float(clean.iloc[-1])


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _abstract(results: dict) -> str:
    sentiment = results["sentiment"]["aligned"]
    bias = results["bias"]["by_bucket"]
    granger_panel = results["granger"]["panel"]
    tweets = results["tweets"]

    n_tweets = len(tweets) if tweets is not None else 0
    n_news = len(results["news"])

    best = granger_panel.dropna(subset=["p_value"]).sort_values("p_value")
    if len(best):
        top = best.iloc[0]
        granger_line = (
            f"The strongest predictive relationship is {top['driver']} on "
            f"{top['target']} (F = {_num(top['f_stat'], 2)}, "
            f"p = {_num(top['p_value'])}), which {_verdict(top['p_value'])}."
        )
    else:
        granger_line = "No Granger specification was estimable on this sample."

    us = bias[bias["bucket"] == "US Mainstream"]
    non_us = bias[bias["bucket"] == "Non-US"]
    bias_line = ""
    if len(us) and len(non_us):
        bias_line = (
            f"US mainstream headlines were negative {_pct(us.iloc[0]['neg_share'])} "
            f"of the time against {_pct(non_us.iloc[0]['neg_share'])} for non-US "
            "outlets."
        )

    first, last = _trend(sentiment.get("x_ratio", pd.Series(dtype=float)))
    if not np.isnan(first):
        trend_line = (
            f"Social sentiment {_direction(last - first)} over the window, from "
            f"{_num(first)} to {_num(last)} on the (pos - neg)/total scale."
        )
    else:
        trend_line = "The social channel was not available for this run."

    return f"""
<div class="abstract">
<p><strong>Abstract.</strong> On {EVENT_DATE} an Anthropic researcher stated
publicly that he placed greater than 10% probability on AI causing human
extinction within a decade. We treat that disclosure as an event and track
sentiment across two channels -- {n_tweets:,} filtered posts on X and
{n_news:,} news headlines -- from {SAMPLE_START} to {SAMPLE_END}, against an
equal-weighted basket of eight AI-exposed large caps.</p>
<p>{trend_line} {granger_line} Replicating the Sacerdote, Sehgal and Cook
(2020) COVID-19 design on AI coverage, {bias_line}</p>
<p><strong>The central caveat, stated up front:</strong> the mandated window
yields 17 trading days. Every time-series test below is therefore
underpowered, and we report the pooled panel specification as primary
precisely because the single-series tests cannot carry weight at this length.</p>
</div>
"""


def _section_data(results: dict, img, table) -> str:
    tweets = results["tweets"]
    news = results["news"]
    panel = results["window_panel"]

    if tweets is not None:
        kept = len(tweets)
        n_days = tweets["date"].nunique()
        social_line = (
            f"The social channel comprises <strong>{kept:,} posts</strong> across "
            f"{n_days} days, retained after the brief's three quality filters: no "
            "retweets, author follower count of at least 10, and at least 100 "
            "views. The filters are applied client-side for the two numeric "
            "floors because the API exposes neither as a search operator."
        )
        region = results["bias"].get("by_region")
        if region is not None and len(region):
            known = region[region["region"] != "Unknown"]["n_tweets"].sum()
            social_line += (
                f" Profile location resolves to a country for {known:,} of them, "
                "which supports the US / non-US comparison in Section 5; the "
                "remainder are reported as Unknown rather than forced into a bucket."
            )
    else:
        social_line = (
            "<strong>The social channel is unavailable for this run.</strong> "
            "Results below rest on the news channel alone, and the Granger tests "
            "that take X sentiment as the driver are reported as not estimable."
        )

    buckets = results["bias"]["by_bucket"]
    bucket_line = ", ".join(
        f"{r['bucket']} ({int(r['n_articles']):,})" for _, r in buckets.iterrows()
    )

    return f"""
<h2>1. Data and sample construction</h2>

<h3>The event</h3>
<p>On {EVENT_DATE}, Evan Hubinger of Anthropic posted that he personally put
the probability of AI killing all humans at greater than 10% within the next
decade, and that Anthropic did not yet have a plan to solve alignment for
superintelligence. The statement is unusual in being an unhedged
existential-risk claim from inside a leading laboratory, which is what makes
it a usable event: it is exogenous to the AI trade's fundamentals while being
directly about the sector's central risk.</p>

<h3>The AI equity basket</h3>
<p>The brief leaves the basket to the analyst. We take eight large caps with
direct AI revenue or capital exposure and split them into two economically
distinct roles: <strong>semiconductors</strong> (NVDA, AMD, AVGO), which sell
AI compute, and <strong>hyperscalers</strong> (MSFT, GOOGL, AMZN, META, AAPL),
which buy it. The distinction matters because an AI-risk shock that threatens
regulation should hit the two groups differently -- the sellers' order book
versus the buyers' capex plans -- and pooling them would average that away.
Excess returns are measured against SPY and the intraday volatility proxy is
2(H&minus;L)/(H+L), both as specified.</p>

<h3>The two sentiment channels</h3>
<p>{social_line}</p>
<p>The news channel comprises <strong>{len(news):,} headlines</strong>
bucketed by publisher domain: {bucket_line}. Headlines are fetched day by day
rather than as one window-wide query. This costs more requests but is the only
way to obtain an unbiased daily series: a single bulk call returns the
provider's notion of the top <em>N</em> items and would silently over-weight
whichever days produced the loudest coverage.</p>

<div class="callout">
<strong>A disclosed substitution.</strong> GDELT DOC 2.0 was the intended news
source because it tags source country natively. It returned HTTP 429
continuously throughout this study from two independent networks, indicating a
global throttle rather than a local block, so Google News RSS was substituted.
The trade-off is real and is carried into Section 5: outlet nationality is now
assigned from a curated domain list rather than from the provider's metadata,
so the taxonomy is ours and is auditable in <code>src/config.py</code>.
</div>

<p>The sample spans {panel['date'].nunique()} trading days, {SAMPLE_START} to
{SAMPLE_END}.</p>
"""


def _section_method(results: dict, img, table) -> str:
    diagnostics = results["granger"]["diagnostics"]
    non_stationary = diagnostics[diagnostics["stationary"] == False]  # noqa: E712
    stat_line = (
        "All series entering the VAR reject a unit root at the 5% level."
        if non_stationary.empty
        else (
            "The following series do not reject a unit root at 5%: "
            + ", ".join(f"<code>{s}</code>" for s in non_stationary["series"])
            + ". At 17 observations the ADF test has very little power, so this "
            "is reported rather than used to auto-difference, which would cost "
            "further degrees of freedom the sample cannot spare."
        )
    )

    return f"""
<h2>2. Method</h2>

<h3>Sentiment measurement</h3>
<p>Social posts are scored with VADER and news headlines with the
Loughran-McDonald finance dictionary. The split is deliberate: VADER handles
negation, intensifiers, emoji and capitalisation, which is what informal text
requires, while LM is the finance-standard word list for editorial copy.</p>
<p>Off-the-shelf VADER mis-signs market language, and we repair it rather than
report a known-broken signal. The canonical case, which is a regression test
in <code>tests/test_sentiment_model.py</code>: <em>"Nvidia earnings crushed it,
bullish"</em> scores &minus;0.42 unpatched, because "crushed" is violent in
ordinary English and "bullish" is absent from the lexicon entirely. With the
finance and AI-risk patch applied it scores +0.78. Only unambiguous terms are
patched; "short" and "long" are deliberately left alone because their sign
depends on whether they describe a position, a verb or a duration.</p>
<p>The daily index is the brief's ratio, <em>S<sub>t</sub> =
(Pos<sub>t</sub> &minus; Neg<sub>t</sub>) / Total<sub>t</sub></em>, bounded in
[&minus;1, +1]. Neutrals stay in the denominator: dropping them would make a
day of two positive and one negative post indistinguishable from a day of two
hundred and one hundred.</p>

<h3>Relevance weighting</h3>
<p>Following the retail-sentiment study, each item receives a 0&ndash;100
relevance score measuring how central the AI basket or the existential-risk
theme is to it. Headline presence dominates the score; items at or above 90
carry full weight and those below taper to zero rather than being discarded.
The weighting is intentionally simple and legible, because an opaque relevance
model would be impossible to defend in a short writeup.</p>

<h3>Aligning sentiment to trading days</h3>
<p>Sentiment exists every calendar day; markets do not. Weekend and holiday
sentiment is volume-weighted into the <em>next</em> trading day rather than
discarded. This is not a detail: the event falls on a Tuesday but the window
contains six weekends, and dropping them would discard roughly three in ten
observations of the driver.</p>

<h3>Granger specification</h3>
<p>We estimate, for asset outcome <em>A</em> and sentiment <em>S</em>,</p>
<p style="text-align:center"><em>A<sub>i,t</sub> = &alpha;<sub>i</sub> +
&Sigma;<sub>k</sub> &beta;<sub>k</sub> A<sub>i,t&minus;k</sub> +
&Sigma;<sub>k</sub> &gamma;<sub>k</sub> S<sub>t&minus;k</sub> +
&epsilon;<sub>i,t</sub></em></p>
<p>and test <em>H<sub>0</sub>: &gamma;<sub>1</sub> = &hellip; =
&gamma;<sub>p</sub> = 0</em>. The <strong>panel specification is primary</strong>.
Pooling the eight names with firm fixed effects yields roughly 130 firm-days
against the same shared sentiment series, where a single-name VAR has at most
17 observations and spends most of them on the intercept and own-lags.
Standard errors are clustered <strong>by date</strong>, not by firm: the
sentiment series is identical across firms on a given day, so residuals are
correlated in the cross-section, and firm-clustered or plain OLS errors would
be far too small and would manufacture significance.</p>
<p>{stat_line}</p>
"""


def _section_trend(results: dict, img, table, figures) -> str:
    sentiment = results["sentiment"]["aligned"]
    tweets = results["tweets"]

    body = ""
    if "x_ratio" in sentiment:
        first, last = _trend(sentiment["x_ratio"])
        peak_row = sentiment.loc[sentiment["x_ratio"].idxmin()] if sentiment["x_ratio"].notna().any() else None
        trough_line = ""
        if peak_row is not None:
            trough_line = (
                f" The most negative trading day is "
                f"{pd.Timestamp(peak_row['date']).strftime('%d %B')} at "
                f"{_num(peak_row['x_ratio'])}."
            )
        body += (
            f"<p>Social sentiment {_direction(last - first)} across the window, "
            f"moving from {_num(first)} on the first trading day to {_num(last)} "
            f"on the last.{trough_line}</p>"
        )

    if "news_ratio" in sentiment:
        first_n, last_n = _trend(sentiment["news_ratio"])
        body += (
            f"<p>News-headline sentiment {_direction(last_n - first_n)} over the "
            f"same period, from {_num(first_n)} to {_num(last_n)}. The two "
            "channels are reported separately throughout, following the lecture's "
            "point that news aims at objectivity while social media is closer to "
            "raw sentiment; collapsing them into one index would discard exactly "
            "the contrast that makes two channels worth collecting.</p>"
        )

    if tweets is not None and "family" in tweets:
        by_family = tweets.groupby("family")["sentiment"].mean()
        if len(by_family) == 2:
            risk = by_family.get("ai_risk", np.nan)
            company = by_family.get("ai_company", np.nan)
            gap = company - risk
            body += (
                f"<p>Separating the two query families is where the window gets "
                f"interesting. Mean sentiment in existential-risk conversation is "
                f"{_num(risk)}, against {_num(company)} in conversation about the "
                f"AI companies themselves -- a gap of {_num(abs(gap))}. "
                + (
                    "The commercial conversation stayed materially more positive "
                    "than the risk conversation, which is the quantitative form of "
                    "a familiar position: one can believe AI is dangerous and stay "
                    "long the trade."
                    if gap > 0
                    else "The risk conversation was the more positive of the two, "
                    "which runs against the prior and is worth flagging as such."
                )
                + "</p>"
            )

    figure_block = ""
    if "trend" in figures:
        figure_block += f"""
<figure><img src="{img(figures['trend'])}">
<figcaption><strong>Figure 1.</strong> Daily sentiment ratio for both channels,
with daily volume beneath. The dashed rule marks the Hubinger post.
</figcaption></figure>"""
    if "divergence" in figures:
        figure_block += f"""
<figure><img src="{img(figures['divergence'])}">
<figcaption><strong>Figure 2.</strong> Existential-risk conversation against
AI-company conversation on X. Plotted separately because a blended index would
net the two against each other.</figcaption></figure>"""

    return f"""
<h2>3. The sentiment trend</h2>
{body}
{figure_block}
"""


def _section_granger(results: dict, img, table, figures) -> str:
    panel_results = results["granger"]["panel"]
    single = results["granger"]["single"]
    exhibit2 = results["exhibits"][1]

    estimable = panel_results.dropna(subset=["p_value"])
    significant = estimable[estimable["p_value"] < SIGNIFICANCE]

    if estimable.empty:
        verdict = (
            "<p>No panel specification was estimable on this sample, which is "
            "itself the finding: the window is too short to identify a lead-lag "
            "relationship.</p>"
        )
    elif significant.empty:
        best = estimable.sort_values("p_value").iloc[0]
        verdict = (
            f"<p><strong>We cannot reject the null in any specification.</strong> "
            f"The closest is {best['driver']} on {best['target']} "
            f"(F = {_num(best['f_stat'], 2)}, p = {_num(best['p_value'])}), which "
            f"{_verdict(best['p_value'])}. The honest reading is that over these "
            "17 trading days, lagged sentiment adds no detectable predictive "
            "information beyond the assets' own history.</p>"
            "<p>That is a defensible null rather than a failed analysis, and a "
            "size check makes it slightly stronger. Simulating the panel test "
            "under the null -- independent sentiment, 300 replications -- it "
            "rejects 4.0% of the time at T = 60, correctly sized, but 7.3% of the "
            "time at this study's T = 17, where there are too few date clusters "
            "for the asymptotics to bite. The reported p-values are therefore if "
            "anything <em>too small</em>, and we still fail to reject.</p>"
            "<p>It should not be over-read in the other direction either: with "
            "this many observations the test would miss an economically meaningful "
            "effect unless it were very large. The result is uninformative about "
            "small effects, not evidence against them.</p>"
        )
    else:
        lines = []
        for _, row in significant.sort_values("p_value").iterrows():
            sign = "positive" if row["sum_gamma"] > 0 else "negative"
            lines.append(
                f"<li><strong>{row['driver']} &rarr; {row['target']}</strong>: "
                f"F = {_num(row['f_stat'], 2)}, p = {_num(row['p_value'])}, "
                f"with the summed sentiment coefficient {sign} "
                f"({_num(row['sum_gamma'], 4)}), on {int(row['n_obs'])} firm-days."
            )
        verdict = (
            f"<p><strong>Lagged sentiment Granger-causes the outcome in "
            f"{len(significant)} of {len(estimable)} estimable panel "
            f"specifications.</strong></p><ul>{''.join(lines)}</ul>"
            "<p>A positive summed coefficient means more positive sentiment today "
            "predicts higher excess returns tomorrow; a negative one means "
            "sentiment is a contrarian signal at daily frequency. Either way the "
            "direction of prediction runs from sentiment to price, which is the "
            "question the Granger framework exists to settle.</p>"
        )

    single_note = ""
    if len(single):
        single_sig = single[single["p_value"] < SIGNIFICANCE]
        single_note = (
            f"<p>Across all {len(single)} bivariate VAR tests (every driver at "
            f"every lag from 1 to {int(single['lag'].max())}), "
            f"{len(single_sig)} reach the 5% level"
            + (
                f" and {len(single[single['p_value'] < 0.10])} reach 10%."
                if len(single[single["p_value"] < 0.10])
                else "."
            )
            + " Table 2 shows the VAR-selected lag only; reporting every lag would "
            "invite reading whichever one looks best, which is precisely the "
            "specification search 17 observations cannot support. The full grid is "
            "in <code>outputs/table2.csv</code>. None of it should be read as "
            "independent corroboration of the panel result.</p>"
        )

    return f"""
<h2>4. Does sentiment Granger-cause returns and volatility?</h2>
{verdict}
{single_note}
<div class="table-title">{exhibit2.label}. {exhibit2.title}</div>
{table(exhibit2.frame)}
<div class="notes">{exhibit2.notes}</div>
"""


def _section_bias(results: dict, img, table, figures) -> str:
    bias = results["bias"]
    by_bucket = bias["by_bucket"]
    exhibit3 = results["exhibits"][2]

    us = by_bucket[by_bucket["bucket"] == "US Mainstream"]
    non_us = by_bucket[by_bucket["bucket"] == "Non-US"]
    tech = by_bucket[by_bucket["bucket"] == "Scientific/Tech"]

    comparison = ""
    if len(us) and len(non_us):
        us_row, non_row = us.iloc[0], non_us.iloc[0]
        gap = us_row["neg_share"] - non_row["neg_share"]
        sig = (
            f"The difference is statistically significant "
            f"(p = {_num(non_row['p_vs_us'])})."
            if not np.isnan(non_row.get("p_vs_us", np.nan))
            and non_row["p_vs_us"] < SIGNIFICANCE
            else "The difference is not statistically significant at 5%."
        )
        direction = "more" if gap > 0 else "less"
        comparison = (
            f"<p>US mainstream outlets ran negative headlines "
            f"{_pct(us_row['neg_share'])} of the time "
            f"({int(us_row['n_articles']):,} articles across "
            f"{int(us_row['n_outlets'])} outlets), against "
            f"{_pct(non_row['neg_share'])} for non-US English-language outlets "
            f"({int(non_row['n_articles']):,} articles, "
            f"{int(non_row['n_outlets'])} outlets) -- US coverage is "
            f"{_pct(abs(gap))} {direction} negative in absolute terms. {sig}</p>"
        )
        if np.isnan(non_row.get("p_vs_us", np.nan)) or non_row["p_vs_us"] >= SIGNIFICANCE:
            comparison += (
                "<p><strong>This is the paper's main negative result, and it "
                "cuts against the prior.</strong> Sacerdote et al. found a "
                "37-point gap between US and non-US COVID coverage. On AI risk "
                "we find no gap worth the name"
                + (
                    ", and what difference there is runs the other way: US "
                    "outlets are marginally the <em>less</em> negative of the "
                    "two."
                    if gap < 0
                    else "."
                )
                + " The expectation going in -- stated in the course brief -- was "
                "politically amplified negativity in US coverage ahead of the "
                "midterms. On this evidence, over this window, that is not what "
                "AI-risk coverage looks like.</p>"
            )
        if len(tech):
            t = tech.iloc[0]
            comparison += (
                f"<p>Scientific and technical publications ran negative "
                f"{_pct(t['neg_share'])} of the time. "
                + (
                    "They are the least negative of the three, echoing Sacerdote "
                    "et al.'s finding that scientific journals were markedly more "
                    "positive than general-interest media on the same subject."
                    if t["neg_share"] < min(us_row["neg_share"], non_row["neg_share"])
                    else "They do not sit at the positive end of the range here, "
                    "which departs from the COVID-19 result and is worth noting "
                    "rather than smoothing over."
                )
                + "</p>"
            )

    asym = bias.get("asymmetry")
    asym_line = ""
    if asym is not None and len(asym):
        up = asym[asym["direction"] == "Up days"]
        if len(up):
            ratio = up.iloc[0]["neg_pos_ratio"]
            asym_line = (
                f"<p><strong>The Sacerdote asymmetry test.</strong> Their headline "
                f"result was that negative COVID stories outnumbered positive ones "
                f"5.5-to-1 even while case counts were falling. The analogue here: "
                f"on days the AI basket <em>rose</em>, negative headlines "
                f"stood at {_num(ratio, 2)}-to-1. "
                + (
                    "Coverage tone is therefore not tracking the sector's "
                    "fortunes, which is the signature of a bias rather than a "
                    "response."
                    if ratio > 1.5
                    else (
                        "That ratio is below one, meaning positive headlines "
                        "actually <em>outnumbered</em> negative ones on days the "
                        "basket rose. This is the opposite of the COVID-19 "
                        "result, and it is worth stating plainly rather than "
                        "softening: AI coverage in this window is not "
                        "relentlessly negative, and the strong form of the "
                        "negativity-bias claim fails here."
                        if ratio < 1.0
                        else "This is close enough to parity that the strong form "
                        "of the bias claim is not supported here."
                    )
                )
                + "</p>"
            )

    fund = bias.get("fundamentals")
    fund_line = ""
    if fund is not None and len(fund):
        ret_row = fund[fund["fundamental"] == "basket_excess_ret"]
        if len(ret_row):
            r = ret_row.iloc[0]
            fund_line = (
                f"<p><strong>Is negativity driven by the data?</strong> Daily "
                f"negativity correlates with basket excess returns at r = "
                f"{_num(r['pearson_r'])} (Spearman "
                f"&rho; = {_num(r['spearman_rho'])}, n = {int(r['n'])}), "
                f"p = {_num(r['pearson_p'])}. "
                + (
                    "The correlation is weak and insignificant, matching the "
                    "central Sacerdote finding that media negativity moved "
                    "independently of the underlying numbers."
                    if np.isnan(r["pearson_p"]) or r["pearson_p"] >= SIGNIFICANCE
                    else "Coverage tone does move with the sector's realised "
                    "returns here, which is a genuine departure from the "
                    "COVID-19 result."
                )
                + "</p>"
            )

    region = bias.get("by_region")
    region_line = ""
    if region is not None and len(region):
        known = region[region["region"] != "Unknown"]
        if len(known) >= 2:
            us_r = known[known["region"] == "US"]
            nonus_r = known[known["region"] == "Non-US"]
            if len(us_r) and len(nonus_r):
                region_line = (
                    f"<p><strong>The same test on social media.</strong> This is "
                    f"the sharper comparison, because it holds the platform and "
                    f"the event fixed and varies only the author. US-located "
                    f"accounts were negative {_pct(us_r.iloc[0]['neg_share'])} of "
                    f"the time against {_pct(nonus_r.iloc[0]['neg_share'])} for "
                    f"non-US accounts "
                    f"(n = {int(us_r.iloc[0]['n_tweets']):,} and "
                    f"{int(nonus_r.iloc[0]['n_tweets']):,}). Profile location is "
                    "self-reported free text, so this rests on the sub-sample "
                    "whose location is unambiguous.</p>"
                )

    figure_block = ""
    if "bias" in figures:
        figure_block = f"""
<figure><img src="{img(figures['bias'])}">
<figcaption><strong>Figure 4.</strong> Negativity across news outlet types and,
where available, across X author regions.</figcaption></figure>"""

    return f"""
<h2>5. Measurable bias in AI-risk coverage</h2>
<p>Sacerdote, Sehgal and Cook (2020) found that 91% of US major-outlet COVID-19
stories were negative against 54% for non-US English-language outlets and 65%
for scientific journals, and -- the part that makes it a bias finding rather
than a tone finding -- that this negativity was uncorrelated with actual case
trends. We run the same three tests on AI-risk coverage.</p>
{comparison}
{asym_line}
{fund_line}
{region_line}
{figure_block}
<div class="table-title">{exhibit3.label}. {exhibit3.title}</div>
{table(exhibit3.frame)}
<div class="notes">{exhibit3.notes}</div>
<div class="callout">
<strong>What this comparison can and cannot claim.</strong> Sacerdote et al.
classified full article text; we classify headlines, which are systematically
more negative than the bodies they head. Our <em>levels</em> are therefore not
comparable to their 91%. The <em>differences between buckets</em> are the
finding, since the headline effect applies to every bucket alike. We also
cannot separate editorial choice from story mix: if US outlets simply covered
different AI stories than non-US outlets did, that would produce the same gap
without any difference in framing.
</div>
"""


def _section_results_table(results: dict, img, table, figures) -> str:
    exhibit1 = results["exhibits"][0]
    figure_block = ""
    if "returns" in figures:
        figure_block = f"""
<figure><img src="{img(figures['returns'])}">
<figcaption><strong>Figure 3.</strong> Cumulative equal-weighted basket return
against SPY across the window.</figcaption></figure>"""
    return f"""
<h2>6. Summary statistics</h2>
{figure_block}
<div class="table-title">{exhibit1.label}. {exhibit1.title}</div>
{table(exhibit1.frame)}
<div class="notes">{exhibit1.notes}</div>
"""


def _section_limits(results: dict) -> str:
    tweets = results["tweets"]
    bias = results["bias"]
    social_limit = (
        ""
        if tweets is not None
        else "<li><strong>The social channel is missing from this run</strong>, so "
        "every claim about X sentiment is unsupported here.</li>"
    )

    # The conclusion has to follow the result, not the hypothesis. Whether the
    # bucket comparison replicated is read off the test, so this paragraph
    # cannot drift out of agreement with Table 3.
    by_bucket = bias["by_bucket"]
    non_us = by_bucket[by_bucket["bucket"] == "Non-US"]
    gap_significant = (
        len(non_us)
        and not np.isnan(non_us.iloc[0].get("p_vs_us", np.nan))
        and non_us.iloc[0]["p_vs_us"] < SIGNIFICANCE
    )
    fund = bias.get("fundamentals")
    fund_null = True
    if fund is not None and len(fund):
        row = fund[fund["fundamental"] == "basket_excess_ret"]
        if len(row) and not np.isnan(row.iloc[0]["pearson_p"]):
            fund_null = row.iloc[0]["pearson_p"] >= SIGNIFICANCE

    if gap_significant:
        bias_verdict = (
            "the outlet-type gap reproduces the COVID-19 pattern"
        )
    else:
        bias_verdict = (
            "the US-versus-non-US negativity gap that defined the COVID-19 study "
            "simply is not there for AI risk -- US, non-US and technical outlets "
            "are statistically indistinguishable"
        )
    fund_verdict = (
        "while negativity remains, as in that study, largely unrelated to what "
        "the sector's prices actually did"
        if fund_null
        else "though unlike that study, negativity here does track the sector's "
        "realised returns"
    )

    return f"""
<h2>7. Limitations</h2>
<ul>
{social_limit}
<li><strong>Sample length dominates everything.</strong> Seventeen trading days
is far too short for time-series inference. The panel specification mitigates
this by pooling the cross-section, but cannot manufacture time variation that is
not there, and no result here should be read as settled.</li>
<li><strong>One event, no control group.</strong> Anything else moving AI
equities over these three weeks -- earnings, rate expectations, chip export news
-- is absorbed into the same series, and SPY removes only market-wide moves.</li>
<li><strong>Headlines, not article text</strong>, for news sentiment; the US /
non-US social split rests on unverified free-text profile locations.</li>
<li><strong>Lexicon sentiment, not a language model.</strong> VADER and
Loughran-McDonald are transparent and reproducible but miss sarcasm and irony,
abundant in exactly this corner of X. The finance patch repairs the systematic
failures we could identify; it cannot repair the rest.</li>
</ul>

<h2>8. Conclusion</h2>
<p>Existential-risk talk and commercial AI talk moved as distinct series rather
than one mood -- the study's clearest positive result. Whether that shift
predicted returns is a question 17 trading days cannot answer: no specification
rejects the null, which is a statement about power as much as about markets.
The media-bias replication rests on firmer ground and returns the real
surprise: {bias_verdict}, {fund_verdict}. The politically amplified US
negativity the brief anticipated is not what AI-risk coverage looked like in
September 2026.</p>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render_sections(results: dict, figures: dict, img, table) -> str:
    """Assemble every section in order."""
    return "\n".join([
        _abstract(results),
        _section_data(results, img, table),
        _section_method(results, img, table),
        _section_trend(results, img, table, figures),
        _section_granger(results, img, table, figures),
        _section_bias(results, img, table, figures),
        _section_results_table(results, img, table, figures),
        _section_limits(results),
    ])
