# AI Use Disclosure — Assignment 4

This assignment was built in collaboration with Claude (Anthropic) in an
agentic coding session. The disclosure below records what the model did, what
it got wrong, and what the author directed.

## What the AI did

- Wrote the module scaffolding in `src/` and the pytest suite in `tests/`.
- Probed and selected data sources, including diagnosing three dead ends.
- Implemented the econometrics: VAR lag selection, Granger F- and Wald tests,
  the pooled panel specification with date-clustered standard errors, and the
  Sacerdote-style negativity comparisons.
- Drafted the report narrative in `src/report_text.py`.

## What the author directed

- The research design and the decision to purchase X data rather than proceed
  news-only.
- The credit budget and the instruction to ration it (hard ceiling in
  `config.TWITTER_CREDIT_BUDGET`, leaving a reserve untouched).
- The instruction to optimise fetch throughput, which produced the parallel
  harvest.
- Final review of every reported figure.

## Errors the AI made and corrected

These are recorded because they are the honest part of the disclosure.

1. **Mis-diagnosed the network as firewalled.** An early `yfinance` call hung
   and saturated outbound connections, making every host return `000`. The
   model initially reported the sandbox as network-restricted. Killing the
   hung process restored everything.
2. **Scrambled `.gitignore`.** A `sort -u` applied to deduplicate the file
   detached every comment from the rule it documented. Caught and rewritten
   by hand.
3. **Lost paid data to an unhandled exception.** The first harvest hit HTTP
   402 mid-run; the exception propagated before the save, discarding 419
   already-paid-for tweets. Fixed with per-(family, day) checkpointing and a
   dedicated `CreditExhausted` type.
4. **Shipped a known-broken sentiment signal, briefly.** Off-the-shelf VADER
   scored *"Nvidia earnings crushed it, bullish"* at −0.42. Rather than
   accept it, a finance and AI-risk lexicon patch was added and pinned with a
   regression test.
5. **Throttled its own harvest 6× too slowly.** A 2s inter-page delay was
   added defensively after one 429. Measurement showed page latency of 1.2s
   and no 429s across four back-to-back pages; the delay alone projected a
   6-hour harvest. Reduced to 0.35s and parallelised to 8 workers, taking the
   run to under 10 minutes.
6. **Killed its own background harvest twice** with over-broad `pkill`
   patterns. No data was lost, because of fix (3).
7. **Under-reported the API spend by roughly a third.** The in-run counter
   tracked only the final harvest, not the aborted ones, so the model reported
   ~503k credits used when the dashboard showed ~884k. The gap was re-fetched
   work from runs it had interrupted or discarded. It should have reconciled
   against the dashboard before quoting a figure.
8. **Analysed a stale taxonomy without noticing.** After the outlet lists were
   expanded (classified coverage 29% to 61%), a run picked up a cached `.pyc`
   and silently produced the old numbers. Caught by checking the output against
   the expected bucket counts rather than trusting that the edit had applied.
9. **Wrote a conclusion that contradicted its own results.** A hand-written
   sentence claimed "coverage negativity differs sharply by outlet type" when
   the finding was the opposite. This is exactly the failure mode
   `src/report_text.py` exists to prevent, and the paragraph has been rewritten
   to derive its verdict from the test outcome.
10. **Wrote a test guaranteed to fail 1 run in 20.** The panel-Granger null
    test asserted a single p-value exceeded 0.05 -- which a correctly sized
    test violates 5% of the time by construction -- and drew from a shared RNG,
    making it order-dependent. Replaced with a rejection-rate check over 120
    replications. Chasing it produced a genuine result: the test is correctly
    sized at T=60 (4.0% at nominal 5%) but over-rejects at the study's T=17
    (7.3%), which is now reported in Section 4 because it strengthens the
    null finding.

## Verification performed

- 90 unit tests, including Granger tests against constructed series with a
  known answer in both directions, and a Monte Carlo size check on the panel
  specification.
- The quality filters were verified against live data: minimum follower count
  and view count in the retained set sit exactly at the specified floors.
- Every number in the report is generated from the results objects at render
  time rather than typed, so the prose cannot drift from the tables.

## Limitations the AI flagged unprompted

- 17 trading days is too short for time-series inference; the panel
  specification is reported as primary for that reason.
- Headline-level news sentiment is not comparable in *level* to Sacerdote et
  al.'s full-text 91% figure; only the across-bucket differences are claimed.
- Self-reported profile location supports the US/non-US social split for only
  a sub-sample, and is reported as such.
