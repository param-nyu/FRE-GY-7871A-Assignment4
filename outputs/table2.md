**Table 2. Granger causality: does lagged sentiment predict returns and volatility?**

| Specification                   | Target            | Sentiment      |   Lags |   N |     F |     p | Sig   | Sum gamma   |
|:--------------------------------|:------------------|:---------------|-------:|----:|------:|------:|:------|:------------|
| Panel (firm FE, date-clustered) | basket_excess_ret | x_ratio        |      2 | 120 | 1.262 | 0.314 |       | 0.025       |
| Panel (firm FE, date-clustered) | basket_excess_ret | news_ratio     |      2 | 120 | 0.663 | 0.531 |       | 0.004       |
| Panel (firm FE, date-clustered) | basket_excess_ret | news_neg_share |      2 | 120 | 1.565 | 0.244 |       | -0.005      |
| Panel (firm FE, date-clustered) | basket_vol        | x_ratio        |      2 | 120 | 2.249 | 0.142 |       | 0.025       |
| Panel (firm FE, date-clustered) | basket_vol        | news_ratio     |      2 | 120 | 2.471 | 0.12  |       | 0.008       |
| Panel (firm FE, date-clustered) | basket_vol        | news_neg_share |      2 | 120 | 2.026 | 0.169 |       | -0.010      |
| Bivariate VAR                   | basket_excess_ret | x_ratio        |      1 |  17 | 0.142 | 0.712 |       | --          |
| Bivariate VAR                   | basket_excess_ret | news_ratio     |      1 |  17 | 0.266 | 0.614 |       | --          |
| Bivariate VAR                   | basket_excess_ret | news_neg_share |      1 |  17 | 0.227 | 0.642 |       | --          |
| Bivariate VAR                   | basket_vol        | x_ratio        |      1 |  17 | 0.254 | 0.622 |       | --          |
| Bivariate VAR                   | basket_vol        | news_ratio     |      1 |  17 | 0.06  | 0.81  |       | --          |
| Bivariate VAR                   | basket_vol        | news_neg_share |      1 |  17 | 0.073 | 0.791 |       | --          |

*Notes.* H0: all coefficients on lagged sentiment are jointly zero. The panel specification pools the eight basket names with firm fixed effects and clusters standard errors by date, since the sentiment series is common across firms on a given day. *** p<0.01, ** p<0.05, * p<0.10. The single-series tests are shown at the VAR-selected lag only, to avoid inviting a specification search across lags that 17 observations cannot support; the full lag grid is in outputs/table2.csv. They are severely underpowered at this sample length and should not be read as independent evidence.