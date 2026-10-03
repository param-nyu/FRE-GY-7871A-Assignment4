"""Download OHLCV for the AI basket + SPY, and the FRED risk-free series."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data_loader import (
    basket_aggregate, build_market_panel, download_ohlcv, download_risk_free,
)

def main() -> None:
    print(f"tickers: {config.ALL_TICKERS}")
    ohlcv = download_ohlcv()
    print(f"OHLCV rows={len(ohlcv)} tickers={ohlcv['ticker'].nunique()} "
          f"range={ohlcv['date'].min().date()} -> {ohlcv['date'].max().date()}")

    rf = download_risk_free()
    print(f"risk-free obs={len(rf)} last={rf.iloc[-1]:.8f}")

    panel = build_market_panel(ohlcv, rf)
    panel.to_csv(config.INTERIM_DIR / "market_panel.csv", index=False)

    window = panel[panel["date"].between(config.SAMPLE_START, config.SAMPLE_END)]
    print(f"\nin-window rows={len(window)} "
          f"trading days={window['date'].nunique()}")
    print(window.groupby("ticker")[["excess_ret", "vol_proxy"]].mean().round(5))

    basket = basket_aggregate(panel, config.AI_BASKET)
    basket.to_csv(config.INTERIM_DIR / "basket_daily.csv", index=False)
    print(f"\nbasket rows={len(basket)}")

if __name__ == "__main__":
    main()
