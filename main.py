import os
import requests
import pandas as pd
from alpaca_trade_api import REST

# --- ENVIRONMENT VARIABLES ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # use live URL when ready

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"

# --- INIT CLIENT ---
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

# --- SCREENER PARAMETERS ---
def passes_screen(ticker):
    """Check Polygon data against swing trade parameters."""
    try:
        # Get last 50 days of daily candles
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/2024-01-01/2025-01-01?adjusted=true&limit=60&apiKey={POLYGON_API_KEY}"
        r = requests.get(url).json()

        if "results" not in r:
            return False

        df = pd.DataFrame(r["results"])
        df["close"] = df["c"]
        
        # Compute moving averages
        df["sma20"] = df["close"].rolling(20).mean()
        df["sma50"] = df["close"].rolling(50).mean()

        latest = df.iloc[-1]

        # RSI calculation (14-period)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        latest_rsi = df["rsi"].iloc[-1]

        # Technical filters
        if not (latest["close"] > latest["sma20"] and latest["close"] > latest["sma50"]):
            return False
        if not (45 <= latest_rsi <= 60):
            return False

        # Get fundamentals from Polygon
        fin_url = f"{POLYGON_BASE_URL}/vX/reference/financials?ticker={ticker}&limit=1&apiKey={POLYGON_API_KEY}"
        fin = requests.get(fin_url).json()
        if "results" not in fin or len(fin["results"]) == 0:
            return False

        fundamentals = fin["results"][0]

        # Basic fundamental checks
        roe = fundamentals.get("financials", {}).get("roe", {}).get("value", 0)
        revenue_growth = fundamentals.get("financials", {}).get("revenueGrowth", {}).get("value", 0)
        eps_growth = fundamentals.get("financials", {}).get("epsGrowth", {}).get("value", 0)
        debt_equity = fundamentals.get("financials", {}).get("debtToEquity", {}).get("value", 1)

        if eps_growth < 0.05 or revenue_growth < 0.05:
            return False
        if roe < 0.15:
            return False
        if debt_equity > 1.0:
            return False

        return True

    except Exception as e:
        print(f"Error screening {ticker}: {e}")
        return False


def main():
    # Get tradable assets from Alpaca
    assets = alpaca.list_assets(status="active")
    tradable = [a.symbol for a in assets if a.tradable]

    # Check account buying power
    account = alpaca.get_account()
    cash = float(account.cash)
    allocation = cash * 0.05  # 5% per trade

    for ticker in tradable:
        if passes_screen(ticker):
            print(f"Buying {ticker} with ${allocation:.2f}")
            try:
                alpaca.submit_order(
                    symbol=ticker,
                    notional=allocation,  # buy with 5% of cash
                    side="buy",
                    type="market",
                    time_in_force="day"
                )
            except Exception as e:
                print(f"Order failed for {ticker}: {e}")


if __name__ == "__main__":
    main()
