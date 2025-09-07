#!/usr/bin/env python3
"""
Alpaca + Polygon Screener Buyer (5% swing entries)

Flow:
  1) Auth to Alpaca (prefers APCA_* env vars) and sanity-check account.
  2) Pull active, tradable US equities from Alpaca (liquidity/practicality filters).
  3) For each candidate, fetch Polygon data and apply screen:
       - TECHNICALS (priority for 5% swings):
           * Close > SMA20 and Close > SMA50
           * RSI(14) in [45, 60]
           * MACD line > signal (bullish)
           * Within 10% of 52-week high
           * Relative Volume >= 1.5 (yesterday vs 30D avg)
       - FUNDAMENTALS (light sanity):
           * EPS YoY >= 5%
           * Revenue YoY >= 5%
           * ROE >= 15%
           * Debt/Equity <= 1.0
           * PEG <= 2.0 (if available)
     (Fundamentals are robust to missing fields; we only pass if available fields meet thresholds.)
  4) If a ticker passes, submit a MARKET BUY with notional = 5% of current buying power.

Notes:
  - No sell logic here; your separate bot handles exits.
  - Keeps logs concise and clear for observability.
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame

# -----------------------------
# Env & Clients
# -----------------------------
def get_env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip().strip('"').strip("'")

ALPACA_API_KEY    = get_env_str("ALPACA_API_KEY") or get_env_str("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = get_env_str("ALPACA_SECRET_KEY") or get_env_str("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = get_env_str("APCA_API_BASE_URL")  # prefer this, like your other bots
if not APCA_API_BASE_URL:
    # Fallback if not provided (kept for portability)
    mode = get_env_str("ALPACA_MODE", "paper").lower()
    APCA_API_BASE_URL = "https://paper-api.alpaca.markets" if mode == "paper" else "https://api.alpaca.markets"

POLYGON_API_KEY   = get_env_str("POLYGON_API_KEY")
POLYGON_BASE      = "https://api.polygon.io"

def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[buy-screener] {now} | {msg}", flush=True)

if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
    log("❌ Missing ALPACA_API_KEY/APCA_API_KEY_ID or ALPACA_SECRET_KEY/APCA_API_SECRET_KEY")
    sys.exit(1)
if not POLYGON_API_KEY:
    log("❌ Missing POLYGON_API_KEY")
    sys.exit(1)

log(f"🔑 Alpaca URL: {APCA_API_BASE_URL} | Key prefix: {ALPACA_API_KEY[:4]}*** | Polygon key prefix: {POLYGON_API_KEY[:4]}***")

api = REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=APCA_API_BASE_URL)

# fail fast on auth
try:
    acct = api.get_account()
    log(f"✅ Alpaca auth OK | account={acct.account_number} | status={acct.status} | cash={acct.cash} | bp={acct.buying_power}")
except Exception as e:
    log(f"❌ Alpaca authentication failed: {e}")
    sys.exit(1)

# -----------------------------
# Parameters
# -----------------------------
# Universe control
MAX_TICKERS_TO_CHECK = int(get_env_str("MAX_TICKERS_TO_CHECK", "200"))  # cap API calls
MIN_PRICE = float(get_env_str("MIN_PRICE", "3.0"))                      # avoid penny stocks
MIN_AVG_DOLLAR_VOL = float(get_env_str("MIN_AVG_DOLLAR_VOL", "5_000_000"))  # 30D avg $-volume

# Screener thresholds (tuned for ~5% swing targets)
RSI_MIN, RSI_MAX = 45.0, 60.0
REL_VOL_MIN = 1.5       # yesterday vs 30D avg
WITHIN_HI_PCT = 0.10    # within 10% of 52w high

# Fundamentals (light)
EPS_GROWTH_MIN = 0.05
REV_GROWTH_MIN = 0.05
ROE_MIN = 0.15
DE_MIN, PEG_MAX = 1.0, 2.0

# Purchase sizing
BUY_FRACTION_OF_BP = float(get_env_str("BUY_FRACTION_OF_BP", "0.05"))  # 5% of buying power per signal
TIME_IN_FORCE = get_env_str("TIME_IN_FORCE", "day")

# -----------------------------
# Helpers: Indicators
# -----------------------------
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, min_periods=n).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

# -----------------------------
# Polygon fetchers
# -----------------------------
def poly_get(url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    params = dict(params or {})
    params["apiKey"] = POLYGON_API_KEY
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 429:
            log("⏳ Polygon rate-limited; sleeping 2s...")
            time.sleep(2)
            r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"Polygon error: {e} | url={url}")
        return None

def fetch_daily_ohlcv(ticker: str, days: int = 260) -> Optional[pd.DataFrame]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(days * 1.6))  # buffer for weekends/holidays
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    j = poly_get(url, {"adjusted": "true", "limit": 5000})
    if not j or "results" not in j:
        return None
    df = pd.DataFrame(j["results"])
    if df.empty:
        return None
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.set_index("t").sort_index()
    df.rename(columns={"c": "close", "v": "volume", "o": "open", "h": "high", "l": "low"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]]

def fetch_fundamentals_latest(ticker: str) -> Dict[str, Optional[float]]:
    """
    Try Polygon financials endpoint (vX). Fields differ by company;
    we pull what we can and return None when missing.
    """
    url = f"{POLYGON_BASE}/vX/reference/financials"
    j = poly_get(url, {"ticker": ticker, "limit": 1})
    out = {"eps_yoy": None, "rev_yoy": None, "roe": None, "de": None, "peg": None}
    if not j or "results" not in j or not j["results"]:
        return out
    fin = j["results"][0]
    f = fin.get("financials", {}) or {}

    # attempt common field names
    # EPS & Revenue YoY growth
    out["eps_yoy"] = (f.get("epsGrowth", {}) or {}).get("value")
    out["rev_yoy"] = (f.get("revenueGrowth", {}) or {}).get("value")

    # Profitability
    out["roe"] = (f.get("roe", {}) or {}).get("value")

    # Leverage
    out["de"] = (f.get("debtToEquity", {}) or {}).get("value")

    # Valuation
    out["peg"] = (f.get("pegRatio", {}) or {}).get("value")

    return out

# -----------------------------
# Screen logic
# -----------------------------
def within_pct_of_high(series: pd.Series, pct: float) -> bool:
    if series.empty:
        return False
    recent_close = series.iloc[-1]
    hi = series.max()
    return (hi - recent_close) / hi <= pct if hi > 0 else False

def rel_volume_yday_vs_30d(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 31:
        return None
    yday_vol = df["volume"].iloc[-1]
    avg30 = df["volume"].iloc[-31:-1].mean()
    if avg30 <= 0:
        return None
    return yday_vol / avg30

def avg_dollar_volume_30d(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 31:
        return None
    px = df["close"].iloc[-31:-1]
    vol = df["volume"].iloc[-31:-1]
    return float((px * vol).mean())

def technicals_pass(df: pd.DataFrame) -> bool:
    closes = df["close"]
    if closes.isna().any() or len(closes) < 50:
        return False

    sma20 = sma(closes, 20).iloc[-1]
    sma50 = sma(closes, 50).iloc[-1]
    close = closes.iloc[-1]
    if not (close > sma20 and close > sma50):
        return False

    rsi_now = rsi(closes, 14).iloc[-1]
    if not (RSI_MIN <= rsi_now <= RSI_MAX):
        return False

    macd_line, macd_sig, _ = macd(closes)
    if not (macd_line.iloc[-1] > macd_sig.iloc[-1]):
        return False

    if not within_pct_of_high(closes.tail(252), WITHIN_HI_PCT):
        return False

    rv = rel_volume_yday_vs_30d(df)
    if rv is None or rv < REL_VOL_MIN:
        return False

    return True

def fundamentals_pass(ticker: str) -> bool:
    f = fetch_fundamentals_latest(ticker)
    # If Polygon lacks fundamentals for the ticker, we conservatively fail
    eps = f["eps_yoy"]; rev = f["rev_yoy"]; roe = f["roe"]; de = f["de"]; peg = f["peg"]

    if eps is None or rev is None or roe is None or de is None:
        return False
    if eps < EPS_GROWTH_MIN: return False
    if rev < REV_GROWTH_MIN: return False
    if roe < ROE_MIN: return False
    if de > DE_MIN: return False
    if peg is not None and peg > PEG_MAX: return False
    return True

# -----------------------------
# Universe & Buy
# -----------------------------
def get_candidate_universe() -> List[str]:
    """
    Build a reasonable, liquid US equities universe from Alpaca.
    Applies some basic symbol filters to avoid OTC, ETFs by pattern, etc.
    """
    assets = api.list_assets(status="active")
    cands = []
    for a in assets:
        # Only US equities that are tradable
        if getattr(a, "class", None) != "us_equity":
            continue
        if not a.tradable:
            continue
        sym = a.symbol.upper()
        # Simple symbol hygiene: exclude obvious ETFs/ETNs by suffixes or dashes, and very long symbols
        if "-" in sym or "/" in sym or len(sym) > 5:
            continue
        cands.append(sym)

    # De-duplicate & cap
    uniq = sorted(set(cands))
    if len(uniq) > MAX_TICKERS_TO_CHECK:
        uniq = uniq[:MAX_TICKERS_TO_CHECK]
    return uniq

def submit_buy_notional(symbol: str, notional: float):
    notional = round(max(0.0, float(notional)), 2)
    if notional <= 0:
        return
    order = api.submit_order(
        symbol=symbol,
        side="buy",
        type="market",
        time_in_force=TIME_IN_FORCE,
        notional=notional,
        client_order_id=f"swing5-buy-{symbol}-{int(time.time()*1000)}",
    )
    oid = getattr(order, "id", "") or getattr(order, "client_order_id", "")
    status = getattr(order, "status", "submitted")
    log(f"🛒 BUY {symbol} | notional=${notional:.2f} | order={oid} | status={status}")

# -----------------------------
# Main
# -----------------------------
def main():
    log(f"Start | universe cap={MAX_TICKERS_TO_CHECK} | thresholds: "
        f"RSI[{RSI_MIN},{RSI_MAX}], RelVol≥{REL_VOL_MIN}, ≤10% from 52wH | "
        f"EPS YoY≥{EPS_GROWTH_MIN}, Rev YoY≥{REV_GROWTH_MIN}, ROE≥{ROE_MIN}, D/E≤{DE_MIN}, PEG≤{PEG_MAX}*"
        " (*if available)")

    symbols = get_candidate_universe()
    log(f"Universe size after filters: {len(symbols)}")

    acct = api.get_account()
    try:
        buying_power = float(acct.buying_power)
    except Exception:
        buying_power = float(acct.cash)
    log(f"Buying power: ${buying_power:.2f}")

    buys_made = 0

    for i, sym in enumerate(symbols, 1):
        try:
            # Fetch OHLCV
            df = fetch_daily_ohlcv(sym, days=300)
            if df is None or df.empty:
                log(f"{i}/{len(symbols)} {sym}: no OHLCV; skip")
                continue

            # Liquidity & price sanity
            last_close = float(df['close'].iloc[-1])
            if last_close < MIN_PRICE:
                continue
            adv = avg_dollar_volume_30d(df) or 0.0
            if adv < MIN_AVG_DOLLAR_VOL:
                continue

            # Technicals
            if not technicals_pass(df):
                continue

            # Fundamentals
            if not fundamentals_pass(sym):
                continue

            # If it passes, buy 5% of *current* buying power
            # Refresh buying power slightly to be conservative
            try:
                buying_power = float(api.get_account().buying_power)
            except Exception:
                pass
            notional = buying_power * BUY_FRACTION_OF_BP
            if notional <= 0:
                log(f"{sym}: no buying power left; skipping")
                continue

            submit_buy_notional(sym, notional)
            buys_made += 1

            # small polite delay to avoid hammering APIs
            time.sleep(0.25)

        except Exception as e:
            log(f"{sym}: error {type(e).__name__}: {e}")

    log(f"Done. Buys placed: {buys_made}")

if __name__ == "__main__":
    main()
