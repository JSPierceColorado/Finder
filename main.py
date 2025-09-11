#!/usr/bin/env python3
"""
Alpaca + Polygon Screener Buyer (5% swing entries)

Enhancements:
  - UTC fix: use datetime.now(timezone.utc)
  - Detailed logging for each skip reason (price, liquidity, technicals, fundamentals)
  - End-of-run summary histogram of skip reasons
  - Periodic progress logs (LOG_PROGRESS_EVERY, default 25)
  - Optional verbose logging (DEBUG_VERBOSE=1) to print every symbol decision
"""

import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone  # UTC fix here
from typing import Dict, Any, List, Optional, Tuple

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
    mode = get_env_str("ALPACA_MODE", "paper").lower()
    APCA_API_BASE_URL = "https://paper-api.alpaca.markets" if mode == "paper" else "https://api.alpaca.markets"

POLYGON_API_KEY   = get_env_str("POLYGON_API_KEY")
POLYGON_BASE      = "https://api.polygon.io"

# Logging controls
DEBUG_VERBOSE      = get_env_str("DEBUG_VERBOSE", "0") in ("1", "true", "True", "YES", "yes")
LOG_PROGRESS_EVERY = int(get_env_str("LOG_PROGRESS_EVERY", "25"))

def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[buy-screener] {now} | {msg}", flush=True)

def vlog(msg: str):
    if DEBUG_VERBOSE:
        log(msg)

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
MAX_TICKERS_TO_CHECK   = int(get_env_str("MAX_TICKERS_TO_CHECK", "200"))
MIN_PRICE              = float(get_env_str("MIN_PRICE", "3.0"))
MIN_AVG_DOLLAR_VOL     = float(get_env_str("MIN_AVG_DOLLAR_VOL", "5_000_000"))

RSI_MIN, RSI_MAX       = 45.0, 60.0
REL_VOL_MIN            = 1.5        # yesterday vs 30D avg
WITHIN_HI_PCT          = 0.10       # within 10% of 52w high

EPS_GROWTH_MIN         = 0.05
REV_GROWTH_MIN         = 0.05
ROE_MIN                = 0.15
DE_MIN, PEG_MAX        = 1.0, 2.0

BUY_FRACTION_OF_BP     = float(get_env_str("BUY_FRACTION_OF_BP", "0.05"))
TIME_IN_FORCE          = get_env_str("TIME_IN_FORCE", "day")

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
    # UTC FIX: was datetime.utcnow().date(); now timezone-aware UTC
    end = datetime.now(timezone.utc).date()
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

    out["eps_yoy"] = (f.get("epsGrowth", {}) or {}).get("value")
    out["rev_yoy"] = (f.get("revenueGrowth", {}) or {}).get("value")
    out["roe"]     = (f.get("roe", {}) or {}).get("value")
    out["de"]      = (f.get("debtToEquity", {}) or {}).get("value")
    out["peg"]     = (f.get("pegRatio", {}) or {}).get("value")

    return out

# -----------------------------
# Screen logic (with reasons)
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

def technicals_pass(df: pd.DataFrame) -> Tuple[bool, str]:
    closes = df["close"]
    if closes.isna().any() or len(closes) < 50:
        return False, "technical:data_insufficient(<50 bars or NaNs)"

    sma20 = sma(closes, 20).iloc[-1]
    sma50 = sma(closes, 50).iloc[-1]
    close = closes.iloc[-1]
    if not (close > sma20 and close > sma50):
        return False, f"technical:below_sma(close={close:.2f},sma20={sma20:.2f},sma50={sma50:.2f})"

    rsi_now = rsi(closes, 14).iloc[-1]
    if not (RSI_MIN <= rsi_now <= RSI_MAX):
        return False, f"technical:rsi_out_of_band(rsi14={rsi_now:.2f},range={RSI_MIN}-{RSI_MAX})"

    macd_line, macd_sig, _ = macd(closes)
    if not (macd_line.iloc[-1] > macd_sig.iloc[-1]):
        return False, f"technical:macd_not_bullish(line={macd_line.iloc[-1]:.4f},sig={macd_sig.iloc[-1]:.4f})"

    if not within_pct_of_high(closes.tail(252), WITHIN_HI_PCT):
        return False, "technical:not_within_10pct_52w_high"

    rv = rel_volume_yday_vs_30d(df)
    if rv is None or rv < REL_VOL_MIN:
        return False, f"technical:relvol_low(rv={0.0 if rv is None else rv:.2f},min={REL_VOL_MIN})"

    return True, "ok"

def fundamentals_pass(ticker: str) -> Tuple[bool, str]:
    f = fetch_fundamentals_latest(ticker)
    eps = f["eps_yoy"]; rev = f["rev_yoy"]; roe = f["roe"]; de = f["de"]; peg = f["peg"]

    # Missing data reasons called out explicitly
    if eps is None: return False, "fundamental:missing_eps_yoy"
    if rev is None: return False, "fundamental:missing_rev_yoy"
    if roe is None: return False, "fundamental:missing_roe"
    if de  is None: return False, "fundamental:missing_de"

    if eps < EPS_GROWTH_MIN: return False, f"fundamental:eps_yoy_low({eps:.3f}<{EPS_GROWTH_MIN})"
    if rev < REV_GROWTH_MIN: return False, f"fundamental:rev_yoy_low({rev:.3f}<{REV_GROWTH_MIN})"
    if roe < ROE_MIN:        return False, f"fundamental:roe_low({roe:.3f}<{ROE_MIN})"
    if de > DE_MIN:          return False, f"fundamental:de_high({de:.3f}>{DE_MIN})"
    if peg is not None and peg > PEG_MAX: return False, f"fundamental:peg_high({peg:.3f}>{PEG_MAX})"

    return True, "ok"

# -----------------------------
# Universe & Buy
# -----------------------------
def get_candidate_universe() -> List[str]:
    assets = api.list_assets(status="active")
    cands = []
    for a in assets:
        if getattr(a, "class", None) != "us_equity":
            continue
        if not a.tradable:
            continue
        sym = a.symbol.upper()
        if "-" in sym or "/" in sym or len(sym) > 5:
            continue
        cands.append(sym)

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
    session_start = time.time()

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
    reasons = Counter()
    checked = 0

    for i, sym in enumerate(symbols, 1):
        t0 = time.time()
        try:
            df = fetch_daily_ohlcv(sym, days=300)
            if df is None or df.empty:
                reasons["data:no_ohlcv"] += 1
                vlog(f"{i}/{len(symbols)} {sym}: no OHLCV; skip")
                continue

            last_close = float(df['close'].iloc[-1])
            if last_close < MIN_PRICE:
                reasons["pre:price_low"] += 1
                vlog(f"{i}/{len(symbols)} {sym}: price {last_close:.2f} < {MIN_PRICE}; skip")
                continue

            adv = avg_dollar_volume_30d(df) or 0.0
            if adv < MIN_AVG_DOLLAR_VOL:
                reasons["pre:liquidity_low"] += 1
                vlog(f"{i}/{len(symbols)} {sym}: 30D $avg vol {adv:.0f} < {MIN_AVG_DOLLAR_VOL}; skip")
                continue

            ok_t, why_t = technicals_pass(df)
            if not ok_t:
                reasons[why_t] += 1
                vlog(f"{i}/{len(symbols)} {sym}: {why_t}; skip")
                continue

            ok_f, why_f = fundamentals_pass(sym)
            if not ok_f:
                reasons[why_f] += 1
                vlog(f"{i}/{len(symbols)} {sym}: {why_f}; skip")
                continue

            # Refresh buying power conservatively
            try:
                buying_power = float(api.get_account().buying_power)
            except Exception:
                pass

            notional = buying_power * BUY_FRACTION_OF_BP
            if notional <= 0:
                reasons["buy:no_buying_power"] += 1
                log(f"{sym}: no buying power left; skipping")
                continue

            submit_buy_notional(sym, notional)
            buys_made += 1

            time.sleep(0.25)  # gentle rate limiting

        except Exception as e:
            reasons[f"error:{type(e).__name__}"] += 1
            log(f"{sym}: error {type(e).__name__}: {e}")
        finally:
            checked += 1
            if LOG_PROGRESS_EVERY > 0 and (i % LOG_PROGRESS_EVERY == 0):
                elapsed = time.time() - session_start
                log(f"Progress {i}/{len(symbols)} | buys={buys_made} | elapsed={elapsed:.1f}s")

            # Per-symbol timing (verbose only)
            vlog(f"{sym}: processed in {(time.time()-t0):.3f}s")

    elapsed = time.time() - session_start
    log(f"Done. Symbols checked: {checked} | Buys placed: {buys_made} | Elapsed: {elapsed:.1f}s")

    # Summary of reasons
    if reasons:
        log("Summary of skip reasons (top 15):")
        for reason, cnt in reasons.most_common(15):
            log(f"  - {reason}: {cnt}")
    else:
        log("No skip reasons recorded (unexpected).")

if __name__ == "__main__":
    main()
