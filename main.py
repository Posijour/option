import logging
import math
import os
import threading
import time
from collections import deque
from threading import Event

import requests

from analytics import (
    calc_mci,
    calc_slope,
    mci_phase,
    mci_value,
    calc_market_olsi_slope,
    classify_olsi_slope,
    phase_confidence,
    okx_liquidity_structure_index,
    classify_mci_olsi_divergence
)
from data_bybit import interpret_bybit_market
from data_okx import get_okx_near_chain, get_okx_spot, get_okx_tickers
from http_server import run_http_server

SUPABASE_URL = "https://qcusrlmueapuqbjwuwvh.supabase.co"
SUPABASE_KEY = "sb_publishable_VsMaZGz98nm5lSQZJ-g-kQ_bUOfSO_r"

SYMBOLS = ["BTC", "ETH"]
OKX_SYMBOLS = ["BTC", "ETH"]

CHECK_INTERVAL = 300
MARKET_LOG_INTERVAL = 30 * 60
STABILITY_WINDOW = 3
MCI_WINDOW = 12
MIN_ACTIVE_OPTIONS = 10
next_bybit_market_log_ts = None

stop_event = Event()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("option-bot")


def _sanitize_for_json(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def send_to_db(event, payload):
    try:
        safe_payload = _sanitize_for_json(payload)
        requests.post(
            f"{SUPABASE_URL}/rest/v1/logs",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            },
            json={
                "ts": int(time.time() * 1000),
                "event": event,
                "symbol": safe_payload.get("symbol"),
                "data": safe_payload
            },
            timeout=5
        )
    except Exception:
        return


def now_ts_ms():
    return int(time.time() * 1000)


def get_okx_olsi(symbol, tickers=None):
    near_chain = get_okx_near_chain(symbol, tickers=tickers)
    spot = get_okx_spot(symbol)
    if spot is None:
        logger.debug("OLSI %s spot unavailable", symbol)
        return 0.0
    return okx_liquidity_structure_index(
        near_chain,
        spot,
        min_active_options=MIN_ACTIVE_OPTIONS,
        logger=logger,
        symbol=symbol,
    )


regime_hist = {s: deque(maxlen=STABILITY_WINDOW) for s in SYMBOLS}
mci_hist = {s: deque(maxlen=MCI_WINDOW) for s in SYMBOLS}
okx_olsi_hist = {s: deque(maxlen=MCI_WINDOW) for s in OKX_SYMBOLS}

phase_hist = {s: deque(maxlen=6) for s in SYMBOLS}
market_phase_hist = deque(maxlen=6)
next_market_log_ts = None

last_state = {}
market_state = {
    "mci": None,
    "slope": None,
    "phase": None,
    "calm_ratio": None,
}


def maybe_log_market_state():
    global next_market_log_ts

    now_ms = now_ts_ms()
    if next_market_log_ts is None:
        next_market_log_ts = now_ms + MARKET_LOG_INTERVAL * 1000
        return

    if now_ms < next_market_log_ts:
        return

    row = {
        "ts_unix_ms": now_ms,
        "symbol": "OKX",
    
        # OKX liquidity
        "okx_olsi_avg": market_state.get("olsi_avg"),
        "okx_olsi_slope": market_state.get("olsi_slope"),
        "okx_liquidity_regime": market_state.get("liquidity_regime"),
    
        # divergence (ОСТАВЛЯЕМ)
        "divergence": market_state.get("divergence"),
        "divergence_diff": market_state.get("divergence_diff"),
        "divergence_strength": market_state.get("divergence_strength"),
        "divergence_class": market_state.get("divergence_class"),
    }

    send_to_db("okx_market_state", row)

    logger.info(
        "OKX MARKET STATE | olsi_avg=%s olsi_slope=%s liquidity=%s div=%s(%s)",
        market_state.get("olsi_avg"),
        market_state.get("olsi_slope"),
        market_state.get("liquidity_regime"),
        market_state.get("divergence"),
        market_state.get("divergence_strength"),
    )

    while next_market_log_ts <= now_ms:
        next_market_log_ts += MARKET_LOG_INTERVAL * 1000

def maybe_log_bybit_market_state():
    global next_bybit_market_log_ts
    if not last_state:
        return
    now_ms = now_ts_ms()
    if next_bybit_market_log_ts is None:
        next_bybit_market_log_ts = now_ms + MARKET_LOG_INTERVAL * 1000
        return

    if now_ms < next_bybit_market_log_ts:
        return

    mci_vals = [
        v["mci"] for v in last_state.values()
        if v.get("mci") is not None
    ]
    slope_vals = [
        v["slope"] for v in last_state.values()
        if v.get("slope") is not None
    ]

    if not mci_vals or not slope_vals:
        return

    bybit_mci = round(sum(mci_vals) / len(mci_vals), 2)
    bybit_slope = round(sum(slope_vals) / len(slope_vals), 3)
    bybit_phase = mci_phase(bybit_mci, bybit_slope)

    row = {
        "ts_unix_ms": now_ms,
        "symbol": "BYBIT",
        "mci": bybit_mci,
        "mci_slope": bybit_slope,
        "mci_phase": bybit_phase,
    }

    send_to_db("bybit_market_state", row)

    logger.info(
        "BYBIT MARKET STATE | mci=%s slope=%s phase=%s",
        bybit_mci,
        bybit_slope,
        bybit_phase,
    )

    while next_bybit_market_log_ts <= now_ms:
        next_bybit_market_log_ts += MARKET_LOG_INTERVAL * 1000


def main():
    threading.Thread(target=run_http_server, args=(stop_event,), daemon=True).start()
    logger.info("service started: symbols=%s interval=%ss", ",".join(SYMBOLS), CHECK_INTERVAL)
    try:
        while not stop_event.is_set():
            cycle_start = time.time()
            okx_tickers_cache = None
            logger.info("cycle started")

            try:
                okx_tickers_cache = get_okx_tickers()
            except Exception as e:
                logger.warning("failed to fetch OKX tickers cache: %s", e)

            for s in SYMBOLS:
                try:

                    bybit_r = interpret_bybit_market(s)

                    if not bybit_r:
                        continue

                    if bybit_r:
                        regime_hist[s].append(bybit_r)
                        mci_hist[s].append(mci_value(bybit_r))


                    mci = calc_mci(mci_hist, s)
                    slope = calc_slope(mci_hist, s)

                    phase = mci_phase(mci, slope)
                    if phase:
                        phase_hist[s].append(phase)


                    confidence = phase_confidence(mci, slope, list(phase_hist[s]))

                    last_state[s] = {
                        "regime": bybit_r,
                        "mci": mci,
                        "slope": slope,
                        "phase": phase,
                        "confidence": confidence,
                    }


                except Exception as e:
                    logger.exception("cycle error for %s: %s", s, e)
                    continue

            for s in OKX_SYMBOLS:
                olsi = get_okx_olsi(s, tickers=okx_tickers_cache)
                okx_olsi_hist[s].append(olsi)

            mci_vals = [v["mci"] for v in last_state.values() if v["mci"] is not None]
            slope_vals = [v["slope"] for v in last_state.values() if v["slope"] is not None]

            if mci_vals and slope_vals:
                market_mci = round(sum(mci_vals) / len(mci_vals), 2)
                market_slope = round(sum(slope_vals) / len(slope_vals), 3)
                market_phase = mci_phase(market_mci, market_slope)
            else:
                market_mci = market_slope = market_phase = None

            market_olsi_vals = []
            for s in OKX_SYMBOLS:
                h = okx_olsi_hist.get(s)
                if h and len(h) >= MCI_WINDOW:
                    market_olsi_vals.append(sum(h) / len(h))
                
            market_olsi_avg = round(sum(market_olsi_vals) / len(market_olsi_vals), 4) if market_olsi_vals else None

            market_divergence = None
            market_divergence_diff = None
            market_divergence_strength = None
            market_divergence_class = None
            market_mci_norm = None
            
            if market_mci is not None and market_olsi_avg is not None:
                (
                    market_divergence,
                    market_divergence_diff,
                    market_divergence_strength,
                    market_divergence_class,
                    market_mci_norm,
                ) = classify_mci_olsi_divergence(market_mci, market_olsi_avg)

            calm_count = sum(1 for v in last_state.values() if v["regime"] == "CALM")
            market_calm_ratio = round(calm_count / len(last_state), 2) if last_state else None

            market_olsi_slope = calc_market_olsi_slope(okx_olsi_hist, OKX_SYMBOLS)
            market_olsi_regime = classify_olsi_slope(market_olsi_slope)

            market_state.update({
                "mci": market_mci,
                "slope": market_slope,
                "phase": market_phase,
                "calm_ratio": market_calm_ratio,
                "olsi_slope": market_olsi_slope,
                "olsi_avg": market_olsi_avg,
                "liquidity_regime": market_olsi_regime,
                "divergence": market_divergence,
                "divergence_diff": market_divergence_diff,
                "divergence_strength": market_divergence_strength,
                "divergence_class": market_divergence_class,
            })

            if market_phase:
                market_phase_hist.append(market_phase)

            maybe_log_market_state()
            maybe_log_bybit_market_state()
            logger.info(
                "cycle finished: market_mci=%s market_slope=%s market_phase=%s calm_ratio=%s liquidity=%s",
                market_mci, market_slope, market_phase, market_calm_ratio, market_olsi_regime
            )

            sleep_for = CHECK_INTERVAL - (time.time() - cycle_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        stop_event.set()
        logger.info("shutdown requested")


if __name__ == "__main__":
    main()
