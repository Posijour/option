import logging
import math
import os
import threading
import time
from collections import deque
from threading import Event

import requests

from analytics import (
    OLSI_THRESHOLD,
    calc_mci,
    calc_slope,
    classify_mci_olsi_divergence,
    mci_phase,
    mci_value,
    calc_okx_olsi_mean,
    calc_okx_olsi_slope,
    calc_market_olsi_slope,
    classify_olsi_slope,
    okx_liquidity_structure_index,
    phase_confidence,
    top_phase_probabilities,
)
from data_bybit import interpret_bybit_market
from data_okx import get_okx_near_chain, get_okx_spot, get_okx_tickers
from http_server import run_http_server

SUPABASE_URL = "https://qcusrlmueapuqbjwuwvh.supabase.co"
SUPABASE_KEY = "sb_publishable_VsMaZGz98nm5lSQZJ-g-kQ_bUOfSO_r"

SYMBOLS = ["BTC", "ETH", "SOL", "MNT", "XRP", "DOGE"]
OKX_SYMBOLS = ["BTC", "ETH"]

CHECK_INTERVAL = 300
MARKET_LOG_INTERVAL = 30 * 60
STABILITY_WINDOW = 3
MCI_WINDOW = 12
MIN_ACTIVE_OPTIONS = 10

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

    market_mci = market_state["mci"]
    market_slope = market_state["slope"]
    market_phase = market_state["phase"]

    row = {
        "ts_unix_ms": now_ms,
        "symbol": "MARKET",
        "mci": market_mci,
        "mci_slope": market_slope,
        "mci_phase": market_phase,
        "liquidity_regime": market_state.get("liquidity_regime"),
        "market_calm_ratio": market_state["calm_ratio"],
    }
    send_to_db("options_market_state", row)

    for symbol in SYMBOLS:
        state = last_state.get(symbol)
        if not state:
            continue

        row = {
            "ts_unix_ms": now_ms,
            "symbol": symbol,
            "regime": state.get("regime"),
            "mci": state.get("mci"),
            "mci_slope": state.get("slope"),
            "mci_phase": state.get("phase"),
            "market_calm_ratio": market_state["calm_ratio"],
        }
        send_to_db("options_ticker_state", row)

    while next_market_log_ts <= now_ms:
        next_market_log_ts += MARKET_LOG_INTERVAL * 1000


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
                    divergence = None
                    divergence_diff = None
                    divergence_strength = None
                    divergence_strength_class = None
                    mci_norm = None

                    bybit_r = interpret_bybit_market(s)

                    okx_olsi = None
                    if s in OKX_SYMBOLS:
                        okx_olsi = get_okx_olsi(s, tickers=okx_tickers_cache)

                    if not bybit_r and okx_olsi is None:
                        continue

                    if bybit_r:
                        regime_hist[s].append(bybit_r)
                        mci_hist[s].append(mci_value(bybit_r))

                    if s in OKX_SYMBOLS:
                        okx_olsi_hist[s].append(okx_olsi)

                    mci = calc_mci(mci_hist, s)
                    slope = calc_slope(mci_hist, s)
                    okx_olsi_avg = None
                    okx_olsi_slope = None

                    phase = mci_phase(mci, slope)
                    if phase:
                        phase_hist[s].append(phase)

                    liquidity_phase = classify_olsi_slope(okx_olsi_slope)

                    phase_divergence = None
                    if phase and liquidity_phase:
                        if phase in ["OVERCOMPRESSED", "ACCUMULATING_CALM"] and liquidity_phase == "LIQUIDITY_EXPANDING":
                            phase_divergence = "PRE_BREAK_TENSION"
                        elif phase == "RELEASING" and liquidity_phase == "LIQUIDITY_CRUSH":
                            phase_divergence = "POST_MOVE_DECAY"
                        elif phase in ["OVERCOMPRESSED", "STABLE_CALM"] and liquidity_phase == "LIQUIDITY_CRUSH":
                            phase_divergence = "FALSE_COMPRESSION"

                    if s in OKX_SYMBOLS:
                        okx_olsi_avg = calc_okx_olsi_mean(okx_olsi_hist, s)
                        okx_olsi_slope = calc_okx_olsi_slope(okx_olsi_hist, s)
                        divergence, divergence_diff, divergence_strength, divergence_strength_class, mci_norm = classify_mci_olsi_divergence(mci, okx_olsi_avg)

                    if s in OKX_SYMBOLS and okx_olsi_avg is not None:
                        send_to_db("okx_olsi", {
                            "ts_unix_ms": now_ts_ms(),
                            "symbol": s,
                            "okx_olsi": okx_olsi,
                            "okx_olsi_avg": okx_olsi_avg,
                            "okx_olsi_slope": okx_olsi_slope,
                            "divergence": divergence,
                            "divergence_diff": divergence_diff,
                            "divergence_strength": divergence_strength,
                            "divergence_strength_class": divergence_strength_class,
                            "liquidity_phase": liquidity_phase,
                            "phase_divergence": phase_divergence,
                            "mci_norm": mci_norm,
                        })

                    confidence = phase_confidence(mci, slope, list(phase_hist[s]))
                    prob_top1, prob_top2 = top_phase_probabilities(mci, slope)

                    last_state[s] = {
                        "regime": bybit_r,
                        "mci": mci,
                        "slope": slope,
                        "phase": phase,
                        "confidence": confidence,
                        "prob_top1": prob_top1,
                        "prob_top2": prob_top2,
                    }

                    ticker_payload = {
                        "ts_unix_ms": now_ts_ms(),
                        "symbol": s,
                        "regime": bybit_r,
                        "mci": mci,
                        "mci_slope": slope,
                        "mci_phase": phase,
                        "mci_phase_confidence": confidence,
                        "mci_phase_prob_top1": prob_top1,
                        "mci_phase_prob_top2": prob_top2,
                        "market_calm_ratio": None,
                        "liquidity_regime": market_state.get("liquidity_regime"),
                        "alert": None,
                    }

                    send_to_db("options_ticker_cycle", ticker_payload)
                    logger.info(
                        "ticker processed: symbol=%s bybit=%s okx_olsi=%s mci=%s slope=%s phase=%s",
                        s, bybit_r, okx_olsi, mci, slope, phase
                    )

                except Exception as e:
                    logger.exception("cycle error for %s: %s", s, e)
                    continue

            mci_vals = [v["mci"] for v in last_state.values() if v["mci"] is not None]
            slope_vals = [v["slope"] for v in last_state.values() if v["slope"] is not None]

            if mci_vals and slope_vals:
                market_mci = round(sum(mci_vals) / len(mci_vals), 2)
                market_slope = round(sum(slope_vals) / len(slope_vals), 3)
                market_phase = mci_phase(market_mci, market_slope)
            else:
                market_mci = market_slope = market_phase = None

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
                "liquidity_regime": market_olsi_regime,
            })

            if market_phase:
                market_phase_hist.append(market_phase)

            maybe_log_market_state()
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
