import requests
import time
import os
import math
import logging
from datetime import datetime, timezone
from collections import deque
import threading
from threading import Event
from http.server import BaseHTTPRequestHandler, HTTPServer

SUPABASE_URL = "https://qcusrlmueapuqbjwuwvh.supabase.co"
SUPABASE_KEY = "sb_publishable_VsMaZGz98nm5lSQZJ-g-kQ_bUOfSO_r"

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


# ================== CONFIG ==================

BYBIT_BASE_URL = "https://api.bybit.com"
OKX_BASE_URL = "https://www.okx.com"

SYMBOLS = ["BTC", "ETH", "SOL", "MNT", "XRP", "DOGE"]


CHECK_INTERVAL = 300  # 5 min
MARKET_LOG_INTERVAL = 30 * 60  # 30 min
STABILITY_WINDOW = 3
MCI_WINDOW = 12

CREDIT_CHEAP = 0.30
CREDIT_EXPENSIVE = 0.15
SKEW_DOWN = 1.20
SKEW_UP = 0.90

OLSI_THRESHOLD = 0.05
MIN_ACTIVE_OPTIONS = 10

NEAR_MIN = 0
NEAR_MAX = 3
MID_MIN = 7
MID_MAX = 14

PORT = int(os.getenv("PORT", "10000"))


# ============================================

stop_event = Event()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("option-bot")

# ---------- TIME (UNIFIED, MS) ----------
def now_ts_ms():
    return int(time.time() * 1000)

# ---------- HTTP ----------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()

    def log_message(self, format, *args):
        return

def run_http_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    server.timeout = 1
    while not stop_event.is_set():
        server.handle_request()


def _safe_float(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None

# ---------- API ----------
def _request_json(url, params=None, timeout=10, retries=2, backoff_s=1.0):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(backoff_s * (attempt + 1))


def get_option_tickers(symbol):
    data = _request_json(
        f"{BYBIT_BASE_URL}/v5/market/tickers",
        params={"category": "option", "baseCoin": symbol},
        timeout=10,
    )
    return data["result"]["list"]

def get_okx_option_instruments(symbol):
    data = _request_json(
        f"{OKX_BASE_URL}/api/v5/public/instruments",
        params={
            "instType": "OPTION",
            "uly": f"{symbol}-USD"
        },
        timeout=10
    )
    return data["data"]


def get_okx_tickers():
    data = _request_json(
        f"{OKX_BASE_URL}/api/v5/market/tickers",
        params={
            "instType": "OPTION"
        },
        timeout=10
    )
    return data["data"]



# ---------- PARSER ----------
def parse_symbol(sym):
    p = sym.split("-")
    return {
        "base": p[0],
        "expiry": datetime.strptime(p[1], "%d%b%y").replace(tzinfo=timezone.utc),
        "strike": float(p[2]),
        "type": "CALL" if p[3] == "C" else "PUT",
    }

def parse_okx_symbol(inst_id):
    # пример: BTC-USD-240329-70000-C
    p = inst_id.split("-")
    return {
        "base": p[0],
        "expiry": datetime.strptime(p[2], "%y%m%d").replace(tzinfo=timezone.utc),
        "strike": float(p[3]),
        "type": "CALL" if p[4] == "C" else "PUT",
    }


# ---------- OPTION CHAIN ----------
def build_option_chain(symbol):
    out = []
    for t in get_option_tickers(symbol):
        try:
            p = parse_symbol(t["symbol"])
            out.append({
                **p,
                "bid": float(t["bid1Price"]),
                "ask": float(t["ask1Price"]),
                "iv": float(t.get("markIv") or 0)
            })
        except Exception:
            continue
    return out

def build_okx_option_chain(symbol, tickers=None):
    instruments = get_okx_option_instruments(symbol)
    tickers = tickers if tickers is not None else get_okx_tickers()

    ticker_map = {t["instId"]: t for t in tickers}
    out = []

    for inst in instruments:
        inst_id = inst["instId"]
        if inst_id not in ticker_map:
            continue

        try:
            parsed = parse_okx_symbol(inst_id)
            t = ticker_map[inst_id]

            bid = _safe_float(t.get("bidPx"))
            ask = _safe_float(t.get("askPx"))
            out.append({
                **parsed,
                "bid": bid,
                "ask": ask,
            })

        except Exception:
            continue

    return out

# ---------- SPREAD SENSOR ----------
def best_credit_spread(opts):
    opts = sorted(opts, key=lambda x: x["strike"])
    best = None
    for i in range(len(opts) - 1):
        w = abs(opts[i]["strike"] - opts[i + 1]["strike"])
        if w <= 0:
            continue
        c = opts[i]["bid"] - opts[i + 1]["ask"]
        if c <= 0:
            continue
        r = c / w
        if not best or r > best["ratio"]:
            best = {"ratio": r}
    return best

# ---------- REGIME ----------
def regime_for_expiry(opts):
    puts = [o for o in opts if o["type"] == "PUT"]
    calls = [o for o in opts if o["type"] == "CALL"]
    if not puts or not calls:
        return None

    ps = best_credit_spread(puts)
    cs = best_credit_spread(calls)

    def cls(s):
        if not s:
            return "expensive"
        if s["ratio"] >= CREDIT_CHEAP:
            return "cheap"
        if s["ratio"] >= CREDIT_EXPENSIVE:
            return "neutral"
        return "expensive"

    downside = cls(ps)
    upside = cls(cs)

    short_vol = ps and cs and ps["ratio"] >= 0.20 and cs["ratio"] >= 0.20

    piv = [p["iv"] for p in puts if p["iv"] > 0]
    civ = [c["iv"] for c in calls if c["iv"] > 0]
    skew = (sum(piv)/len(piv))/(sum(civ)/len(civ)) if piv and civ else 1.0

    if short_vol:
        return "CALM"
    if skew > SKEW_DOWN and downside != "cheap":
        return "DIRECTIONAL_DOWN"
    if skew < SKEW_UP and upside != "cheap":
        return "DIRECTIONAL_UP"
    return "UNCERTAIN"

def interpret_bybit_market(symbol):
    chain = build_option_chain(symbol)
    now = datetime.now(timezone.utc)
    near, mid = [], []
    for o in chain:
        dte = (o["expiry"] - now).days
        if NEAR_MIN <= dte <= NEAR_MAX:
            near.append(o)
        elif MID_MIN <= dte <= MID_MAX:
            mid.append(o)
    if not near or not mid:
        return None
    r1, r2 = regime_for_expiry(near), regime_for_expiry(mid)
    return r1 if r1 == r2 else "UNCERTAIN"

def get_okx_near_chain(symbol, tickers=None):
    chain = build_okx_option_chain(symbol, tickers=tickers)
    if not chain:
        return []
    now = datetime.now(timezone.utc)
    near_opts = []
    for o in chain:
        dte_hours = (o["expiry"] - now).total_seconds() / 3600
        if 0 <= dte_hours <= 72:
            near_opts.append(o)
    return near_opts


def get_okx_spot(symbol):
    data = _request_json(
        f"{OKX_BASE_URL}/api/v5/market/index-tickers",
        params={"instId": f"{symbol}-USD"},
        timeout=10
    )
    data = data["data"]
    if not data:
        return None
    return _safe_float(data[0].get("idxPx"))


def atm_weight(strike, spot):
    if spot is None or spot <= 0:
        return 0.0
    return math.exp(-abs(strike - spot) / spot)


def okx_liquidity_structure_index(chain, spot, symbol=None):
    if not chain:
        logger.debug("OLSI %s empty near chain", symbol or "N/A")
        return 0.0

    weighted_total = 0.0
    weighted_active = 0.0
    weighted_calls = 0.0
    weighted_puts = 0.0
    weighted_rel_spread_sum = 0.0
    spread_weight_sum = 0.0
    n_active = 0

    for o in chain:
        w = atm_weight(o["strike"], spot)
        weighted_total += w

        bid = o.get("bid")
        ask = o.get("ask")
        if bid is None or ask is None or bid <= 0 or ask <= bid:
            continue

        n_active += 1
        weighted_active += w

        mid = (ask + bid) / 2
        if mid > 0 and w > 0:
            weighted_rel_spread_sum += w * ((ask - bid) / mid)
            spread_weight_sum += w

        if o["type"] == "CALL":
            weighted_calls += w
        elif o["type"] == "PUT":
            weighted_puts += w

    if weighted_total <= 0:
        logger.debug("OLSI %s weighted_total<=0 spot=%s", symbol or "N/A", spot)
        return 0.0

    if n_active < MIN_ACTIVE_OPTIONS or weighted_active <= 0:
        logger.debug(
            "OLSI %s too few active: n_total=%d n_active=%d",
            symbol or "N/A", len(chain), n_active
        )
        return 0.0

    if spread_weight_sum <= 0:
        logger.debug("OLSI %s spread_weight_sum<=0", symbol or "N/A")
        return 0.0

    asr = weighted_active / weighted_total
    avg_rel_spread = weighted_rel_spread_sum / spread_weight_sum
    nss = 1 / (1 + avg_rel_spread)

    cp_sum = weighted_calls + weighted_puts
    if cp_sum <= 0:
        logger.debug("OLSI %s no call/put structure", symbol or "N/A")
        return 0.0

    sb = min(weighted_calls, weighted_puts) / cp_sum

    return round(asr * nss * sb, 4)


def get_okx_olsi(symbol, tickers=None):
    near_chain = get_okx_near_chain(symbol, tickers=tickers)
    spot = get_okx_spot(symbol)
    if spot is None:
        logger.debug("OLSI %s spot unavailable", symbol)
        return 0.0
    return okx_liquidity_structure_index(near_chain, spot, symbol=symbol)

# ---------- STATE ----------
regime_hist = {s: deque(maxlen=STABILITY_WINDOW) for s in SYMBOLS}
mci_hist = {s: deque(maxlen=MCI_WINDOW) for s in SYMBOLS}
OKX_SYMBOLS = ["BTC", "ETH"]

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

def mci_value(reg):
    return 1 if reg == "CALM" else -1 if reg.startswith("DIRECTIONAL") else 0

def calc_mci(symbol):
    h = mci_hist[symbol]
    return round(sum(h)/len(h), 2) if len(h) == MCI_WINDOW else None

def calc_slope(symbol):
    h = list(mci_hist[symbol])
    if len(h) < MCI_WINDOW:
        return None
    half = MCI_WINDOW // 2
    return round(sum(h[half:]) / half - sum(h[:half]) / half, 3)

def calc_okx_olsi(symbol):
    h = okx_olsi_hist[symbol]
    if len(h) < MCI_WINDOW:
        return None
    return round(sum(h)/len(h), 4)

def calc_market_olsi_slope():
    slopes = []

    for s in OKX_SYMBOLS:
        val = calc_okx_olsi_slope(s)
        if val is not None:
            slopes.append(val)

    if not slopes:
        return None

    return round(sum(slopes) / len(slopes), 4)

def classify_liquidity(slope):
    if slope is None:
        return None

    if slope > 0.15:
        return "LIQUIDITY_EXPANDING_STRONG"
    if slope > 0.05:
        return "LIQUIDITY_EXPANDING"
    if slope < -0.15:
        return "LIQUIDITY_CRUSH_STRONG"
    if slope < -0.05:
        return "LIQUIDITY_CRUSH"

    return "LIQUIDITY_NEUTRAL"

def calc_okx_olsi_slope(symbol):
    h = list(okx_olsi_hist[symbol])
    if len(h) < MCI_WINDOW:
        return None

    first = h[0]
    last = h[-1]

    if first == 0:
        return None

    return round((last - first) / first, 4)


def norm_slope(x, cap=0.2):
    if x is None:
        return None
    return max(-1.0, min(1.0, x / cap))

def classify_market_liquidity(slope):
    if slope is None:
        return None

    if slope > OLSI_THRESHOLD:
        return "LIQUIDITY_EXPANSION"

    if slope < -0.01:
        return "LIQUIDITY_CRUSH"

    return "LIQUIDITY_STABLE"

def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 3)

def calc_market_mci():
    return mean([calc_mci(s) for s in SYMBOLS])

def calc_market_slope():
    return mean([calc_slope(s) for s in SYMBOLS])

def calc_market_phase():
    return mci_phase(
        calc_market_mci(),
        calc_market_slope()
    )

PHASE_CENTERS = {
    "ACCUMULATING_CALM": (0.6,  0.05),
    "STABLE_CALM":       (0.7,  0.00),
    "OVERCOMPRESSED":   (0.9,  0.00),
    "BREAKING_COMPRESSION": (0.9, -0.08),
    "RELEASING":        (0.6, -0.05),
}

def probabilistic_phase(mci, slope):
    if mci is None or slope is None:
        return {}

    scores = {}
    for phase, (mci_c, slope_c) in PHASE_CENTERS.items():
        dist = abs(mci - mci_c) + abs(slope - slope_c)
        scores[phase] = 1 / (1 + dist)

    total = sum(scores.values())
    return {k: round(v / total, 2) for k, v in scores.items()}

def mci_phase(mci, slope):
    if mci is None or slope is None:
        return None

    if mci >= 0.75 and abs(slope) <= 0.02:
        return "OVERCOMPRESSED"

    if mci >= 0.75 and slope < -0.05:
        return "BREAKING_COMPRESSION"

    if mci >= 0.40 and slope < -0.02:
        return "RELEASING"

    if 0.50 <= mci <= 0.75 and slope > 0.02:
        return "ACCUMULATING_CALM"

    if 0.60 <= mci <= 0.80 and abs(slope) <= 0.02:
        return "STABLE_CALM"

    return None

def phase_confidence(mci, slope, history):
    if mci is None or slope is None:
        return None

    mci_depth = min(1.0, max(0.0, (mci - 0.4) / 0.6))
    slope_strength = min(1.0, abs(slope) / 0.1)

    if history:
        stability = history.count(history[-1]) / len(history)
    else:
        stability = 0.5

    confidence = (
        0.4 * mci_depth +
        0.3 * slope_strength +
        0.3 * stability
    )

    return round(min(confidence, 1.0), 2)


def top_phase_probabilities(mci, slope):
    probs = probabilistic_phase(mci, slope)
    top = sorted(probs.items(), key=lambda x: -x[1])[:2]
    if not top:
        return None, None
    first = f"{top[0][0]}:{top[0][1]}"
    second = f"{top[1][0]}:{top[1][1]}" if len(top) > 1 else None
    return first, second


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
    
        # MCI
        "mci": market_mci,
        "mci_slope": market_slope,
        "mci_phase": market_phase,
    
        # MITI
        "liquidity_regime": market_state.get("liquidity_regime"),
    
        # structure
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

# ---------- MAIN ----------
def main():
    threading.Thread(target=run_http_server, daemon=True).start()
    try:
        while not stop_event.is_set():
            cycle_start = time.time()
            okx_tickers_cache = None

            try:
                okx_tickers_cache = get_okx_tickers()
            except Exception as e:
                logger.warning("failed to fetch OKX tickers cache: %s", e)
        
            # ====== 1. СЧИТАЕМ ТИКЕРЫ ======
            for s in SYMBOLS:
                try:
                    divergence = None
                    divergence_diff = None
                    
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

                    mci = calc_mci(s)
                    slope = calc_slope(s)
                    okx_olsi_avg = None
                    okx_olsi_slope = None

                    if s in OKX_SYMBOLS:
                        okx_olsi_avg = calc_okx_olsi(s)
                        okx_olsi_slope = calc_okx_olsi_slope(s)

                    if s in OKX_SYMBOLS and okx_olsi_avg is not None:
                        send_to_db("okx_olsi", {
                            "ts_unix_ms": now_ts_ms(),
                            "symbol": s,
                            "okx_olsi": okx_olsi,
                            "okx_olsi_avg": okx_olsi_avg,
                            "okx_olsi_slope": okx_olsi_slope,
                        })

                    # ===== DIVERGENCE ENGINE (Slope-based) =====
                    if s in OKX_SYMBOLS and slope is not None and okx_olsi_slope is not None:
                        divergence_diff = round(norm_slope(slope) - norm_slope(okx_olsi_slope), 4)

                        if abs(divergence_diff) >= 1.0:
                            divergence = "EXTREME"
                        elif abs(divergence_diff) >= 0.6:
                            divergence = "STRONG"
                        elif abs(divergence_diff) >= 0.3:
                            divergence = "MODERATE"
                        else:
                            divergence = "NONE"

                        send_to_db("okx_divergence", {
                            "ts_unix_ms": now_ts_ms(),
                            "symbol": s,
                            "mci_slope": slope,
                            "okx_olsi_slope": okx_olsi_slope,
                            "divergence": divergence,
                            "divergence_diff": divergence_diff,
                        })

                    # ===== PHASE CALCULATION =====
                    phase = mci_phase(mci, slope)
                    if phase:
                        phase_hist[s].append(phase)

                    # ===== OLSI PHASE (OKX liquidity regime) =====
                    liquidity_phase = None

                    if okx_olsi_slope is not None:
                        if okx_olsi_slope > OLSI_THRESHOLD:
                            liquidity_phase = "LIQUIDITY_EXPANDING"
                        elif okx_olsi_slope < -OLSI_THRESHOLD:
                            liquidity_phase = "LIQUIDITY_CRUSH"
                        else:
                            liquidity_phase = "LIQUIDITY_FLAT"

                    # ===== PHASE DIVERGENCE (Structure vs Liquidity) =====
                    phase_divergence = None

                    if phase and liquidity_phase:

                        if phase in ["OVERCOMPRESSED", "ACCUMULATING_CALM"] and liquidity_phase == "LIQUIDITY_EXPANDING":
                            phase_divergence = "PRE_BREAK_TENSION"

                        elif phase == "RELEASING" and liquidity_phase == "LIQUIDITY_CRUSH":
                            phase_divergence = "POST_MOVE_DECAY"

                        elif phase in ["OVERCOMPRESSED", "STABLE_CALM"] and liquidity_phase == "LIQUIDITY_CRUSH":
                            phase_divergence = "FALSE_COMPRESSION"

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
                        "okx_olsi": okx_olsi,
                        "okx_olsi_avg": okx_olsi_avg,
                        "okx_olsi_slope": okx_olsi_slope,
                        "divergence": divergence,
                        "divergence_diff": divergence_diff,
                        "liquidity_phase": liquidity_phase,
                        "phase_divergence": phase_divergence,
                        "market_calm_ratio": None,
                        "liquidity_regime": market_state.get("liquidity_regime"),
                        "alert": None,
                    }

                    send_to_db("options_ticker_cycle", ticker_payload)

                except Exception as e:
                    logger.exception("cycle error for %s: %s", s, e)
                    continue
            
            # ====== 2. СЧИТАЕМ РЫНОК ЦЕЛИКОМ ======
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
        
            market_olsi_slope = calc_market_olsi_slope()
            market_olsi_regime = classify_market_liquidity(market_olsi_slope)
            
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

            sleep_for = CHECK_INTERVAL - (time.time() - cycle_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        stop_event.set()
        logger.info("shutdown requested")

if __name__ == "__main__":
    main()
