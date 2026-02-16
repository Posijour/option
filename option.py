import requests
import time
import os
import math
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

IV_THRESHOLD = 0.02

NEAR_MIN = 0
NEAR_MAX = 3
MID_MIN = 7
MID_MAX = 14

PORT = int(os.getenv("PORT", "10000"))


# ============================================

stop_event = Event()

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
    server.serve_forever()

# ---------- API ----------
def get_option_tickers(symbol):
    r = requests.get(
        f"{BYBIT_BASE_URL}/v5/market/tickers",
        params={"category": "option", "baseCoin": symbol},
        timeout=10
    )
    r.raise_for_status()
    return r.json()["result"]["list"]

def get_okx_option_instruments(symbol):
    r = requests.get(
        f"{OKX_BASE_URL}/api/v5/public/instruments",
        params={
            "instType": "OPTION",
            "uly": f"{symbol}-USD"
        },
        timeout=10
    )
    r.raise_for_status()
    return r.json()["data"]


def get_okx_tickers():
    r = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/tickers",
        params={
            "instType": "OPTION"
        },
        timeout=10
    )
    r.raise_for_status()
    return r.json()["data"]



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

def build_okx_option_chain(symbol):
    instruments = get_okx_option_instruments(symbol)
    tickers = get_okx_tickers()

    ticker_map = {t["instId"]: t for t in tickers}

    out = []


    for inst in instruments:
        inst_id = inst["instId"]

        if inst_id not in ticker_map:
            continue

        try:
            parsed = parse_okx_symbol(inst_id)
            t = ticker_map[inst_id]

            bid = float(t.get("bidPx") or 0)

            if bid > 0:
                out.append({
                    **parsed,
                    "bid": bid,
                    "ask": bid,
                    "iv": bid,
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

def get_okx_spot(symbol):
    r = requests.get(
        f"{OKX_BASE_URL}/api/v5/market/index-tickers",
        params={"instId": f"{symbol}-USD"},
        timeout=10
    )
    r.raise_for_status()
    data = r.json()["data"]
    if not data:
        return None
    return float(data[0]["idxPx"])

def get_okx_atm_iv(symbol):
    chain = build_okx_option_chain(symbol)
    if not chain:
        return None

    spot = get_okx_spot(symbol)
    if not spot:
        return None

    # оставляем только опционы near expiry
    now = datetime.now(timezone.utc)
    near_opts = []

    for o in chain:
        dte = (o["expiry"] - now).days
        if NEAR_MIN <= dte <= NEAR_MAX:
            near_opts.append(o)

    if not near_opts:
        return None

    # ищем страйк максимально близкий к spot
    atm = min(near_opts, key=lambda x: abs(x["strike"] - spot))

    return atm["iv"] if atm["iv"] > 0 else None


def interpret_okx_market(symbol):
    iv = get_okx_atm_iv(symbol)
    return iv

# ---------- STATE ----------
regime_hist = {s: deque(maxlen=STABILITY_WINDOW) for s in SYMBOLS}
mci_hist = {s: deque(maxlen=MCI_WINDOW) for s in SYMBOLS}
OKX_SYMBOLS = ["BTC", "ETH"]

okx_iv_hist = {s: deque(maxlen=MCI_WINDOW) for s in OKX_SYMBOLS}

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

def calc_okx_iv(symbol):
    h = okx_iv_hist[symbol]
    if len(h) < MCI_WINDOW:
        return None
    return round(sum(h)/len(h), 4)

def calc_market_iv_slope():
    slopes = []

    for s in OKX_SYMBOLS:
        val = calc_okx_iv_slope(s)
        if val is not None:
            slopes.append(val)

    if not slopes:
        return None

    return round(sum(slopes) / len(slopes), 4)

def classify_miti(slope):
    if slope is None:
        return None

    if slope > 0.15:
        return "IV_EXPANDING_STRONG"
    if slope > 0.05:
        return "IV_EXPANDING"
    if slope < -0.15:
        return "IV_CRUSH_STRONG"
    if slope < -0.05:
        return "IV_CRUSH"

    return "IV_NEUTRAL"

def calc_okx_iv_slope(symbol):
    h = list(okx_iv_hist[symbol])
    if len(h) < MCI_WINDOW:
        return None

    first = h[0]
    last = h[-1]

    if first == 0:
        return None

    return round((last - first) / first, 4)

def classify_market_iv(slope):
    if slope is None:
        return None

    if slope > IV_THRESHOLD:
        return "IV_EXPANSION"

    if slope < -0.01:
        return "IV_CRUSH"

    return "IV_STABLE"

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
        "miti_regime": market_state.get("iv_regime"),
    
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
        while True:
            cycle_start = time.time()
        
            # ====== 1. СЧИТАЕМ ТИКЕРЫ ======
            for s in SYMBOLS:
                try:
                    divergence = None
                    divergence_diff = None
                    
                    bybit_r = interpret_bybit_market(s)

                    okx_r = None
                    if s in OKX_SYMBOLS:
                        okx_r = interpret_okx_market(s)

                    if not bybit_r and not okx_r:
                        continue

                    if bybit_r:
                        regime_hist[s].append(bybit_r)
                        mci_hist[s].append(mci_value(bybit_r))

                    if s in OKX_SYMBOLS:
                        okx_iv = okx_r
                        if okx_iv:
                            okx_iv_hist[s].append(okx_iv)

                    mci = calc_mci(s)
                    slope = calc_slope(s)
                    okx_iv_avg = None
                    okx_iv_slope = None

                    if s in OKX_SYMBOLS:
                        okx_iv_avg = calc_okx_iv(s)
                        okx_iv_slope = calc_okx_iv_slope(s)
                    
                    if s in OKX_SYMBOLS and okx_iv_avg is not None:
                        send_to_db("okx_atm_iv", {
                            "ts_unix_ms": now_ts_ms(),
                            "symbol": s,
                            "okx_iv_avg": okx_iv_avg,
                            "okx_iv_slope": okx_iv_slope,
                        })

                                        # ===== DIVERGENCE ENGINE (Slope-based) =====
                    if s in OKX_SYMBOLS and slope is not None and okx_iv_slope is not None:
                        divergence_diff = round(slope - okx_iv_slope, 4)

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
                            "okx_iv_slope": okx_iv_slope,
                            "divergence": divergence,
                            "divergence_diff": divergence_diff,
                        })


                    # ===== PHASE CALCULATION =====
                    phase = mci_phase(mci, slope)
                    if phase:
                        phase_hist[s].append(phase)

                    # ===== IV PHASE (OKX ATM IV regime) =====
                    iv_phase = None

                    if okx_iv_slope is not None:
                        if okx_iv_slope > IV_THRESHOLD:
                            iv_phase = "IV_EXPANDING"
                        elif okx_iv_slope < -IV_THRESHOLD:
                            iv_phase = "IV_CRUSH"
                        else:
                            iv_phase = "IV_FLAT"

                    # ===== PHASE DIVERGENCE (Structure vs Volatility) =====
                    phase_divergence = None

                    if phase and iv_phase:

                        if phase in ["OVERCOMPRESSED", "ACCUMULATING_CALM"] and iv_phase == "IV_EXPANDING":
                            phase_divergence = "PRE_BREAK_TENSION"

                        elif phase == "RELEASING" and iv_phase == "IV_CRUSH":
                            phase_divergence = "POST_MOVE_DECAY"

                        elif phase in ["OVERCOMPRESSED", "STABLE_CALM"] and iv_phase == "IV_CRUSH":
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
                        "okx_iv_avg": okx_iv_avg,
                        "okx_iv_slope": okx_iv_slope,
                        "divergence": divergence,
                        "divergence_diff": divergence_diff,
                        "iv_phase": iv_phase,
                        "phase_divergence": phase_divergence,
                        "market_calm_ratio": None,
                        "miti_regime": market_state.get("iv_regime"),
                        "alert": None,
                    }

                    send_to_db("options_ticker_cycle", ticker_payload)

                except Exception:
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
        
            market_iv_slope = calc_market_iv_slope()
            market_iv_regime = classify_market_iv(market_iv_slope)
            
            market_state.update({
                "mci": market_mci,
                "slope": market_slope,
                "phase": market_phase,
                "calm_ratio": market_calm_ratio,
                "iv_slope": market_iv_slope,
                "iv_regime": market_iv_regime,
            })

            if market_phase:
                market_phase_hist.append(market_phase)

            maybe_log_market_state()

            sleep_for = CHECK_INTERVAL - (time.time() - cycle_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        stop_event.set()

if __name__ == "__main__":
    main()
