import requests
import time
import csv
import os
from datetime import datetime, timezone
from collections import defaultdict, deque
import threading
from threading import Event
from http.server import BaseHTTPRequestHandler, HTTPServer

# ================== CONFIG ==================

BASE_URL = "https://api.bybit.com"
SYMBOLS = ["BTC", "ETH", "SOL", "MNT", "XRP", "DOGE"]

LOG_ALERTS = True          # ✅ CSV on

CHECK_INTERVAL = 300  # 5 min
MARKET_LOG_INTERVAL = 30 * 60  # 30 min
STABILITY_WINDOW = 3
MCI_WINDOW = 12

CREDIT_CHEAP = 0.30
CREDIT_EXPENSIVE = 0.15
SKEW_DOWN = 1.20
SKEW_UP = 0.90

NEAR_MIN = 0
NEAR_MAX = 3
MID_MIN = 7
MID_MAX = 14

HISTORY_FILE = "market_regime_history.csv"

TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

PORT = int(os.getenv("PORT", "10000"))

ALERT_COOLDOWN = {
    "CALM_COMPRESSION": 60,
    "CALM_DECAY": 30,
    "DIRECTIONAL_PRESSURE": 30,
    "PHASE_SHIFT": 60,
}

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
    print(f"HTTP health server listening on port {PORT}", flush=True)
    server.serve_forever()

# ---------- API ----------
def get_option_tickers(symbol):
    r = requests.get(
        f"{BASE_URL}/v5/market/tickers",
        params={"category": "option", "baseCoin": symbol},
        timeout=10
    )
    r.raise_for_status()
    return r.json()["result"]["list"]

# ---------- PARSER ----------
def parse_symbol(sym):
    p = sym.split("-")
    return {
        "base": p[0],
        "expiry": datetime.strptime(p[1], "%d%b%y").replace(tzinfo=timezone.utc),
        "strike": float(p[2]),
        "type": "CALL" if p[3] == "C" else "PUT",
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

def interpret_market(symbol):
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

# ---------- STATE ----------
regime_hist = {s: deque(maxlen=STABILITY_WINDOW) for s in SYMBOLS}
mci_hist = {s: deque(maxlen=MCI_WINDOW) for s in SYMBOLS}
last_phase = {s: None for s in SYMBOLS}
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
alert_ts = defaultdict(dict)

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
    market_conf = phase_confidence(market_mci, market_slope, list(market_phase_hist))
    market_p1, market_p2 = top_phase_probabilities(market_mci, market_slope)

    log_row({
        "ts_unix_ms": now_ms,
        "symbol": "MARKET",
        "regime": "MARKET",
        "mci": market_mci,
        "mci_slope": market_slope,
        "mci_phase": market_phase,
        "mci_phase_confidence": market_conf,
        "mci_phase_prob_top1": market_p1,
        "mci_phase_prob_top2": market_p2,
        "market_calm_ratio": market_state["calm_ratio"],
        "alert": "MARKET_STATE",
    })

    for symbol in SYMBOLS:
        state = last_state.get(symbol)
        if not state:
            continue

        symbol_conf = phase_confidence(
            state.get("mci"),
            state.get("slope"),
            list(phase_hist[symbol])
        )
        symbol_p1, symbol_p2 = top_phase_probabilities(
            state.get("mci"),
            state.get("slope")
        )

        log_row({
            "ts_unix_ms": now_ms,
            "symbol": symbol,
            "regime": state.get("regime"),
            "mci": state.get("mci"),
            "mci_slope": state.get("slope"),
            "mci_phase": state.get("phase"),
            "mci_phase_confidence": symbol_conf,
            "mci_phase_prob_top1": symbol_p1,
            "mci_phase_prob_top2": symbol_p2,
            "market_calm_ratio": market_state["calm_ratio"],
            "alert": "MARKET_STATE_TICKER",
        })

    print(f"MARKET SNAPSHOT logged at {datetime.now(timezone.utc)}", flush=True)
    while next_market_log_ts <= now_ms:
        next_market_log_ts += MARKET_LOG_INTERVAL * 1000

def log_row(row):
    exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(row.keys())
        w.writerow(row.values())

def log_alert(symbol, alert_type, mci, slope, phase):
    ts = now_ts_ms()
    print(
        f"ALERT {symbol} {alert_type} mci={mci} slope={slope} phase={phase} ts={ts}",
        flush=True
    )

    if LOG_ALERTS:
        log_row({
            "ts_unix_ms": ts,
            "symbol": symbol,
            "regime": "",
            "mci": mci,
            "mci_slope": slope,
            "mci_phase": phase,
            "mci_phase_confidence": None,
            "mci_phase_prob_top1": None,
            "mci_phase_prob_top2": None,
            "market_calm_ratio": None,
            "alert": alert_type,
        })

# ---------- ALERTS (MS) ----------
def maybe_alert(symbol, phase, mci, slope):
    now_ms = now_ts_ms()
    def ok(t):
        return now_ms - alert_ts[symbol].get(t, 0) > ALERT_COOLDOWN[t] * 60 * 1000

    if mci is not None and slope is not None:
        if mci > 0.7 and abs(slope) < 0.01 and ok("CALM_COMPRESSION"):
            alert_ts[symbol]["CALM_COMPRESSION"] = now_ms
            log_alert(symbol, "CALM_COMPRESSION", mci, slope, phase)

        if mci > 0.4 and slope < 0 and ok("CALM_DECAY"):
            alert_ts[symbol]["CALM_DECAY"] = now_ms
            log_alert(symbol, "CALM_DECAY", mci, slope, phase)

        if mci < 0.2 and slope < 0 and ok("DIRECTIONAL_PRESSURE"):
            alert_ts[symbol]["DIRECTIONAL_PRESSURE"] = now_ms
            log_alert(symbol, "DIRECTIONAL_PRESSURE", mci, slope, phase)


        if phase and phase != last_phase[symbol] and ok("PHASE_SHIFT"):
            alert_ts[symbol]["PHASE_SHIFT"] = now_ms
            log_alert(symbol, f"PHASE_SHIFT:{phase}", mci, slope, phase)
            last_phase[symbol] = phase


# ---------- DAILY LOG ----------
def daily_sender(stop_event):
    if not TG_TOKEN or not TG_CHAT_ID:
        return

    while not stop_event.is_set():
        now = datetime.now(timezone.utc)
        if now.hour == 11 and now.minute < 2:
            if os.path.isfile(HISTORY_FILE):
                with open(HISTORY_FILE, "rb") as f:
                    requests.post(
                        f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument",
                        data={"chat_id": TG_CHAT_ID},
                        files={"document": f},
                        timeout=20
                    )
                open(HISTORY_FILE, "w").close()
            stop_event.wait(120)
        stop_event.wait(30)

# ---------- MAIN ----------
def main():
    print("BOOT OK", flush=True)
    print("Options Market Regime Engine started", flush=True)

    threading.Thread(target=run_http_server, daemon=True).start()
    threading.Thread(target=daily_sender, args=(stop_event,), daemon=True).start()
    try:
        while True:
            cycle_start = time.time()
            print("\nCycle:", datetime.now(timezone.utc), flush=True)
        
            # ====== 1. СЧИТАЕМ ТИКЕРЫ ======
            for s in SYMBOLS:
                try:
                    r = interpret_market(s)
                    if not r:
                        continue
        
                    regime_hist[s].append(r)
                    mci_hist[s].append(mci_value(r))
        
                    mci = calc_mci(s)
                    slope = calc_slope(s)
                    phase = mci_phase(mci, slope)
                    if phase:
                        phase_hist[s].append(phase)

                    confidence = phase_confidence(mci, slope, list(phase_hist[s]))
                    prob_top1, prob_top2 = top_phase_probabilities(mci, slope)

                    last_state[s] = {
                        "regime": r,
                        "mci": mci,
                        "slope": slope,
                        "phase": phase,
                        "confidence": confidence,
                        "prob_top1": prob_top1,
                        "prob_top2": prob_top2,
                    }
        
                    maybe_alert(s, phase, mci, slope)
        
                    log_row({
                        "ts_unix_ms": now_ts_ms(),
                        "symbol": s,
                        "regime": r,
                        "mci": mci,
                        "mci_slope": slope,
                        "mci_phase": phase,
                        "mci_phase_confidence": confidence,
                        "mci_phase_prob_top1": prob_top1,
                        "mci_phase_prob_top2": prob_top2,
                        "market_calm_ratio": None,
                        "alert": None,
                    })
        
                    print(s, r, mci, slope, phase, flush=True)
        
                except Exception as e:
                    print(s, "ERROR:", e, flush=True)
        
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
        
            market_state.update({
                "mci": market_mci,
                "slope": market_slope,
                "phase": market_phase,
                "calm_ratio": market_calm_ratio,
            })

            if market_phase:
                market_phase_hist.append(market_phase)

            maybe_log_market_state()

            sleep_for = CHECK_INTERVAL - (time.time() - cycle_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        stop_event.set()
        print("Shutting down", flush=True)

if __name__ == "__main__":
    main()
