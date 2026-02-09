import requests
import time
import csv
import os
from datetime import datetime, timezone
from collections import defaultdict, deque
import threading
from threading import Event
import requests as tg_requests
from http.server import BaseHTTPRequestHandler, HTTPServer

# ================== CONFIG ==================

BASE_URL = "https://api.bybit.com"
SYMBOLS = ["BTC", "ETH", "SOL", "MNT", "XRP", "DOGE"]

SEND_ALERTS_TO_TG = False   # ⛔ Telegram off
LOG_ALERTS = True          # ✅ CSV on

CHECK_INTERVAL = 300  # 5 min
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
LAST_UPDATE_ID = 0

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

# ---------- TELEGRAM ----------
def tg_send(text):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        tg_requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text},
            timeout=5
        )
    except Exception:
        pass

def tg_poll():
    global LAST_UPDATE_ID
    if not TG_TOKEN:
        return

    try:
        r = tg_requests.get(
            f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
            params={"offset": LAST_UPDATE_ID + 1, "timeout": 10},
            timeout=15
        )
        data = r.json()
        for upd in data.get("result", []):
            LAST_UPDATE_ID = upd["update_id"]
            msg = upd.get("message", {})
            text = msg.get("text", "")
            chat_id = msg.get("chat", {}).get("id")

            if text == "/status" and chat_id == int(TG_CHAT_ID):
                tg_send(build_status_text())

    except Exception:
        pass


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

def mci_phase(mci, slope):
    if mci is None or slope is None:
        return None
    if 0.50 <= mci <= 0.75 and slope > 0.02:
        return "ACCUMULATING_CALM"
    if 0.60 <= mci <= 0.80 and -0.02 <= slope <= 0.02:
        return "STABLE_CALM"
    if mci >= 0.75 and abs(slope) <= 0.02:
        return "OVERCOMPRESSED"
    if 0.40 <= mci <= 0.70 and slope < -0.02:
        return "RELEASING"
    return None


def build_status_text():
    lines = ["📊 MARKET STATUS\n"]

    if market_state["mci"] is not None:
        lines.append(
            f"🌍 MARKET\n"
            f"MCI: {market_state['mci']} | "
            f"slope: {market_state['slope']} | "
            f"phase: {market_state['phase']}\n"
            f"CALM ratio: {market_state['calm_ratio']}\n"
        )
    
    lines.append("—" * 20)

    if not last_state:
        return "Нет данных"

    # --- агрегаты ---
    mci_vals = [v["mci"] for v in last_state.values() if v["mci"] is not None]
    avg_mci = round(sum(mci_vals)/len(mci_vals), 2) if mci_vals else None

    regimes = defaultdict(int)
    phases = defaultdict(int)

    for v in last_state.values():
        regimes[v["regime"]] += 1
        if v["phase"]:
            phases[v["phase"]] += 1

    lines.append(
        f"Market avg MCI: {avg_mci}\n"
        f"CALM: {regimes.get('CALM',0)} | "
        f"UNCERTAIN: {regimes.get('UNCERTAIN',0)} | "
        f"DIRECTIONAL: {regimes.get('DIRECTIONAL_UP',0)+regimes.get('DIRECTIONAL_DOWN',0)}\n"
    )

    if phases:
        dom_phase = max(phases, key=phases.get)
        lines.append(f"Dominant phase: {dom_phase}\n")

    lines.append("—"*20)

    # --- по тикерам ---
    for sym, v in last_state.items():
        lines.append(
            f"{sym}: {v['regime']} | "
            f"MCI {v['mci']} | "
            f"slope {v['slope']} | "
            f"{v['phase']}"
        )

    return "\n".join(lines)

def log_row(row):
    exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(row.keys())
        w.writerow(row.values())

def log_alert(symbol, alert_type, mci, slope, phase):
    if not LOG_ALERTS:
        return

    log_row({
        "ts_unix_ms": now_ts_ms(),
        "symbol": symbol,
        "regime": "",
        "mci": mci,
        "mci_slope": slope,
        "mci_phase": phase,
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
            if SEND_ALERTS_TO_TG:
                tg_send(f"⚠️ {symbol} CALM_COMPRESSION")

        if mci > 0.4 and slope < 0 and ok("CALM_DECAY"):
            alert_ts[symbol]["CALM_DECAY"] = now_ms
            log_alert(symbol, "CALM_DECAY", mci, slope, phase)
            if SEND_ALERTS_TO_TG:
                tg_send(f"⚠️ {symbol} CALM_DECAY")

        if mci < 0.2 and slope < 0 and ok("DIRECTIONAL_PRESSURE"):
            alert_ts[symbol]["DIRECTIONAL_PRESSURE"] = now_ms
            log_alert(symbol, "DIRECTIONAL_PRESSURE", mci, slope, phase)
            if SEND_ALERTS_TO_TG:
                tg_send(f"⚠️ {symbol} DIRECTIONAL_PRESSURE")


        if phase and phase != last_phase[symbol] and ok("PHASE_SHIFT"):
            alert_ts[symbol]["PHASE_SHIFT"] = now_ms
            log_alert(symbol, f"PHASE_SHIFT:{phase}", mci, slope, phase)
            if SEND_ALERTS_TO_TG:
                tg_send(f"🔁 {symbol} PHASE_SHIFT → {phase}")
            last_phase[symbol] = phase


# ---------- DAILY LOG ----------
def daily_sender(stop_event):
    while not stop_event.is_set():
        now = datetime.now(timezone.utc)
        if now.hour == 11 and now.minute < 2:
            if os.path.isfile(HISTORY_FILE):
                tg_send("📎 Daily options log")
                with open(HISTORY_FILE, "rb") as f:
                    tg_requests.post(
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
    threading.Thread(
        target=lambda: [tg_poll() or time.sleep(5) for _ in iter(int, 1)],
        daemon=True
    ).start()
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
        
                    last_state[s] = {
                        "regime": r,
                        "mci": mci,
                        "slope": slope,
                        "phase": phase,
                    }
        
                    maybe_alert(s, phase, mci, slope)
        
                    log_row({
                        "ts_unix_ms": now_ts_ms(),
                        "symbol": s,
                        "regime": r,
                        "mci": mci,
                        "mci_slope": slope,
                        "mci_phase": phase,
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
        
            sleep_for = CHECK_INTERVAL - (time.time() - cycle_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        stop_event.set()
        print("Shutting down", flush=True)

if __name__ == "__main__":
    main()


