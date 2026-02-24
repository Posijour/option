import time
from datetime import datetime, timezone

import requests

from analytics import regime_for_expiry

BYBIT_BASE_URL = "https://api.bybit.com"
NEAR_MIN = 0
NEAR_MAX = 3
MID_MIN = 7
MID_MAX = 14


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


def parse_symbol(sym):
    p = sym.split("-")
    return {
        "base": p[0],
        "expiry": datetime.strptime(p[1], "%d%b%y").replace(tzinfo=timezone.utc),
        "strike": float(p[2]),
        "type": "CALL" if p[3] == "C" else "PUT",
    }


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
