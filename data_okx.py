import math
import time
from datetime import datetime, timezone

import requests

OKX_BASE_URL = "https://www.okx.com"


def _safe_float(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


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


def parse_okx_symbol(inst_id):
    p = inst_id.split("-")
    return {
        "base": p[0],
        "expiry": datetime.strptime(p[2], "%y%m%d").replace(tzinfo=timezone.utc),
        "strike": float(p[3]),
        "type": "CALL" if p[4] == "C" else "PUT",
    }


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
