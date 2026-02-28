import math

CREDIT_CHEAP = 0.30
CREDIT_EXPENSIVE = 0.15
SKEW_DOWN = 1.20
SKEW_UP = 0.90
OLSI_THRESHOLD = 0.05
MCI_WINDOW = 12

PHASE_CENTERS = {
    "ACCUMULATING_CALM": (0.6,  0.05),
    "STABLE_CALM":       (0.7,  0.00),
    "OVERCOMPRESSED":   (0.9,  0.00),
    "BREAKING_COMPRESSION": (0.9, -0.08),
    "RELEASING":        (0.6, -0.05),
}


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
    skew = (sum(piv) / len(piv)) / (sum(civ) / len(civ)) if piv and civ else 1.0

    if short_vol:
        return "CALM"
    if skew > SKEW_DOWN and downside != "cheap":
        return "DIRECTIONAL_DOWN"
    if skew < SKEW_UP and upside != "cheap":
        return "DIRECTIONAL_UP"
    return "UNCERTAIN"


def atm_weight(strike, spot, symbol=None, k=2.5):
    if spot is None or spot <= 0:
        return 0.0

    if symbol == "BTC":
        k = 2.0
    elif symbol == "ETH":
        k = 3.0

    return math.exp(-k * abs(strike - spot) / spot)


def okx_liquidity_structure_index(chain, spot, min_active_options, logger, symbol=None):
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
        w = atm_weight(o["strike"], spot, symbol=symbol)
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

    if n_active < min_active_options or weighted_active <= 0:
        logger.debug(
            "OLSI %s too few active: n_total=%d n_active=%d",
            symbol or "N/A", len(chain), n_active
        )
        return 0.0

    if spread_weight_sum <= 0:
        logger.debug("OLSI %s spread_weight_sum<=0", symbol or "N/A")
        return 0.0

    asr = min(1.0, weighted_active / max(weighted_total, 1e-6))
    avg_rel_spread = weighted_rel_spread_sum / spread_weight_sum
    nss = 1 / (1 + avg_rel_spread)

    cp_sum = weighted_calls + weighted_puts
    if cp_sum <= 0:
        logger.debug("OLSI %s no call/put structure", symbol or "N/A")
        return 0.0

    sb = min(weighted_calls, weighted_puts) / cp_sum

    return round(asr * nss * sb, 4)


def mci_value(reg):
    return 1 if reg == "CALM" else -1 if reg.startswith("DIRECTIONAL") else 0


def calc_mci(hist, symbol):
    h = hist[symbol]
    return round(sum(h) / len(h), 2) if len(h) == MCI_WINDOW else None


def calc_slope(hist, symbol):
    h = list(hist[symbol])
    if len(h) < MCI_WINDOW:
        return None
    half = MCI_WINDOW // 2
    return round(sum(h[half:]) / half - sum(h[:half]) / half, 3)


def calc_okx_olsi_mean(hist, symbol):
    h = hist[symbol]
    if len(h) < MCI_WINDOW:
        return None
    return round(sum(h) / len(h), 4)


def calc_okx_olsi_slope(hist, symbol):
    h = list(hist[symbol])
    if len(h) < MCI_WINDOW:
        return None

    first = h[0]
    last = h[-1]

    if first == 0:
        return None

    return round((last - first) / first, 4)


def calc_market_olsi_slope(hist, symbols):
    slopes = []

    for s in symbols:
        val = calc_okx_iv_slope(hist, s)
        if val is not None:
            slopes.append(val)

    if not slopes:
        return None

    return round(sum(slopes) / len(slopes), 4)

def classify_olsi_slope(
    slope,
    threshold=OLSI_THRESHOLD
):
    if slope is None:
        return None
    if slope > threshold:
        return "LIQUIDITY_EXPANDING"
    if slope < -threshold:
        return "LIQUIDITY_CRUSH"
    return "LIQUIDITY_FLAT"


def classify_divergence_strength(strength):
    if strength is None:
        return None

    if strength < 0.10:
        return "NONE"

    if strength < 0.20:
        return "WEAK"

    if strength <= 0.35:
        return "MODERATE"

    return "STRONG"
    
    
def classify_mci_olsi_divergence(mci, okx_olsi_avg, threshold=0.10):
    if mci is None or okx_olsi_avg is None:
        return None, None, None, None, None

    mci_norm = round((mci + 1) / 2, 3)
    diff = round(mci_norm - okx_olsi_avg, 3)
    strength = round(abs(diff), 3)

    if abs(diff) < threshold:
        div_type = "ALIGNED"
    elif diff >= threshold:
        div_type = "CALM_WITHOUT_LIQUIDITY"
    else:
        div_type = "LIQUIDITY_WITHOUT_CALM"

    strength_class = classify_divergence_strength(strength)

    return div_type, diff, strength, strength_class, mci_norm


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


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
