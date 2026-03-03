import logging
import threading
import time
from collections import deque
from threading import Event

from analytics import (
    calc_mci,
    calc_slope,
    mci_phase,
    mci_value,
    calc_market_olsi_slope,
    classify_olsi_slope,
    phase_confidence,
    okx_liquidity_structure_index,
    classify_mci_olsi_divergence,
)
from bot_models import FeedHealth, MarketState, RuntimeMetrics, SymbolState
from config import AppConfig
from data_bybit import interpret_bybit_market
from data_okx import get_okx_near_chain, get_okx_spot, get_okx_tickers
from http_server import run_http_server
from runtime_services import TelemetryBuffer, send_telegram_alert

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("option-bot")

stop_event = Event()


class BotRuntime:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.regime_hist = {s: deque(maxlen=cfg.stability_window) for s in cfg.symbols}
        self.mci_hist = {s: deque(maxlen=cfg.mci_window) for s in cfg.symbols}
        self.okx_olsi_hist = {s: deque(maxlen=cfg.mci_window) for s in cfg.okx_symbols}
        self.phase_hist = {s: deque(maxlen=cfg.mci_window) for s in cfg.symbols}
        self.last_state: dict[str, SymbolState] = {}
        self.market_state = MarketState()
        self.next_market_log_ts = None
        self.next_bybit_market_log_ts = None

        self.metrics = RuntimeMetrics(
            health={"bybit": FeedHealth(), "okx": FeedHealth()}
        )
    self.telemetry = TelemetryBuffer(
            cfg.supabase_url, cfg.supabase_key, cfg.db_batch_size, self.metrics, logger
        )

def get_okx_olsi(self, symbol, tickers=None):
        near_chain = get_okx_near_chain(symbol, tickers=tickers)
        spot = get_okx_spot(symbol)
        if spot is None:
            logger.debug("OLSI %s spot unavailable", symbol)
            return 0.0
        return okx_liquidity_structure_index(
            near_chain,
            spot,
            min_active_options=self.cfg.min_active_options,
            logger=logger,
            symbol=symbol,
        )

def update_feed_health(self, feed_name, has_data):
        state = self.metrics.health[feed_name]
        if has_data:
            state.bad_windows = 0
            if state.degraded:
                send_telegram_alert(
                    self.cfg.telegram_bot_token,
                    self.cfg.telegram_chat_id,
                    logger,
                    self.metrics,
                    f"{feed_name} recovered",
                )
                state.degraded = False
            return
        state.bad_windows += 1
        if state.bad_windows >= self.cfg.degrade_windows and not state.degraded:
            send_telegram_alert(
                self.cfg.telegram_bot_token,
                self.cfg.telegram_chat_id,
                logger,
                self.metrics,
                f"{feed_name} degraded",
            )
            state.degraded = True

    def _send_market_logs(self):
        now_ms = int(time.time() * 1000)
        if self.next_market_log_ts is None:
            self.next_market_log_ts = now_ms + self.cfg.market_log_interval * 1000
        if self.next_bybit_market_log_ts is None:
            self.next_bybit_market_log_ts = now_ms + self.cfg.market_log_interval * 1000

        if now_ms >= self.next_market_log_ts:
            row = {
                "ts_unix_ms": now_ms,
                "symbol": "MARKET",
                "okx_olsi_avg": self.market_state.olsi_avg,
                "okx_olsi_slope": self.market_state.olsi_slope,
                "okx_liquidity_regime": self.market_state.liquidity_regime,
                "divergence_type": self.market_state.divergence,
                "divergence_diff": self.market_state.divergence_diff,
                "divergence_strength": self.market_state.divergence_strength,
            }
            self.telemetry.enqueue("okx_market_state", row)
            while self.next_market_log_ts <= now_ms:
                self.next_market_log_ts += self.cfg.market_log_interval * 1000

        if now_ms >= self.next_bybit_market_log_ts and self.last_state:
            mci_vals = [v.mci for v in self.last_state.values() if v.mci is not None]
            slope_vals = [v.slope for v in self.last_state.values() if v.slope is not None]
            confidence_vals = [
                v.confidence for v in self.last_state.values() if v.confidence is not None
            ]
            regimes = [v.regime for v in self.last_state.values() if v.regime]
            if mci_vals and slope_vals:
                bybit_mci = round(sum(mci_vals) / len(mci_vals), 2)
                bybit_slope = round(sum(slope_vals) / len(slope_vals), 3)
                row = {
                    "ts_unix_ms": now_ms,
                    "symbol": "BYBIT",
                    "regime": max(set(regimes), key=regimes.count) if regimes else None,
                    "mci": bybit_mci,
                    "mci_slope": bybit_slope,
                    "mci_phase": mci_phase(bybit_mci, bybit_slope),
                    "confidence": round(sum(confidence_vals) / len(confidence_vals), 2)
                    if confidence_vals
                    else None,
                }
                self.telemetry.enqueue("bybit_market_state", row)
            while self.next_bybit_market_log_ts <= now_ms:
                self.next_bybit_market_log_ts += self.cfg.market_log_interval * 1000

    def compute_market_state(self):
        mci_vals = [v.mci for v in self.last_state.values() if v.mci is not None]
        slope_vals = [v.slope for v in self.last_state.values() if v.slope is not None]

        if mci_vals and slope_vals:
            market_mci = round(sum(mci_vals) / len(mci_vals), 2)
            market_slope = round(sum(slope_vals) / len(slope_vals), 3)
            market_phase = mci_phase(market_mci, market_slope)
        else:
            market_mci = market_slope = market_phase = None

        market_olsi_vals = []
        for s in self.cfg.okx_symbols:
            h = self.okx_olsi_hist.get(s)
            if h and len(h) >= self.cfg.mci_window:
                market_olsi_vals.append(sum(h) / len(h))
        market_olsi_avg = (
            round(sum(market_olsi_vals) / len(market_olsi_vals), 4) if market_olsi_vals else None
        )

            div, div_diff, div_strength, div_class, _ = classify_mci_olsi_divergence(
            market_mci, market_olsi_avg
        )

        calm_count = sum(1 for v in self.last_state.values() if v.regime == "CALM")
        calm_ratio = round(calm_count / max(len(self.cfg.symbols), 1), 2)

        olsi_slope = calc_market_olsi_slope(self.okx_olsi_hist, self.cfg.okx_symbols)
        liquidity_regime = classify_olsi_slope(olsi_slope)

        self.market_state = MarketState(
            mci=market_mci,
            slope=market_slope,
            phase=market_phase,
            calm_ratio=calm_ratio,
            olsi_slope=olsi_slope,
            olsi_avg=market_olsi_avg,
            liquidity_regime=liquidity_regime,
            divergence=div,
            divergence_diff=div_diff,
            divergence_strength=div_strength,
            divergence_class=div_class,
        )

    def select_next_interval(self):
        if self.metrics.health["bybit"].degraded or self.metrics.health["okx"].degraded:
            return self.cfg.min_check_interval
        if self.market_state.phase in ("OVERCOMPRESSED", "BREAKING_COMPRESSION"):
            return self.cfg.min_check_interval
        if self.market_state.phase is None:
            return min(self.cfg.max_check_interval, self.cfg.check_interval + 60)
        return self.cfg.check_interval

    def metrics_text(self):
        return (
            f"bot_cycles_total {self.metrics.cycles_total}\n"
            f"bot_cycles_over_budget {self.metrics.cycles_over_budget}\n"
            f"bot_bybit_errors_total {self.metrics.bybit_errors}\n"
            f"bot_okx_errors_total {self.metrics.okx_errors}\n"
            f"bot_db_send_errors_total {self.metrics.db_send_errors}\n"
            f"bot_db_sent_events_total {self.metrics.db_sent_events}\n"
            f"bot_alerts_sent_total {self.metrics.alerts_sent}\n"
            f"bot_last_cycle_ms {self.metrics.last_cycle_ms:.2f}\n"
            f"bot_last_sleep_s {self.metrics.last_sleep_s:.2f}\n"
            f"bot_bybit_degraded {1 if self.metrics.health['bybit'].degraded else 0}\n"
            f"bot_okx_degraded {1 if self.metrics.health['okx'].degraded else 0}\n"
        )

    def run_cycle(self):
        cycle_start = time.time()
        self.metrics.cycles_total += 1
        okx_tickers_cache = None
        bybit_has_data = False
        okx_has_data = False

        try:
            okx_tickers_cache = get_okx_tickers()
        except Exception as e:
            self.metrics.okx_errors += 1
            logger.warning("failed to fetch OKX tickers cache: %s", e)

        for s in self.cfg.symbols:
            try:
                bybit_r = interpret_bybit_market(s)
                if not bybit_r:
                    continue
                bybit_has_data = True
                self.regime_hist[s].append(bybit_r)
                self.mci_hist[s].append(mci_value(bybit_r))
                mci = calc_mci(self.mci_hist, s)
                slope = calc_slope(self.mci_hist, s)
                phase = mci_phase(mci, slope)
                if phase:
                    self.phase_hist[s].append(phase)
                self.last_state[s] = SymbolState(
                    regime=bybit_r,
                    mci=mci,
                    slope=slope,
                    phase=phase,
                    confidence=phase_confidence(mci, slope, list(self.phase_hist[s])),
                )
            except Exception as e:
                self.metrics.bybit_errors += 1
                logger.exception("cycle error for %s: %s", s, e)

        for s in self.cfg.okx_symbols:
            try:
                olsi = self.get_okx_olsi(s, tickers=okx_tickers_cache)
                self.okx_olsi_hist[s].append(olsi)
                okx_has_data = True
            except Exception as e:
                self.metrics.okx_errors += 1
                logger.exception("OKX cycle error for %s: %s", s, e)

        self.update_feed_health("bybit", bybit_has_data)
        self.update_feed_health("okx", okx_has_data)

        self.compute_market_state()

        elapsed = time.time() - cycle_start
        self.metrics.last_cycle_ms = elapsed * 1000
        over_budget = elapsed > self.cfg.cycle_budget_seconds
        if over_budget:
            self.metrics.cycles_over_budget += 1
            logger.warning("cycle over budget %.2fs > %.2fs", elapsed, self.cfg.cycle_budget_seconds)
        else:
            self._send_market_logs()

        next_interval = self.select_next_interval()
        sleep_for = max(0.0, next_interval - elapsed)
        self.metrics.last_sleep_s = sleep_for
        return sleep_for


def main():
    cfg = AppConfig.from_env()
    logger.setLevel(logging.getLevelName(__import__("os").getenv("LOG_LEVEL", "INFO")))

    runtime = BotRuntime(cfg)
    runtime.telemetry.start()
    
    threading.Thread(
        target=run_http_server,
        args=(stop_event,),
        kwargs={"metrics_getter": runtime.metrics_text},
        daemon=True,
    ).start()

    logger.info("service started: symbols=%s interval=%ss", ",".join(cfg.symbols), cfg.check_interval)
    try:
        while not stop_event.is_set():
            sleep_for = runtime.run_cycle()
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        runtime.telemetry.stop()


if __name__ == "__main__":
    main()
