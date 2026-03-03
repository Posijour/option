import os
from dataclasses import dataclass


def _parse_list(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _parse_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _parse_float(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True)
class AppConfig:
    supabase_url: str
    supabase_key: str
    symbols: list[str]
    okx_symbols: list[str]
    check_interval: int
    market_log_interval: int
    stability_window: int
    mci_window: int
    min_active_options: int
    degrade_windows: int
    cycle_budget_seconds: float
    min_check_interval: int
    max_check_interval: int
    db_batch_size: int

    telegram_bot_token: str | None
    telegram_chat_id: str | None


    @classmethod
    def from_env(cls):
        cfg = cls(
            supabase_url=os.getenv("SUPABASE_URL", "").strip(),
            supabase_key=os.getenv("SUPABASE_KEY", "").strip(),
            symbols=_parse_list("SYMBOLS", ["BTC", "ETH"]),
            okx_symbols=_parse_list("OKX_SYMBOLS", ["BTC", "ETH"]),
            check_interval=_parse_int("CHECK_INTERVAL", 300),
            market_log_interval=_parse_int("MARKET_LOG_INTERVAL", 30 * 60),
            stability_window=_parse_int("STABILITY_WINDOW", 3),
            mci_window=_parse_int("MCI_WINDOW", 12),
            min_active_options=_parse_int("MIN_ACTIVE_OPTIONS", 10),
            degrade_windows=_parse_int("DEGRADE_WINDOWS", 3),
            cycle_budget_seconds=_parse_float("CYCLE_BUDGET_SECONDS", 8.0),
            min_check_interval=_parse_int("MIN_CHECK_INTERVAL", 120),
            max_check_interval=_parse_int("MAX_CHECK_INTERVAL", 600),
            db_batch_size=_parse_int("DB_BATCH_SIZE", 10),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        )
        cfg.validate()
        return cfg

    def validate(self):
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY are required")
        if self.check_interval <= 0:
            raise ValueError("CHECK_INTERVAL must be > 0")
        if self.min_check_interval <= 0 or self.max_check_interval < self.min_check_interval:
            raise ValueError("MIN/MAX_CHECK_INTERVAL invalid")
