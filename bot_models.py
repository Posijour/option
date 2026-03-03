from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SymbolState:
    regime: str
    mci: Optional[float]
    slope: Optional[float]
    phase: Optional[str]
    confidence: Optional[float]


@dataclass
class MarketState:
    mci: Optional[float] = None
    slope: Optional[float] = None
    phase: Optional[str] = None
    calm_ratio: Optional[float] = None
    olsi_slope: Optional[float] = None
    olsi_avg: Optional[float] = None
    liquidity_regime: Optional[str] = None
    divergence: Optional[str] = None
    divergence_diff: Optional[float] = None
    divergence_strength: Optional[float] = None
    divergence_class: Optional[str] = None


@dataclass
class FeedHealth:
    bad_windows: int = 0
    degraded: bool = False


@dataclass
class RuntimeMetrics:
    cycles_total: int = 0
    cycles_over_budget: int = 0
    bybit_errors: int = 0
    okx_errors: int = 0
    db_send_errors: int = 0
    db_sent_events: int = 0
    last_cycle_ms: float = 0.0
    last_sleep_s: float = 0.0
    alerts_sent: int = 0
    health: dict[str, FeedHealth] = field(default_factory=dict)
