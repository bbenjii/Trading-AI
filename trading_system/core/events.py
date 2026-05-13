from dataclasses import dataclass
from datetime import datetime

@dataclass
class MarketEvent:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class SignalEvent:
    strategy_id: str
    symbol: str
    target_weight: float
    reason: str

@dataclass
class OrderEvent:
    symbol: str
    side: str
    quantity: int
    order_type: str

@dataclass
class FillEvent:
    symbol: str
    side: str
    quantity: int
    fill_price: float
    commission: float