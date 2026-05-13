from typing import Literal
from datetime import datetime, timedelta, time
from trading_system.app.config import market_open_time, market_close_time


class TradingClock:
    def __init__(self, start, end, mode: Literal["simulation", "live"] = "simulation"):
        self.start = start
        self.end = end
        self.mode = mode

        self.now = self.start

    def is_market_open(self):
        return market_open_time < self.now.time() < market_close_time

    def now(self):
        ...

    def next_tick(self):
        ...

    def advance_time(self, hours=0, minutes=0, seconds=0):
        self.now += timedelta(hours=hours, minutes=minutes, seconds=seconds)
