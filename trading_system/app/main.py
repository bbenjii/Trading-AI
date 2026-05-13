from datetime import datetime, timedelta, time
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading_system.app.config import (
    load_config,
    market_open_time,
    market_close_time,
    pre_market_open_time,
)
from trading_system.core.clock import TradingClock

def _market_open(current_time: datetime):
    return market_open_time < current_time.time() < market_close_time


def simulate(
        start_date: str = "2026-04-06",
        simulation_days: int = 1
):
    starting_time = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = starting_time + timedelta(days=simulation_days)
    clock = TradingClock(start=starting_time, end=end_date)

    config = load_config()

    while clock.now < end_date:
        print(clock.now, clock.is_market_open())

        clock.advance_time(minutes=1)


    pass


if __name__ == "__main__":
    simulate()
