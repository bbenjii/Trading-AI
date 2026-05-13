from datetime import datetime, timedelta, time
from typing import Optional

from pydantic import BaseModel

market_open_time = time(hour=9, minute=30)
market_close_time = time(hour=16, minute=0)

pre_market_open_time = time(hour=6, minute=30)

class Configs(BaseModel):
    market_open_time : time
    market_close_time : time
    pre_market_open_time : time
    starting_cash : Optional[float] = None


config = Configs(
        market_open_time=time(hour=9, minute=30),
        market_close_time=time(hour=16, minute=0),
        pre_market_open_time=time(hour=6, minute=30),
        starting_cash=1_000
    )


def load_config():
    return config
