from __future__ import annotations
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict

import pandas as pd
import yfinance as yf
from click import DateTime
from pydantic import BaseModel

from services.database import untracked_symbols_service
from utils import function_timer, logger


def timestamp_to_datetime(value: Optional[float]) -> Optional[str]:
    """Convert various timestamp formats to a UTC string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt_obj = value.astimezone(timezone.utc)
    elif isinstance(value, (int, float)):
        dt_obj = datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        return None
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


class Ticker(BaseModel):
    symbol: str
    name: str
    currency: Optional[str] = None
    regularMarketTime: Optional[str] = None
    regularMarketPrice: Optional[float] = None
    regularMarketDayHigh: Optional[float] = None
    regularMarketDayLow: Optional[float] = None
    fiftyTwoWeekHigh: Optional[float] = None
    fiftyTwoWeekLow: Optional[float] = None


class MarketCandle(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int]


class YahooStockMarket:
    """
    Wrapper around yfinance to keep a consistent interface with the previous Yahoo API client.
    """

    def __init__(self, *, auto_adjust: bool = False) -> None:
        self.auto_adjust = auto_adjust

    def _dataframe_to_candles(self, df: pd.DataFrame) -> List[MarketCandle]:
        if df.empty:
            return []

        candles: List[MarketCandle] = []
        for index, row in df.iterrows():
            ts = timestamp_to_datetime(index.to_pydatetime() if hasattr(index, "to_pydatetime") else index)
            if ts is None:
                continue
            if pd.isna(row["Open"]) or pd.isna(row["Close"]):
                continue
            candles.append(
                MarketCandle(
                    date=ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=None if pd.isna(row["Volume"]) else int(row["Volume"]),
                )
            )
        return candles

    def get_stock_info(self, symbol: str = "AVGO") -> Optional[Ticker]:
        ticker = yf.Ticker(symbol)
        try:
            info: Dict = ticker.get_info()
        except Exception as exc:  # pragma: no cover - remote errors
            logger.error("Unable to fetch %s info via yfinance: %s", symbol, exc)
            return None

        fast_info = getattr(ticker, "fast_info", {}) or {}

        return Ticker(
            symbol=symbol.upper(),
            name=info.get("shortName") or info.get("longName") or symbol.upper(),
            currency=info.get("currency") or fast_info.get("currency"),
            regularMarketTime=timestamp_to_datetime(info.get("regularMarketTime")),
            regularMarketPrice=info.get("regularMarketPrice") or fast_info.get("last_price"),
            regularMarketDayHigh=info.get("regularMarketDayHigh") or fast_info.get("day_high"),
            regularMarketDayLow=info.get("regularMarketDayLow") or fast_info.get("day_low"),
            fiftyTwoWeekHigh=info.get("fiftyTwoWeekHigh") or fast_info.get("year_high"),
            fiftyTwoWeekLow=info.get("fiftyTwoWeekLow") or fast_info.get("year_low"),
        )
    PeriodInterval = Literal["1m", "1h", "1d", "1w", "1mo"]
    
    def get_multiple_stock_df(
            self,
            ticker_symbols: list[str],
            interval: PeriodInterval = "1d",
            length: int = 10,
            end_time=None,
            prepost=False,
    ):
        df_set = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            fut_map = {pool.submit(self.get_stock_df, ticker_symbol=ticker_symbol, interval=interval, length=length, end_time=end_time, prepost=prepost): (i, ticker_symbol) for i, ticker_symbol in enumerate(ticker_symbols)}
            for fut in as_completed(fut_map):
                i, ticker_symbol = fut_map[fut]
                try:
                    df = fut.result()
                    if df is not None:
                        df_set[ticker_symbols[i]] = df
                except Exception as e:
                    logger.warning("Worker failed for %s: %s", ticker_symbol, e)
        
        return df_set
        
    def get_stock_df(
            self,
            ticker_symbol: str,
            interval: PeriodInterval = "1d",
            length: int = 10,
            end_time = None,
            prepost=False,

    ):
        ticker = yf.Ticker(ticker_symbol)
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        interval_map = {
            "1m": timedelta(minutes=length),
            "1h": timedelta(hours=length),
            "1d": timedelta(days=length),
            "1w": timedelta(weeks=length),
            "1mo": timedelta(days=30 * length),
        }

        start_time = end_time - interval_map[interval]

        try:
            df = ticker.history(
                start=start_time,
                end=end_time,
                interval=interval,
                auto_adjust=self.auto_adjust,
                actions=False,
                prepost=prepost,
            )
        except Exception as exc:  # pragma: no cover - remote errors
            try:
                res = self.get_stock_info(symbol=ticker_symbol)
            except Exception:
                untracked_symbols_service.add_untracked_symbols([ticker_symbol])

            logger.error("Unable to fetch %s history via yfinance: %s", ticker_symbol, exc)
            return None
        
        return df
        
    @function_timer
    def get_stock_history(
        self,
        symbol: str = "AVGO",
        days: int = 7,
        interval: str = "1d",
        return_df: bool = True,
        start_time = None, end_time = None,
            prepost = False,
    ):
        
        ticker = yf.Ticker(symbol)
        if start_time and end_time:
            end = end_time
            start = start_time
        else:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)

        try:
            df = ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=self.auto_adjust,
                actions=False,
                prepost=prepost,
            ) 
        except Exception as exc:  # pragma: no cover - remote errors
            logger.error("Unable to fetch %s history via yfinance: %s", symbol, exc)
            return None
        
        if return_df:
            return {"df":df}
        
        return self._dataframe_to_candles(df)

    # @function_timer
    def get_intraday_history(
        self,
        symbol: str = "AVGO",
        interval: str = "1m",
        include_pre_post: bool = True,
    ) -> Optional[List[MarketCandle]]:
        ticker = yf.Ticker(symbol)
        try:
            df = ticker.history(
                period="5d",
                interval=interval,
                auto_adjust=self.auto_adjust,
                actions=False,
                prepost=include_pre_post,
            )
        except Exception as exc:
            logger.error("Unable to fetch %s intraday data via yfinance: %s", symbol, exc)
            return None

        if df.empty:
            return []

        # Filter to the most recent trading session
        normalized = df.index.normalize()
        target_day = normalized.max()
        session_df = df[normalized == target_day]
        if session_df.empty and len(normalized.unique()) > 1:
            # fallback to previous day if last day has no rows (rare)
            second_last = sorted(set(normalized))[-2]
            session_df = df[normalized == second_last]
        
        candles = self._dataframe_to_candles(session_df)
        length = len(candles)
        return candles
    
    def get_momentum(self):
        pass
    
    def get_most_active_symbols(self, count: int = 200) -> List[Dict[str, Optional[str]]]:
        try:
            result = yf.screen("most_actives", count=count)
        except Exception as exc:
            logger.error("Unable to fetch most active symbols via yfinance: %s", exc)
            return []

        entries = result.get("quotes") or result.get("records") or []
        symbols: List[Dict[str, Optional[str]]] = []
        for entry in entries[:count]:
            symbol = entry.get("symbol") or entry.get("ticker")
            if not symbol:
                continue
            symbols.append(
                {
                    "symbol": symbol,
                    "shortName": entry.get("shortName") or entry.get("companyName"),
                }
            )
        return symbols


def continuous_fetch(symbol: str = "NVDA", days: int = 5, delay_seconds: float = 2.0) -> None:
    """
    Simple helper to repeatedly poll yfinance for manual profiling or debugging.
    """
    stock_market = YahooStockMarket()

    num = 0
    stock_cache: Optional[List[MarketCandle]] = None
    start_time = time.perf_counter()
    
    while True:
        stock = stock_market.get_stock_info(symbol=symbol)
        if stock:
            num += 1
            stock_cache = stock
            # print(f"Fetch number: {num}")
            # print(f"Elapsed time: {time.perf_counter() - start_time:.4f} seconds")
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now}: {round(stock_cache.regularMarketPrice, 4)} $")
            time.sleep(delay_seconds)
        else:
            break

    elapsed_time = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Number of fetches: {num}")
    if stock_cache:
        print(f"Number of candles: {len(stock_cache)}")


def main() -> None:
    symbol = "NVDA"
    stock_market = YahooStockMarket()

    ticker = stock_market.get_stock_info(symbol=symbol)
    daily_candles = stock_market.get_stock_history(symbol=symbol, days=60) or []
    
    df = daily_candles.get("df")
    list_of_dicts = df.to_dict(orient='index')

    intraday = stock_market.get_intraday_history(symbol=symbol, include_pre_post=True) or []
    most_active = stock_market.get_most_active_symbols(count=10)

    print(ticker)
    print(f"Fetched {len(daily_candles)} daily candles for {symbol}")
    print(f"Fetched {len(intraday)} intraday candles for {symbol}")
    print("Most active symbols:", [entry["symbol"] for entry in most_active if entry.get("symbol")])


if __name__ == "__main__":
    # continuous_fetch(delay_seconds=0.5)
    main()
    # print(YahooStockMarket().get_stock_info("$BRK"))