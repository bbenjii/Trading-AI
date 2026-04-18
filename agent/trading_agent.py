from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from agent.article_db_analyzer import get_tickers
from services.marketdata import YahooStockMarket


class TradingAgent:
    def __init__(
        self,
        start_date: str = "2026-04-06",
        simulation_days: int = 1,
        news_lookback_hours: int = 4,
        price_interval: str = "1h",
        decision_interval_minutes: int = 60,
        position_size_pct: float = 0.1,
        max_positions: int = 10,
        commission_per_trade: float = 1.0,
        slippage_bps: int = 5,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.05,
        min_signal_score: float = 0.0,
        min_recent_return_pct: float = 0.0,
        signal_deterioration_threshold: Optional[float] = None,
        max_consecutive_weak_signals: int = 2,
        min_price: float = 5.0,
        us_equities_only: bool = True,
        allow_etfs: bool = False,
        allow_reentry: bool = True,
        reentry_cooldown_minutes: int = 60,
        loss_reentry_cooldown_minutes: int = 24 * 60,
        max_entries_per_symbol: int = 2,
        max_loss_entries_per_symbol: int = 1,
    ):
        self.start_date = start_date
        self.simulation_days = simulation_days
        self.news_lookback_hours = news_lookback_hours
        self.price_interval = price_interval
        self.decision_interval = timedelta(minutes=decision_interval_minutes)
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_signal_score = min_signal_score
        self.min_recent_return_pct = min_recent_return_pct
        self.signal_deterioration_threshold = (
            min_signal_score if signal_deterioration_threshold is None else signal_deterioration_threshold
        )
        self.max_consecutive_weak_signals = max_consecutive_weak_signals
        self.min_price = min_price
        self.us_equities_only = us_equities_only
        self.allow_etfs = allow_etfs
        self.allow_reentry = allow_reentry
        self.reentry_cooldown_minutes = reentry_cooldown_minutes
        self.loss_reentry_cooldown_minutes = loss_reentry_cooldown_minutes
        self.max_entries_per_symbol = max_entries_per_symbol
        self.max_loss_entries_per_symbol = max_loss_entries_per_symbol

        self.starting_time = datetime.strptime(self.start_date, "%Y-%m-%d")
        self.starting_cash = 100_000.0
        self.cash = self.starting_cash
        self.market = YahooStockMarket()

        self.open_hour = 9
        self.open_minute = 30
        self.close_hour = 16
        self.close_minute = 0
        self.premarket_hour = 4
        self.postmarket_hour = 20

        self.first_session_open = self._session_open(self.starting_time)
        self.final_day = self.starting_time + timedelta(days=self.simulation_days - 1)
        self.final_liquidation_time = self._session_close(self.final_day)
        self.end_time = self.final_liquidation_time + timedelta(hours=8)
        self.now = self.starting_time

        self.positions: dict[str, dict] = {}
        self.trade_log: list[dict] = []
        self.equity_curve: list[dict] = []
        self.price_histories: dict[str, pd.DataFrame] = {}
        self.symbol_metadata_cache: dict[str, dict] = {}
        self.last_exit_time: dict[str, datetime] = {}
        self.last_exit_pnl: dict[str, float] = {}
        self.last_exit_reason: dict[str, str] = {}
        self.entry_counts: dict[str, int] = {}
        self.loss_exit_counts: dict[str, int] = {}
        self.invalid_symbols: set[str] = set()
        self.final_liquidation_done = False
        self.foreign_suffixes = (
            ".HK", ".DE", ".TO", ".L", ".PA", ".AS", ".BR", ".SW", ".MI", ".MC",
            ".OL", ".CO", ".ST", ".HE", ".IR", ".WA", ".AX", ".NZ", ".SI", ".KS",
            ".KQ", ".T", ".F", ".BE", ".DU", ".HM", ".MU", ".VI", ".IC", ".SS", ".SZ",
        )
        self.allowed_us_exchanges = {"NYQ", "NMS", "NGM", "NCM", "ASE", "PCX", "BTS", "CBOE", "NASDAQ", "NYSE", "AMEX"}

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path("runs") / f"trading_agent_{self.start_date}_{self.run_id}"

    def _session_open(self, value: datetime) -> datetime:
        return value.replace(hour=self.open_hour, minute=self.open_minute, second=0, microsecond=0)

    def _session_close(self, value: datetime) -> datetime:
        return value.replace(hour=self.close_hour, minute=self.close_minute, second=0, microsecond=0)

    def _premarket_open(self, value: datetime) -> datetime:
        return value.replace(hour=self.premarket_hour, minute=0, second=0, microsecond=0)

    def _postmarket_close(self, value: datetime) -> datetime:
        return value.replace(hour=self.postmarket_hour, minute=0, second=0, microsecond=0)

    def is_weekend(self) -> bool:
        return self.now.weekday() >= 5

    def market_open(self) -> bool:
        if self.is_weekend():
            return False
        return self._session_open(self.now) <= self.now < self._session_close(self.now)

    def premarket_open(self) -> bool:
        if self.is_weekend():
            return False
        return self._premarket_open(self.now) <= self.now < self._session_open(self.now)

    def postmarket_open(self) -> bool:
        if self.is_weekend():
            return False
        return self._session_close(self.now) <= self.now <= self._postmarket_close(self.now)

    def _normalize_history_df(self, df: pd.DataFrame) -> pd.DataFrame:
        history = df.copy()
        if getattr(history.index, "tz", None) is not None:
            history.index = history.index.tz_convert(None)
        return history.sort_index()

    def load_symbol_history(self, symbol: str) -> Optional[pd.DataFrame]:
        if symbol in self.invalid_symbols:
            return None
        if symbol in self.price_histories:
            return self.price_histories[symbol]

        history = self.market.get_stock_history(
            symbol,
            return_df=True,
            start_time=self.first_session_open - timedelta(hours=max(self.news_lookback_hours, 24)),
            end_time=self.final_liquidation_time,
            interval=self.price_interval,
            prepost=False,
        )
        df = (history or {}).get("df")
        if df is None or df.empty:
            self.invalid_symbols.add(symbol)
            return None

        normalized = self._normalize_history_df(df)
        self.price_histories[symbol] = normalized
        return normalized

    def get_price(self, symbol: str, time: datetime) -> Optional[float]:
        history = self.load_symbol_history(symbol)
        if history is None or history.empty:
            return None

        try:
            upto_now = history.loc[:time]
            if upto_now.empty:
                return None
            return float(upto_now.iloc[-1]["Close"])
        except Exception:
            return None

    def get_position_value(self, symbol: str, position: dict) -> float:
        price = self.get_price(symbol, self.now)
        if price is None:
            return 0.0
        return position["qty"] * price

    def portfolio_value(self) -> float:
        positions_value = sum(
            self.get_position_value(symbol, position)
            for symbol, position in self.positions.items()
        )
        return self.cash + positions_value

    def record_equity(self):
        equity = self.portfolio_value()
        self.equity_curve.append(
            {
                "timestamp": self.now,
                "cash": round(self.cash, 2),
                "positions": round(equity - self.cash, 2),
                "equity": round(equity, 2),
            }
        )

    def get_symbol_metadata(self, symbol: str) -> dict:
        if symbol in self.symbol_metadata_cache:
            return self.symbol_metadata_cache[symbol]
        metadata = self.market.get_symbol_metadata(symbol) or {}
        self.symbol_metadata_cache[symbol] = metadata
        return metadata

    def symbol_in_cooldown(self, symbol: str) -> bool:
        if not self.allow_reentry:
            return symbol in self.last_exit_time
        if self.entry_counts.get(symbol, 0) >= self.max_entries_per_symbol:
            return True
        if self.loss_exit_counts.get(symbol, 0) >= self.max_loss_entries_per_symbol:
            return True
        last_exit = self.last_exit_time.get(symbol)
        if last_exit is None:
            return False
        cooldown_minutes = self.reentry_cooldown_minutes
        if self.last_exit_pnl.get(symbol, 0.0) <= 0:
            cooldown_minutes = max(cooldown_minutes, self.loss_reentry_cooldown_minutes)
        return (self.now - last_exit).total_seconds() < cooldown_minutes * 60

    def is_symbol_allowed(self, symbol: str) -> bool:
        if symbol in self.invalid_symbols:
            return False

        normalized_symbol = symbol.upper()
        if (
            "^" in normalized_symbol
            or "=X" in normalized_symbol
            or normalized_symbol.endswith("-USD")
            or normalized_symbol.endswith("USD")
            or normalized_symbol.startswith("^")
            or normalized_symbol.endswith("=X")
            or normalized_symbol.endswith("=F")
            or normalized_symbol.endswith(self.foreign_suffixes)
        ):
            self.invalid_symbols.add(symbol)
            return False

        metadata = self.get_symbol_metadata(symbol)
        quote_type = (metadata.get("quoteType") or "").lower()
        exchange = (metadata.get("exchange") or "").upper()
        currency = (metadata.get("currency") or "").upper()

        if quote_type in {"currency", "index", "mutualfund"}:
            self.invalid_symbols.add(symbol)
            return False

        if not self.allow_etfs and quote_type == "etf":
            self.invalid_symbols.add(symbol)
            return False

        if quote_type and quote_type not in {"equity", "etf"}:
            self.invalid_symbols.add(symbol)
            return False

        if self.us_equities_only:
            if currency and currency != "USD":
                self.invalid_symbols.add(symbol)
                return False
            if exchange and exchange not in self.allowed_us_exchanges:
                self.invalid_symbols.add(symbol)
                return False

        if self.load_symbol_history(symbol) is None:
            self.invalid_symbols.add(symbol)
            return False

        price = self.get_price(symbol, self.now)
        if price is None or price < self.min_price:
            self.invalid_symbols.add(symbol)
            return False

        return True

    def enter_position(self, symbol: str, signal_score: float, reason: str = "news_signal") -> bool:
        if (
            symbol in self.positions
            or len(self.positions) >= self.max_positions
            or self.symbol_in_cooldown(symbol)
            or not self.is_symbol_allowed(symbol)
        ):
            return False

        price = self.get_price(symbol, self.now)
        if price is None:
            return False

        target_notional = self.portfolio_value() * self.position_size_pct
        fill_price = price * (1 + self.slippage_bps / 10_000)
        qty = int(target_notional // fill_price)
        total_cost = qty * fill_price + self.commission_per_trade
        if qty <= 0 or total_cost > self.cash:
            return False

        self.cash -= total_cost
        self.entry_counts[symbol] = self.entry_counts.get(symbol, 0) + 1
        self.positions[symbol] = {
            "qty": qty,
            "entry_price": fill_price,
            "entry_time": self.now,
            "signal_score": signal_score,
            "reason": reason,
            "weak_signal_count": 0,
        }
        self.trade_log.append(
            {
                "timestamp": self.now,
                "symbol": symbol,
                "side": "BUY",
                "qty": qty,
                "price": round(fill_price, 4),
                "reason": reason,
            }
        )
        print(f"BUY {symbol} qty={qty} price={fill_price:.2f} signal={signal_score:.3f} time={self.now}")
        return True

    def exit_position(self, symbol: str, reason: str) -> bool:
        position = self.positions.get(symbol)
        if position is None:
            return False

        price = self.get_price(symbol, self.now)
        if price is None:
            return False

        fill_price = price * (1 - self.slippage_bps / 10_000)
        proceeds = position["qty"] * fill_price - self.commission_per_trade
        pnl = proceeds - position["qty"] * position["entry_price"]
        self.cash += proceeds
        self.trade_log.append(
            {
                "timestamp": self.now,
                "symbol": symbol,
                "side": "SELL",
                "qty": position["qty"],
                "price": round(fill_price, 4),
                "reason": reason,
                "pnl": round(pnl, 2),
            }
        )
        print(f"SELL {symbol} qty={position['qty']} price={fill_price:.2f} pnl={pnl:.2f} reason={reason} time={self.now}")
        del self.positions[symbol]
        self.last_exit_time[symbol] = self.now
        self.last_exit_pnl[symbol] = pnl
        self.last_exit_reason[symbol] = reason
        if pnl <= 0:
            self.loss_exit_counts[symbol] = self.loss_exit_counts.get(symbol, 0) + 1
        return True

    def should_exit_position(self, symbol: str, position: dict, latest_signal: Optional[float]) -> Optional[str]:
        price = self.get_price(symbol, self.now)
        if price is None:
            return None

        pnl_pct = (price / position["entry_price"]) - 1
        if pnl_pct <= -self.stop_loss_pct:
            return "stop_loss"
        if pnl_pct >= self.take_profit_pct:
            return "take_profit"
        if latest_signal is None or latest_signal <= self.signal_deterioration_threshold:
            position["weak_signal_count"] = position.get("weak_signal_count", 0) + 1
        else:
            position["weak_signal_count"] = 0

        if position.get("weak_signal_count", 0) >= self.max_consecutive_weak_signals:
            return "signal_deterioration"
        return None

    def symbols_analysis(self):
        print("Analyzing symbols...")
        symbols = get_tickers(hours_back=self.news_lookback_hours, now=self.now)
        if symbols.empty:
            return

        ranked_signals = {
            row["symbol"]: row
            for _, row in symbols.iterrows()
        }

        for symbol, position in list(self.positions.items()):
            row = ranked_signals.get(symbol)
            latest_signal = None if row is None else row["signal_score"]
            exit_reason = self.should_exit_position(symbol, position, latest_signal)
            if exit_reason:
                self.exit_position(symbol, exit_reason)

        if len(self.positions) >= self.max_positions:
            return

        for _, row in symbols.iterrows():
            symbol = row["symbol"]
            if symbol in self.positions or not self.is_symbol_allowed(symbol):
                continue

            signal_score = row["signal_score"]
            recent_return_pct = row["recent_return_pct"]

            if (
                signal_score > self.min_signal_score
                and pd.notna(recent_return_pct)
                and recent_return_pct > self.min_recent_return_pct
            ):
                self.enter_position(symbol, signal_score)

            if len(self.positions) >= self.max_positions:
                break

    def final_liquidation(self):
        self.now = self.final_liquidation_time
        for symbol in list(self.positions.keys()):
            self.exit_position(symbol, "final_liquidation")
        self.final_liquidation_done = True

    def build_trade_log_df(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price", "reason", "pnl"])
        return pd.DataFrame(self.trade_log)

    def build_equity_curve_df(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame(columns=["timestamp", "cash", "positions", "equity"])
        equity_df = pd.DataFrame(self.equity_curve).sort_values("timestamp").reset_index(drop=True)
        equity_df["running_peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = equity_df["equity"] - equity_df["running_peak"]
        equity_df["drawdown_pct"] = equity_df["equity"] / equity_df["running_peak"] - 1
        return equity_df

    def build_symbol_pnl_df(self, trade_log_df: pd.DataFrame) -> pd.DataFrame:
        if trade_log_df.empty:
            return pd.DataFrame(columns=["symbol", "realized_pnl", "sell_trades", "wins", "losses"])

        sells = trade_log_df[trade_log_df["side"] == "SELL"].copy()
        if sells.empty:
            return pd.DataFrame(columns=["symbol", "realized_pnl", "sell_trades", "wins", "losses"])

        symbol_pnl_df = (
            sells.groupby("symbol", as_index=False)
            .agg(
                realized_pnl=("pnl", "sum"),
                sell_trades=("pnl", "count"),
                wins=("pnl", lambda pnl: int((pnl > 0).sum())),
                losses=("pnl", lambda pnl: int((pnl <= 0).sum())),
            )
            .sort_values("realized_pnl", ascending=False)
            .reset_index(drop=True)
        )
        symbol_pnl_df["realized_pnl"] = symbol_pnl_df["realized_pnl"].round(2)
        return symbol_pnl_df

    def calculate_summary(self, trade_log_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
        sells = trade_log_df[trade_log_df["side"] == "SELL"].copy() if not trade_log_df.empty else pd.DataFrame()
        win_rate = float((sells["pnl"] > 0).mean()) if not sells.empty else 0.0
        max_drawdown = float(equity_df["drawdown"].min()) if not equity_df.empty else 0.0
        max_drawdown_pct = float(equity_df["drawdown_pct"].min()) if not equity_df.empty else 0.0
        final_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else self.portfolio_value()

        return {
            "run_id": self.run_id,
            "start_date": self.start_date,
            "simulation_days": self.simulation_days,
            "news_lookback_hours": self.news_lookback_hours,
            "price_interval": self.price_interval,
            "allow_reentry": self.allow_reentry,
            "starting_cash": round(self.starting_cash, 2),
            "final_cash": round(self.cash, 2),
            "final_equity": round(final_equity, 2),
            "net_pnl": round(final_equity - self.starting_cash, 2),
            "total_trades": int(len(trade_log_df)),
            "closed_trades": int(len(sells)),
            "win_rate": round(win_rate, 4),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 4),
        }

    def save_run_outputs(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        trade_log_df = self.build_trade_log_df()
        equity_df = self.build_equity_curve_df()
        symbol_pnl_df = self.build_symbol_pnl_df(trade_log_df)
        summary = self.calculate_summary(trade_log_df, equity_df)
        summary_df = pd.DataFrame([summary])

        trade_log_df.to_csv(self.run_dir / "trade_log.csv", index=False)
        equity_df.to_csv(self.run_dir / "equity_curve.csv", index=False)
        symbol_pnl_df.to_csv(self.run_dir / "symbol_pnl.csv", index=False)
        summary_df.to_csv(self.run_dir / "summary.csv", index=False)
        return summary

    def print_summary(self):
        summary = self.save_run_outputs()
        print(f"Final cash: {summary['final_cash']:.2f}")
        print(f"Final equity: {summary['final_equity']:.2f}")
        print(f"Net PnL: {summary['net_pnl']:.2f}")
        print(f"Net Pnl (%): {summary['net_pnl'] / summary['starting_cash'] * 100:.2f}%")
        print(f"Trades: {summary['total_trades']}")
        print(f"Win rate: {summary['win_rate']:.2%}")
        print(f"Max drawdown: {summary['max_drawdown']:.2f} ({summary['max_drawdown_pct']:.2%})")
        print(f"Run outputs saved to: {self.run_dir}")

    def advance_time(self):
        if self.market_open():
            self.now += self.decision_interval
            return

        if self.premarket_open():
            next_open = self._session_open(self.now)
            self.now = min(next_open, self.now + timedelta(hours=1))
            return

        if self.postmarket_open() or self.is_weekend():
            next_day = (self.now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            self.now = next_day
            return

        self.now += timedelta(hours=1)

    def run(self):
        print(self.now)
        print(self.end_time)

        while self.now <= self.end_time:
            if not self.final_liquidation_done and self.now >= self.final_liquidation_time:
                self.final_liquidation()
                self.record_equity()

            self.record_equity()

            if self.premarket_open():
                print(f"Current time: {self.now}, Premarket open: {self.premarket_open()}")
                self.advance_time()
                continue

            if self.postmarket_open():
                print(f"Current time: {self.now}, Postmarket open: {self.postmarket_open()}")
                self.advance_time()
                continue

            if self.market_open():
                print(
                    f"""Market open -- Time: {self.now.strftime("%Y-%m-%d %H:%M")}\n Holdings: {list(self.positions.keys())}\n Equity: {self.portfolio_value():.2f}\n"""
                )
                self.symbols_analysis()

            self.advance_time()

        if not self.final_liquidation_done:
            self.final_liquidation()
        self.record_equity()
        self.print_summary()


if __name__ == "__main__":
    # agent = TradingAgent()

    agent = TradingAgent(
        start_date="2026-03-02",
        simulation_days=11,
        news_lookback_hours=24,
        price_interval="1h",
        allow_reentry=True,
        reentry_cooldown_minutes=60,
    )
    agent.run()
