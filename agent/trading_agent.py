from datetime import datetime, timezone, timedelta

from agent.article_db_analyzer import get_tickers
from services.marketdata import YahooStockMarket

class TradingAgent:
    def __init__(self):
        self.today = "2026-04-08"
        self.starting_time = datetime.strptime(self.today, "%Y-%m-%d", )
        self.now = self.starting_time
        self.end_time = self.starting_time + timedelta(days=1)
        self.premarket_start = datetime.strptime(f"{self.today} 04:00", "%Y-%m-%d %H:%M")
        self.postmarket_start = datetime.strptime(f"{self.today} 20:00", "%Y-%m-%d %H:%M")
        self.increment = timedelta(minutes=1)

        self.opening_time = datetime.strptime(f"{self.today} 09:30", "%Y-%m-%d %H:%M")
        self.closing_time = datetime.strptime(f"{self.today} 16:00", "%Y-%m-%d %H:%M")
        
        self.holdings_details = {}
        
        self.current_holdings: list = []
        self.history = YahooStockMarket().get_stock_history("NVDA", return_df=True, start_time=self.starting_time,
                                                            end_time=self.end_time, interval="1m", prepost=False)
        pass

    def market_open(self):
        return self.opening_time <= self.now <= self.closing_time

    def premarket_open(self):
        return self.premarket_start <= self.now <= self.opening_time

    def postmarket_open(self):
        return self.closing_time <= self.now <= self.postmarket_start

    def get_history_row(self, time: datetime):
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M")
            row = self.history.loc[timestamp]
            return row
        except Exception as e:
            return None

    def sleep(self, x=1):
        self.now += self.increment * x

    def symbols_analysis(self):
        print("Analyzing symbols...")
        if len(self.current_holdings) == 10:
            symbols = get_tickers(hours_back=4, now=self.now, symbol_filter=self.current_holdings)
        else:
            symbols = get_tickers(hours_back=4, now=self.now)

        for index, row in symbols.iterrows():
            symbol = row["symbol"]
            score = row["score"]
            return_percentage = row["return"]
            score_2 = row["score2"]
            weighted_sentiment_score = row["weighted_sentiment_score"]

            if score >= 0 and return_percentage > 0 and score_2 >= 0 and weighted_sentiment_score > 0 and len(self.current_holdings) < 10:
                ticker_info = YahooStockMarket().get_stock_info(symbol)
                if symbol not in self.current_holdings:
                    print(f"Buying {ticker_info.name} ({symbol}) at ${ticker_info.regularMarketPrice}, timestamp: {self.now}")

                    self.current_holdings.append(symbol)

            if len(self.current_holdings) == 10:
                break

        pass

    def run(self):
        print(self.now)
        print(self.end_time)

        while self.now < self.end_time:
            if self.premarket_open():
                print(f"Current time: {self.now}, Premarket open: {self.premarket_open()}")
                self.sleep(60)
                continue

            elif self.postmarket_open():
                print(f"Current time: {self.now}, Postmarket open: {self.postmarket_open()}")
                self.sleep(60)
                continue

            elif self.market_open():
                # row = self.get_history_row(self.now)
                print(f"""Market open -- Time: {self.now.strftime("%H:%M")}\n Holdings: {self.current_holdings}\n""")

                if len(self.current_holdings) < 10:
                    self.symbols_analysis()
                    pass

            self.sleep(30)

if __name__ == "__main__":
    agent = TradingAgent()
    agent.run()
    # symbols = get_tickers(hours_back=24)
    pass