import nltk 
import os
root = os.path.dirname(os.path.abspath(__file__))
download_dir = os.path.join(root, 'nltk_data')
nltk.data.path.append(download_dir)
from .marketwatch_news_scraper import MarketWatchScraper
from .yahoo_news_scraper import YahooScraper
from .base_news_scraper import BaseScraper
from .ddg_news_scraper import DdgScraper
from .newsapi_news_scraper import NewsApiScraper
from .seekingalpha_news_scraper import SeekingAlphaScraper
from .benzinga_news_scraper import BenzingaNewsScraper
from .france24_news_scraper import France24Scraper
