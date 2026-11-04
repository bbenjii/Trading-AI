from services.scrapers.publisher_feed_scraper import PublisherFeedScraper
from utils.function_timer import function_timer
from utils.logger import logger


class MarketWatchScraper(PublisherFeedScraper):
    FEED_URL = 'https://feeds.content.dowjones.io/public/rss/mw_topstories'
    ARTICLE_HOSTS = {'marketwatch.com', 'www.marketwatch.com'}

    def __init__(self, **kwargs):
        super().__init__(scraper_name='MarketWatch Scraper', **kwargs)


@function_timer
def fetch_marketwatch_articles():
    scraper = MarketWatchScraper()
    articles = scraper.scrape()
    for article in articles:
        logger.info(article)


@function_timer
def async_fetch_marketwatch_articles():
    scraper = MarketWatchScraper(limit=100, async_scrape=True)
    articles = scraper.scrape()
    # for article in articles:
    #     logger.info(article)


if __name__ == "__main__":
    # fetch_marketwatch_articles()
    async_fetch_marketwatch_articles()
    # url = "https://www.wsj.com/world/trump-says-hed-rather-end-war-than-send-tomahawks-to-ukraine-04956387"
    # scraper = MarketWatchScraper()
    # article = scraper.extract_article(url)
    # print(article)