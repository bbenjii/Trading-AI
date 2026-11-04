from services.scrapers.publisher_feed_scraper import PublisherFeedScraper


class SeekingAlphaScraper(PublisherFeedScraper):
    FEED_URL = 'https://seekingalpha.com/market_currents.xml'
    ARTICLE_HOSTS = {'seekingalpha.com', 'www.seekingalpha.com'}

    def __init__(self, **kwargs):
        super().__init__(scraper_name='Seeking Alpha Scraper', **kwargs)


if __name__ == "__main__":
    SeekingAlphaScraper().scrape()
