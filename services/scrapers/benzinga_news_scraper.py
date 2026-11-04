from services.scrapers.publisher_feed_scraper import PublisherFeedScraper


class BenzingaNewsScraper(PublisherFeedScraper):
    FEED_URL = 'https://www.benzinga.com/feed'
    ARTICLE_HOSTS = {'benzinga.com', 'www.benzinga.com'}

    def __init__(self, **kwargs):
        super().__init__(scraper_name='Benzinga Scraper', **kwargs)


if __name__ == "__main__":
    scraper = BenzingaNewsScraper()
    # links = scraper.get_article_links()
    # 
    # print(links)

    articles = scraper.scrape()
    for art in articles:
        print(
            f"Title: {art.title}\nURL: {art.url}\nPublish Date: {art.publish_date}\nAuthors: {art.authors}\n Content: {art.content}")
    pass