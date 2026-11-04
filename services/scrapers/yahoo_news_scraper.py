import newspaper
from bs4 import BeautifulSoup
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from services.scrapers.base_news_scraper import BaseScraper, Article
from utils.function_timer import function_timer
from utils.logger import logger
    
class YahooScraper(BaseScraper):
    BASE_URL = "https://finance.yahoo.com/news/"
    # HEADERS = {"User-Agent": "Mozilla/5.0"}

    def __init__(self, **kwargs):
        super().__init__(base_url=self.BASE_URL, scraper_name="Yahoo Scraper",**kwargs)
    
    def get_article_links(self) -> List[str]:
        """
        Fetch latest Yahoo Finance article URLs from the topic stream.
        Return a list of absolute article URLs. The base class enforces limit.
        """
        html = self.fetch_html(self.BASE_URL)
        if not html:
            return []
        soup = BeautifulSoup(html, "lxml")

        # Primary selector: real article links in the topic stream
        candidates = soup.select(
            "div[data-testid='topic-stream'] a[role='article'][href*='/news/']"
        )

        # Fallback in case Yahoo changes layout
        if not candidates:
            candidates = soup.select("ul[class*='stream-items'] li a.subtle-link[href*='/news/']")

        if not candidates:
            candidates = soup.select("a.titles[href*='/articles/'], a.titles[href*='/news/']")
        
        candidates.extend(soup.select("a[href*='/news/'], a[href*='/articles/']"))
        links, seen = [], set()
        
        for a in candidates:
            href = a.get("href")
            if not href:
                continue
            href = urljoin("https://finance.yahoo.com", href)
            parts = urlparse(href)
            if (parts.scheme != "https" or parts.hostname != "finance.yahoo.com"
                    or not (parts.path.startswith("/news/") or "/articles/" in parts.path)
                    or not parts.path.endswith(".html")):
                continue

            url = self.normalize_url(href)

            # Skip videos and duplicates
            if self.is_video_url(url) or url in seen:
                continue

            seen.add(url)
            links.append(url)

        
        return links
        
    def extract_article(self, url: str) -> Optional[Article]:
        """
        Extract one article. 
        """
        html = self.fetch_html(url)
        if not html:
            return None

        news_article = newspaper.article(url=url, input_html=html)
        publish_date = news_article.publish_date
        authors = news_article.authors
        title = news_article.title
        
        soup = BeautifulSoup(html, "html.parser")
        # Article body
        paragraphs = soup.select("div[data-testid='article-body'] p")
        if not paragraphs:
            paragraphs = soup.select("div[data-test-locator='article-body'] p")

        content = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        content = content or (news_article.text or "").strip()

        article = Article(
            url=self.normalize_url(url),
            title=title or "",
            content=content or "",
            publish_date=str(publish_date) if publish_date else None,
            authors=authors or None,
            summary=None,
        )

        try:
            update = self.gemini_client.summarize_article(article) if self.gemini_client else None
            if update:
                article = article.model_copy(update=update)
        except Exception as e:
            article.summary = news_article.summary

        return article


@function_timer
def fetch_yahoo_articles():
    scraper = YahooScraper()
    articles = scraper.scrape()
    for article in articles:
        logger.info(article)
        
@function_timer
def async_fetch_yahoo_articles():
    scraper = YahooScraper(limit=100, async_scrape=True)
    articles = scraper.scrape()
    # for article in articles:
    #     logger.info(article)
    
if __name__ == "__main__":
    # fetch_yahoo_articles()
    async_fetch_yahoo_articles()
