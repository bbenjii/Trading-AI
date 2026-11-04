"""Read public publisher feeds, retaining whether the supplied text is an excerpt."""
import json
import re
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import requests

from models import Article
from services.scrapers.base_news_scraper import BaseScraper


def plain_text(html):
    soup = BeautifulSoup(html or '', 'html.parser')
    for node in soup.select('script, style, iframe'):
        node.decompose()
    return re.sub(r'\s+([.,;:!?])', r'\1', soup.get_text(' ', strip=True))


class PublisherFeedScraper(BaseScraper):
    FEED_URL = ''
    ARTICLE_HOSTS = set()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feed_articles = {}

    def get_article_links(self):
        self.feed_articles = {}
        response = requests.get(self.FEED_URL, headers=self.headers, timeout=15)
        response.raise_for_status()
        feed = BeautifulSoup(response.text, 'xml')
        if feed.find('rss') is None or feed.find('channel') is None:
            raise ValueError('Publisher did not return an RSS feed')
        items = feed.find_all('item')
        self.scrape_metadata.update(feed_valid=True, feed_item_count=len(items))
        for item in items:
            link, title = item.find('link'), item.find('title')
            if link is None or title is None:
                continue
            url = self.normalize_url(link.get_text(strip=True))
            parts = urlsplit(url)
            if parts.scheme != 'https' or parts.hostname not in self.ARTICLE_HOSTS:
                continue
            if self.is_video_url(url) or url in self.feed_articles:
                continue
            encoded, description = item.find('encoded'), item.find('description')
            full = plain_text(encoded.get_text()) if encoded else ''
            content = full or plain_text(description.get_text() if description else '')
            date, author = item.find('pubDate'), item.find('creator')
            self.feed_articles[url] = Article(
                url=url, title=title.get_text(' ', strip=True), content=content,
                publish_date=parsedate_to_datetime(date.get_text()).isoformat() if date else None,
                authors=[author.get_text(strip=True)] if author else None,
                source=self.scraper_name.removesuffix(' Scraper'),
                source_content_type='full' if full else 'excerpt',
            )
        return list(self.feed_articles)

    def extract_article(self, url):
        article = self.feed_articles[url]
        if article.content:
            return article
        # Some feeds only list headlines. Use publicly rendered article metadata;
        # do not call rejected private detail APIs or treat challenge text as a body.
        response = requests.get(url, headers=self.headers, timeout=15)
        response.raise_for_status()
        page = BeautifulSoup(response.text, 'html.parser')
        for script in page.select('script[type="application/ld+json"]'):
            try:
                data = json.loads(script.get_text())
            except ValueError:
                continue
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entries_with_graph = entry.get('@graph', [entry])
                for entry in entries_with_graph:
                    if not isinstance(entry, dict):
                        continue
                    kind = entry.get('@type', [])
                    kind = [kind] if isinstance(kind, str) else kind
                    if not any(t in kind for t in ('Article', 'NewsArticle', 'BlogPosting')):
                        continue
                    full = plain_text(entry.get('articleBody'))
                    content = full or plain_text(entry.get('description'))
                    if content:
                        return article.model_copy(update={
                            'content': content,
                            'publish_date': entry.get('datePublished') or article.publish_date,
                            'source_content_type': 'full' if full else 'excerpt',
                        })
        raise ValueError('No public article text or excerpt available')
