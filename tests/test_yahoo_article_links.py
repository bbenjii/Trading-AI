import ast
from pathlib import Path
from bs4 import BeautifulSoup
import unittest
from urllib.parse import urljoin, urlparse


class FakeBase:
    def __init__(self, **kwargs):
        self.limit = kwargs.get('limit', 2)

    @staticmethod
    def normalize_url(url):
        return url.split('?')[0]

    @staticmethod
    def is_video_url(url):
        return '/video/' in url


class YahooLinksTest(unittest.TestCase):
    def scraper(self, html):
        source = ast.parse(Path('services/scrapers/yahoo_news_scraper.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
            ast.alias(name='annotations')], level=0), cls], type_ignores=[])
        ast.fix_missing_locations(module)
        scope = {'BaseScraper': FakeBase, 'BeautifulSoup': BeautifulSoup,
                 'urljoin': urljoin, 'urlparse': urlparse}
        exec(compile(module, '<isolated Yahoo scraper>', 'exec'), scope)
        scraper = scope['YahooScraper'](limit=2)
        scraper.fetch_html = lambda url: html
        return scraper

    def test_current_yahoo_article_paths_are_discovered_and_deduplicated(self):
        html = '''<main>
        <a class="subtle-link titles" href="/markets/stocks/articles/company-earnings-123.html">Company earnings</a>
        <a class="subtle-link titles" href="/markets/stocks/articles/company-earnings-123.html?src=feed">Duplicate</a>
        <a class="subtle-link titles" href="/technology/ai/articles/company-expansion-456.html">Expansion</a>
        <a href="https://example.com/articles/unrelated.html">External link</a>
        </main>'''
        links = self.scraper(html).get_article_links()
        self.assertEqual(links, [
            'https://finance.yahoo.com/markets/stocks/articles/company-earnings-123.html',
            'https://finance.yahoo.com/technology/ai/articles/company-expansion-456.html'])

    def test_existing_topic_paths_still_work(self):
        html = '<div data-testid="topic-stream"><a role="article" href="/news/company-news.html">News</a></div>'
        self.assertEqual(self.scraper(html).get_article_links(),
                         ['https://finance.yahoo.com/news/company-news.html'])
