import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests
from models import Article
from services.scrapers.base_news_scraper import BaseScraper
from services.scrapers.yahoo_news_scraper import YahooScraper
from services.scrapers.marketwatch_news_scraper import MarketWatchScraper
from services.scrapers.seekingalpha_news_scraper import SeekingAlphaScraper
from services.scrapers.benzinga_news_scraper import BenzingaNewsScraper


def response(text, status=200):
    result = Mock(text=text, status_code=status)
    if status >= 400:
        result.raise_for_status.side_effect = requests.HTTPError(response=result)
    return result


class IngestionTest(unittest.TestCase):
    def setUp(self):
        # Legacy code uses Session.get; prevent any real HTTP while proving the regressions.
        legacy_http = patch('requests.sessions.Session.get', return_value=response('', 403))
        legacy_http.start()
        self.addCleanup(legacy_http.stop)

    def test_yahoo_current_article_links_exclude_navigation_and_duplicates(self):
        scraper = YahooScraper(limit=1)
        scraper.fetch_html = Mock(return_value='''<a href="https://www.yahoo.com/news/">Nav</a>
            <a href="https://finance.yahoo.com/news/first-123.html?tracking=1">One</a>
            <a href="https://finance.yahoo.com/news/first-123.html">Same</a>
            <a href="/news/second-456.html">Two</a>
            <a href="https://finance.yahoo.com/news/">Section navigation</a>
            <a href="/articles/third-789.html">Three</a>''')
        self.assertEqual(scraper.get_article_links(), [
            'https://finance.yahoo.com/news/first-123.html',
            'https://finance.yahoo.com/news/second-456.html',
            'https://finance.yahoo.com/articles/third-789.html'])

    def test_duplicate_is_skipped_before_limit_in_both_execution_modes(self):
        for asynchronous in [False, True]:
            scraper = BaseScraper(limit=1, async_scrape=asynchronous, verify_db=True,
                                  article_db_service=Mock())
            scraper.get_article_links = Mock(return_value=[
                'https://example.com/old?tracking=1', 'https://example.com/new'])
            scraper.article_db_service.url_exists_in_db.side_effect = lambda url: url.endswith('/old')
            scraper.extract_article = Mock(side_effect=lambda url: Article(
                url=url, title='New story', content='A newly published story.'))
            result = scraper.scrape()
            self.assertEqual([a.url for a in result], ['https://example.com/new'])
            self.assertEqual(scraper.scrape_metadata['status'], 'success')
            self.assertEqual(scraper.scrape_metadata['duplicate_count'], 1)

    def test_already_saved_articles_are_idle_instead_of_failed(self):
        scraper = BaseScraper(limit=1, verify_db=True, article_db_service=Mock())
        scraper.get_article_links = Mock(return_value=['https://example.com/old'])
        scraper.article_db_service.url_exists_in_db.return_value = True
        scraper.extract_article = Mock()
        self.assertEqual(scraper.scrape(), [])
        self.assertEqual(scraper.scrape_metadata['status'], 'no_new')
        scraper.extract_article.assert_not_called()

    def test_network_failure_is_recorded_as_failed_source(self):
        scraper = BaseScraper()
        scraper.get_article_links = Mock(side_effect=requests.HTTPError())
        self.assertEqual(scraper.scrape(), [])
        self.assertEqual(scraper.scrape_metadata['status'], 'failed')

    def test_extraction_failure_is_recorded_instead_of_silent_zero(self):
        scraper = BaseScraper(limit=1, async_scrape=True)
        scraper.get_article_links = Mock(return_value=['https://example.com/news'])
        scraper.extract_article = Mock(return_value=None)
        self.assertEqual(scraper.scrape(), [])
        self.assertEqual(scraper.scrape_metadata['status'], 'failed')
        self.assertEqual(scraper.scrape_metadata['extraction_failed_count'], 1)

    def test_feed_excerpt_is_cleaned_and_normalized_without_fetching_full_page(self):
        scraper = MarketWatchScraper(limit=1)
        xml = '''<rss xmlns:dc="http://purl.org/dc/elements/1.1/"><channel><item>
            <title>Company update</title><link>https://www.marketwatch.com/story/update?mod=rss</link>
            <description><![CDATA[<p>Sales rose <b>15%</b>.</p>]]></description>
            <pubDate>Fri, 23 Oct 2026 12:00:00 GMT</pubDate><dc:creator>Author</dc:creator>
            </item></channel></rss>'''
        with patch('requests.get', return_value=response(xml)) as get:
            result = scraper.scrape()
        self.assertEqual(get.call_count, 1)
        self.assertEqual(result[0].url, 'https://www.marketwatch.com/story/update')
        self.assertEqual(result[0].content, 'Sales rose 15%.')
        self.assertEqual(result[0].source_content_type, 'excerpt')
        self.assertEqual(result[0].authors, ['Author'])
        self.assertIn('2026-10-23', result[0].publish_date)

    def test_benzinga_public_feed_full_body_is_used(self):
        scraper = BenzingaNewsScraper(limit=1)
        xml = '''<rss xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel>
          <item><title>Update</title><link>https://www.benzinga.com/news/update</link>
          <description>Preview</description><content:encoded><![CDATA[
          <p>Complete publisher-provided body.</p><script>ignored()</script>
          ]]></content:encoded></item></channel></rss>'''
        with patch('requests.get', return_value=response(xml)):
            result = scraper.scrape()
        self.assertEqual(result[0].content, 'Complete publisher-provided body.')
        self.assertEqual(result[0].source_content_type, 'full')

    def test_seekingalpha_uses_public_metadata_excerpt_not_challenge_text(self):
        scraper = SeekingAlphaScraper(limit=1)
        xml = '''<rss><channel><item><title>Sales rise</title>
          <link>https://seekingalpha.com/news/123-sales-rise</link></item></channel></rss>'''
        page = '''<html><script type="application/ld+json">{
          "@type":"NewsArticle","headline":"Sales rise", "datePublished":"2026-10-23",
          "description":"Company sales increased by 15%."}</script>
          <footer>Please enable JavaScript and cookies.</footer></html>'''
        with patch('requests.get', side_effect=[response(xml), response(page)]):
            result = scraper.scrape()
        self.assertEqual(result[0].content, 'Company sales increased by 15%.')
        self.assertEqual(result[0].source_content_type, 'excerpt')

    def test_malformed_feed_does_not_become_success(self):
        scraper = MarketWatchScraper(limit=1)
        with patch('requests.get', return_value=response('<html>Access denied</html>')):
            self.assertEqual(scraper.scrape(), [])
        self.assertEqual(scraper.scrape_metadata['status'], 'failed')

    def test_discovery_ignores_links_outside_publisher(self):
        scraper = MarketWatchScraper(limit=1)
        xml = '''<rss><channel><item><title>Bad</title><link>http://127.0.0.1/private</link>
          <description>Do not follow</description></item></channel></rss>'''
        with patch('requests.get', return_value=response(xml)) as get:
            self.assertEqual(scraper.scrape(), [])
        self.assertEqual(get.call_count, 1)


class PipelineHealthTest(unittest.TestCase):
    def controller(self, statuses, articles=None):
        from test_article_enrichment import PipelineWiringTest
        from bson import ObjectId
        import time
        from datetime import datetime, timezone
        cls = PipelineWiringTest.controller_class()
        cls.run.__globals__.update(ObjectId=ObjectId, time=time, datetime=datetime,
                                   timezone=timezone)
        pipeline = cls.__new__(cls)
        pipeline.configs = {'limit': 1, 'async_scrape': True, 'verify_db': True}
        pipeline.active_scrapers = [Mock(scraper_name=str(i), scrape_metadata={'status': s})
                                    for i, s in enumerate(statuses)]
        pipeline.fetch_latest_news_articles = Mock(return_value=articles or [])
        pipeline.enrich_articles = lambda items: items
        pipeline.pipeline_db_service = Mock()
        pipeline.article_db_service = Mock()
        pipeline.article_db_service.insert_many_articles.return_value = {
            'inserted_count': len(articles or []), 'inserted_ids': {}}
        return pipeline

    def test_all_source_failures_are_persisted_as_failed_and_returned(self):
        pipeline = self.controller(['failed', 'failed'])
        result = pipeline.run()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failed_sources'], 2)
        self.assertEqual(pipeline.pipeline_db_service.insert_one_execution.call_args.args[0][
            'status'], 'failed')

    def test_duplicates_only_run_is_no_new_instead_of_failure(self):
        result = self.controller(['no_new', 'no_new']).run()
        self.assertEqual(result['status'], 'no_new')
        self.assertEqual(result['articles_inserted'], 0)

    def test_mixed_source_failure_is_partial_even_if_other_source_has_no_new(self):
        result = self.controller(['failed', 'no_new']).run()
        self.assertEqual(result['status'], 'partial')

    def test_enrichment_failure_prevents_success_status(self):
        article = Article(url='https://example.com', title='Title', content='Body',
                          enrichment_status='failed')
        result = self.controller(['success'], articles=[article]).run()
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['enrichment']['failed'], 1)

    def test_successful_new_article_run_reports_inserted_count(self):
        article = Article(url='https://example.com', title='Title', content='Body',
                          summary_short='Summary.', enrichment_status='complete')
        result = self.controller(['success'], articles=[article]).run()
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['articles_inserted'], 1)

    def test_racing_duplicate_insert_is_no_new(self):
        article = Article(url='https://example.com', title='Title', content='Body',
                          summary_short='Summary.', enrichment_status='complete')
        pipeline = self.controller(['success'], articles=[article])
        pipeline.article_db_service.insert_many_articles.return_value = {
            'inserted_count': 0, 'inserted_ids': {}}
        self.assertEqual(pipeline.run()['status'], 'no_new')

    def test_handler_returns_503_for_failed_run(self):
        source = ast.parse(Path('main_cloud_functions.py').read_text())
        definition = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                          and n.name == 'run_article_fetch_pipeline')
        pipeline = Mock()
        pipeline.run.return_value = {'status': 'failed', 'articles_fetched': 0,
                                     'articles_inserted': 0, 'failed_sources': 4}
        scope = {'pipeline': pipeline, 'logger': Mock()}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), '<handler>', 'exec'), scope)
        body, status = scope['run_article_fetch_pipeline'](None)
        self.assertEqual(status, 503)
        self.assertEqual(body['status'], 'failed')


class ExcerptSummaryTest(unittest.TestCase):
    def test_enrichment_receives_excerpt_context_and_allows_shorter_summary(self):
        from services.llm.gemini_service import GeminiService
        from models import LlmSummary
        with patch('services.llm.gemini_service.genai.Client'):
            service = GeminiService('test-key')
        service.send_request = Mock(return_value=LlmSummary(summary_short='Sales increased.'))
        article = Article(url='https://example.com', title='Sales increase',
                          content='Sales increased 15%.', source_content_type='excerpt')
        service.summarize_article(article, raise_errors=True)
        import json
        payload = json.loads(service.send_request.call_args.kwargs['prompt_data'])
        self.assertEqual(payload['article']['source_content_type'], 'excerpt')
        self.assertTrue(any('do not infer missing details' in rule
                            for rule in payload['requirements']))


if __name__ == '__main__':
    unittest.main()
