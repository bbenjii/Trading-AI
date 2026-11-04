import unittest
from datetime import datetime, timezone
from unittest.mock import Mock

import mongomock

from models import Article
from services.article_enrichment import (
    ArticleEnricher, backfill_missing_summaries, missing_summary_filter,
)


class ProviderError(Exception):
    def __init__(self, code):
        self.code = code


class ArticleEnrichmentTest(unittest.TestCase):
    def setUp(self):
        self.client = Mock(summary_model="test-model")
        self.sleep = Mock()
        self.enricher = ArticleEnricher(self.client, sleep=self.sleep)
        self.article = Article(url='https://example.com/news', title='AAPL news',
                               content='Apple announced a new product.', tickers=['AAPL'])

    def test_ticker_only_article_still_needs_enrichment(self):
        self.assertTrue(self.enricher.needs_enrichment(self.article))
        for summary in ['', '  ', None]:
            self.assertTrue(self.enricher.needs_enrichment(
                self.article.model_copy(update={'summary_short': summary})))

    def test_valid_summary_does_not_repeat_request(self):
        article = self.article.model_copy(update={'summary_short': 'Existing summary.'})
        updated = self.enricher.enrich(article)
        self.client.summarize_article.assert_not_called()
        self.assertEqual(updated.enrichment_status, 'complete')
        self.assertEqual(updated.enrichment_attempts, 0)

    def test_retries_transient_failure_and_preserves_existing_values(self):
        self.client.summarize_article.side_effect = [ProviderError(429), {
            'summary_short': 'Apple announced a product.', 'tickers': [],
            'summary_extended': None, 'sentiment': 'positive', 'title': 'Overwrite'}]
        updated = self.enricher.enrich(self.article)
        self.assertEqual(updated.enrichment_status, 'complete')
        self.assertEqual(updated.enrichment_attempts, 2)
        self.assertIsNone(updated.enrichment_last_error)
        self.assertEqual(updated.tickers, ['AAPL'])
        self.assertEqual(updated.title, self.article.title)
        self.assertEqual(updated.content, self.article.content)
        self.assertEqual(updated.sentiment, 'positive')
        self.sleep.assert_called_once_with(2)
        self.assertIsNotNone(updated.enrichment_attempted_at)

    def test_terminal_error_is_not_retried_or_logged_with_sensitive_message(self):
        error = ProviderError(403)
        error.args = ('secret-token-in-error',)
        self.client.summarize_article.side_effect = error
        updated = self.enricher.enrich(self.article)
        self.assertEqual(updated.enrichment_status, 'failed')
        self.assertEqual(updated.enrichment_attempts, 1)
        self.assertNotIn('secret-token', updated.enrichment_last_error)
        self.assertFalse(updated.enrichment_retryable)
        self.sleep.assert_not_called()

    def test_empty_or_ticker_only_results_never_count_as_success(self):
        self.client.summarize_article.side_effect = [None, {'tickers': ['AAPL']}, {}]
        updated = self.enricher.enrich(self.article)
        self.assertEqual(updated.enrichment_status, 'failed')
        self.assertEqual(updated.enrichment_attempts, 3)
        self.assertEqual(self.sleep.call_args_list[0].args, (2,))
        self.assertEqual(self.sleep.call_args_list[1].args, (4,))
        self.assertEqual(updated.tickers, ['AAPL'])

    def test_later_run_can_recover_and_accumulates_attempt_count(self):
        article = self.article.model_copy(update={
            'enrichment_status': 'failed', 'enrichment_attempts': 3,
            'enrichment_last_error': 'TimeoutError'})
        self.client.summarize_article.return_value = {'summary_short': 'Recovered.'}
        updated = self.enricher.enrich(article)
        self.assertEqual(updated.enrichment_attempts, 4)
        self.assertEqual(updated.enrichment_status, 'complete')
        self.assertIsNone(updated.enrichment_last_error)

    def test_no_client_marks_failure_without_attempt(self):
        updated = ArticleEnricher(None).enrich(self.article)
        self.assertEqual(updated.enrichment_status, 'failed')
        self.assertEqual(updated.enrichment_attempts, 0)


    def test_attempt_callback_records_pending_before_each_provider_call(self):
        observed = []
        self.client.summarize_article.side_effect = [TimeoutError(), {'summary_short': 'Recovered.'}]
        updated = self.enricher.enrich(self.article, on_attempt=lambda a: observed.append(
            (a.enrichment_status, a.enrichment_attempts)))
        self.assertEqual(observed, [('pending', 1), ('pending', 2)])
        self.assertEqual(updated.enrichment_status, 'complete')


class BackfillTest(unittest.TestCase):
    def setUp(self):
        self.collection = mongomock.MongoClient().dev.articles
        self.client = Mock(summary_model="test-model")
        self.client.summarize_article.return_value = {
            'summary_short': 'New summary.', 'tickers': [], 'sentiment': 'positive'}
        self.enricher = ArticleEnricher(self.client, sleep=Mock())
        for i, summary in enumerate([None, '', ' \n', 'Existing summary.']):
            doc = {'url': f'https://example.com/{i}', 'title': 'Title',
                   'content': 'Body', 'tickers': ['AAPL'], 'summary_short': summary,
                   'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc)}
            if i == 0:
                del doc['summary_short']
            self.collection.insert_one(doc)

    def test_preview_counts_missing_null_and_whitespace_without_calls_or_writes(self):
        before = list(self.collection.find())
        result = backfill_missing_summaries(self.collection, self.enricher, dry_run=True)
        self.assertEqual(result['candidates'], 3)
        self.assertEqual(list(self.collection.find()), before)
        self.client.summarize_article.assert_not_called()

    def test_backfill_updates_missing_fields_and_preserves_content_dates_and_tickers(self):
        result = backfill_missing_summaries(self.collection, self.enricher, batch_size=2)
        self.assertEqual(result['completed'], 3)
        self.assertEqual(result['failed'], 0)
        self.assertEqual(self.collection.count_documents(missing_summary_filter()), 0)
        doc = self.collection.find_one({'url': 'https://example.com/0'})
        self.assertEqual(doc['tickers'], ['AAPL'])
        self.assertEqual(doc['content'], 'Body')
        self.assertEqual(doc['created_at'], datetime(2026, 1, 1))
        self.assertEqual(doc['enrichment_status'], 'complete')
        self.assertEqual(doc['enrichment_attempts'], 1)
        self.assertIsNone(doc['enrichment_last_error'])
        self.assertEqual(self.collection.find_one({'url': 'https://example.com/3'})[
            'summary_short'], 'Existing summary.')
        again = backfill_missing_summaries(self.collection, self.enricher)
        self.assertEqual(again['processed'], 0)
        self.assertEqual(self.client.summarize_article.call_count, 3)

    def test_terminal_provider_failure_is_persisted_and_halts_backfill(self):
        self.client.summarize_article.side_effect = ProviderError(404)
        result = backfill_missing_summaries(self.collection, self.enricher, batch_size=1)
        self.assertEqual(result['failed'], 1)
        self.assertTrue(result['halted'])
        doc = self.collection.find_one({'enrichment_status': 'failed'})
        self.assertEqual(doc['enrichment_attempts'], 1)
        self.assertIsNone(doc.get('summary_short'))

    def test_limit_is_enforced_and_resume_finishes_remaining(self):
        result = backfill_missing_summaries(self.collection, self.enricher, limit_total=1)
        self.assertEqual(result['processed'], 1)
        self.assertEqual(self.collection.count_documents(missing_summary_filter()), 2)
        resumed = backfill_missing_summaries(self.collection, self.enricher)
        self.assertEqual(resumed['completed'], 2)


    def test_backfill_does_not_overwrite_summary_completed_by_another_writer(self):
        def concurrent_update(article, **kwargs):
            self.collection.update_one({'url': article.url}, {'$set': {
                'summary_short': 'Another worker finished.', 'enrichment_status': 'complete'}})
            return {'summary_short': 'Our result.'}
        self.client.summarize_article.side_effect = concurrent_update
        result = backfill_missing_summaries(self.collection, self.enricher, limit_total=1)
        self.assertEqual(result['skipped'], 1)
        self.assertEqual(self.collection.find_one({'url': 'https://example.com/0'})[
            'summary_short'], 'Another worker finished.')

    def test_repeated_transient_failures_stop_after_three_articles(self):
        self.client.summarize_article.side_effect = ProviderError(503)
        result = backfill_missing_summaries(self.collection, self.enricher, workers=1)
        self.assertEqual(result['failed'], 3)
        self.assertEqual(self.client.summarize_article.call_count, 9)
        self.assertTrue(result['halted'])
        self.assertEqual(self.collection.count_documents({'enrichment_status': 'failed'}), 3)


    def test_explicit_article_scope_leaves_older_missing_summaries_untouched(self):
        selected = list(self.collection.find().sort('_id', -1).limit(2))
        ids = [doc['_id'] for doc in selected]
        result = backfill_missing_summaries(self.collection, self.enricher, article_ids=ids)
        self.assertEqual(result['candidates'], 1)
        self.assertEqual(result['completed'], 1)
        self.assertEqual(self.client.summarize_article.call_count, 1)
        self.assertEqual(self.collection.count_documents(missing_summary_filter()), 2)
        empty = backfill_missing_summaries(self.collection, self.enricher, article_ids=[])
        self.assertEqual(empty['processed'], 0)


class GeminiErrorPropagationTest(unittest.TestCase):
    def test_provider_errors_reach_retry_handler_only_when_requested(self):
        from services.llm.gemini_service import GeminiService
        from unittest.mock import patch
        with patch('services.llm.gemini_service.genai.Client') as factory:
            factory.return_value.models.generate_content.side_effect = ProviderError(429)
            service = GeminiService('test-key')
            with self.assertRaises(ProviderError):
                service.summarize_article(Article(url='https://example.com', title='Title',
                                                  content='Body'), raise_errors=True)
            self.assertIsNone(service.send_request({'task': 'test'}))
            self.assertEqual(service.summary_model, 'gemini-3.5-flash-lite')


class PipelineWiringTest(unittest.TestCase):
    @staticmethod
    def controller_class():
        # Load only the controller definition: normal module imports create live DB/index clients.
        import ast
        from pathlib import Path
        from services.article_enrichment import backfill_missing_summaries
        source = ast.parse(Path('pipelines/article_pipeline_controller.py').read_text())
        definition = next(n for n in source.body if isinstance(n, ast.ClassDef)
                          and n.name == 'ArticlePipelineController')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
            ast.alias(name='annotations')], level=0), definition], type_ignores=[])
        ast.fix_missing_locations(module)
        scope = {'ArticleEnricher': ArticleEnricher, 'logger': Mock(),
                 'backfill_missing_summaries': backfill_missing_summaries,
                 'function_timer': lambda fn: fn}
        for name in ['gemini_client', 'article_service', 'pipeline_execution_service',
                     'catalyst_cluster_service', 'YahooScraper', 'MarketWatchScraper',
                     'BaseScraper', 'NewsApiScraper', 'SeekingAlphaScraper',
                     'BenzingaNewsScraper', 'France24Scraper']:
            scope[name] = Mock()
        exec(compile(module, '<isolated pipeline>', 'exec'), scope)
        return scope['ArticlePipelineController']

    def test_ingestion_uses_shared_enrichment_and_persistable_metadata(self):
        cls = self.controller_class()
        controller = cls.__new__(cls)
        controller.gemini_client = Mock(summary_model='test-model')
        controller.gemini_client.summarize_article.return_value = {'summary_short': 'Summary.'}
        article = Article(url='https://example.com', title='Title', content='Body', tickers=['AAPL'])
        self.assertTrue(controller._needs_enrichment(article))
        updated = controller.enrich_articles([article])[0]
        self.assertEqual(updated.model_dump()['enrichment_status'], 'complete')
        self.assertEqual(updated.model_dump()['enrichment_attempts'], 1)
        self.assertEqual(updated.tickers, ['AAPL'])

    def test_existing_backfill_entrypoint_uses_safe_dry_run(self):
        cls = self.controller_class()
        controller = cls.__new__(cls)
        controller.gemini_client = Mock(summary_model='test-model')
        controller.article_db_service = Mock()
        controller.article_db_service.collection = mongomock.MongoClient().dev.articles
        controller.article_db_service.collection.insert_one({
            'url': 'https://example.com', 'title': 'Title', 'content': 'Body',
            'summary_short': None, 'tickers': ['AAPL'],
            'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc)})
        result = controller.backfill_missing_enrichment_since(
            datetime(2025, 1, 1, tzinfo=timezone.utc), dry_run=True)
        self.assertEqual(result['candidates'], 1)
        controller.gemini_client.summarize_article.assert_not_called()

    def test_scrapers_leave_ai_calls_to_tracked_pipeline_stage(self):
        cls = self.controller_class()
        controller = cls()
        self.assertIsNone(controller.configs['gemini_client'])
        self.assertIsNotNone(controller.gemini_client)


if __name__ == '__main__':
    unittest.main()
