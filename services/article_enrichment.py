"""Shared enrichment and resumable backfill without scraper or database startup effects."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import time

from models import Article, LlmSummary


ENRICHMENT_FIELDS = set(LlmSummary.model_fields) | {
    'ticker_sentiments', 'ticker_sentiment_reasoning', 'keyword_map',
}
STATUS_FIELDS = {
    'enrichment_status', 'enrichment_attempts', 'enrichment_last_error',
    'enrichment_attempted_at', 'enrichment_model', 'enrichment_retryable',
}


def missing_summary_filter():
    # Equality to None includes both missing and explicitly null fields in MongoDB.
    return {'$or': [{'summary_short': None}, {'summary_short': {'$regex': r'^\s*$'}}]}


def enrichment_updates(article):
    fields = article.model_dump()
    return {
        key: value for key, value in fields.items()
        if key in STATUS_FIELDS or (
            key in ENRICHMENT_FIELDS and value not in (None, '', [], {})
        )
    }


class ArticleEnricher:
    MAX_ATTEMPTS = 3

    def __init__(self, client, sleep=time.sleep):
        self.client = client
        self.sleep = sleep

    @staticmethod
    def needs_enrichment(article):
        return not (article.summary_short or '').strip()

    def enrich(self, article, on_attempt=None):
        metadata = {
            'enrichment_status': 'complete',
            'enrichment_last_error': None,
            'enrichment_retryable': True,
        }
        if not self.needs_enrichment(article):
            return article.model_copy(update=metadata)

        metadata['enrichment_status'] = 'pending'
        metadata['enrichment_attempts'] = article.enrichment_attempts
        metadata['enrichment_model'] = getattr(self.client, 'summary_model', None)
        if self.client is None:
            metadata.update(enrichment_status='failed',
                            enrichment_last_error='Gemini client unavailable',
                            enrichment_retryable=False)
            return article.model_copy(update=metadata)

        for attempt in range(self.MAX_ATTEMPTS):
            metadata['enrichment_attempts'] += 1
            metadata['enrichment_attempted_at'] = datetime.now(timezone.utc)
            metadata['enrichment_status'] = 'pending'
            if on_attempt:
                on_attempt(article.model_copy(update=metadata))
            try:
                fields = self.client.summarize_article(article, raise_errors=True)
                summary = (fields or {}).get('summary_short')
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError('Missing usable short summary')
                # Only fill missing enrichment fields; never replace source article data.
                updates = {
                    key: value for key, value in fields.items()
                    if key in ENRICHMENT_FIELDS
                    and value not in (None, '', [], {})
                    and (getattr(article, key, None) in (None, '', [], {})
                         or (key == 'summary_short' and self.needs_enrichment(article)))
                }
                updated = Article.model_validate({**article.model_dump(), **updates})
                metadata.update(enrichment_status='complete', enrichment_last_error=None,
                                enrichment_retryable=True)
                return updated.model_copy(update=metadata)
            except Exception as exc:
                code = getattr(exc, 'code', None) or getattr(exc, 'status_code', None)
                retryable = (
                    code in (408, 429, 500, 502, 503, 504)
                    or isinstance(exc, (TimeoutError, ConnectionError, ValueError))
                    or type(exc).__name__ in ('ReadTimeout', 'ConnectTimeout', 'ConnectError')
                )
                # Provider messages may contain request data or credentials. Keep only type/code.
                metadata.update(
                    enrichment_status='failed', enrichment_retryable=retryable,
                    enrichment_last_error=type(exc).__name__ + (f' (HTTP {code})' if code else ''),
                )
                if not retryable or attempt == self.MAX_ATTEMPTS - 1:
                    break
                self.sleep(2 ** (attempt + 1))
        return article.model_copy(update=metadata)


def backfill_missing_summaries(collection, enricher, *, batch_size=25, limit_total=None,
                               dry_run=False, workers=1, since=None, progress=None,
                               article_ids=None):
    if batch_size < 1 or workers < 1 or (limit_total is not None and limit_total < 1):
        raise ValueError('Batch size, workers, and limit must be positive')
    query = missing_summary_filter()
    if article_ids is not None:
        query = {'$and': [query, {'_id': {'$in': list(article_ids)}}]}
    if since is not None:
        query = {'$and': [query, {'created_at': {'$gte': since}}]}
    result = {'candidates': collection.count_documents(query), 'processed': 0,
              'completed': 0, 'failed': 0, 'skipped': 0, 'dry_run': dry_run, 'halted': False}
    if dry_run:
        return result

    # Snapshot the upper bound, so concurrent ingestion cannot extend this run indefinitely.
    latest = collection.find_one(query, {'_id': 1}, sort=[('_id', -1)])
    if latest is None:
        return result
    last_id = None
    consecutive_failures = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while limit_total is None or result['processed'] < limit_total:
            bounds = {'$lte': latest['_id']}
            if last_id is not None:
                bounds['$gt'] = last_id
            page_query = {'$and': [query, {'_id': bounds}]}
            count = batch_size if limit_total is None else min(
                batch_size, limit_total - result['processed'])
            docs = list(collection.find(page_query, {'embedding': 0}).sort('_id', 1).limit(count))
            if not docs:
                break
            last_id = docs[-1]['_id']
            # Schedule at most one request per worker at once, including on provider failure.
            for offset in range(0, len(docs), workers):
                group = docs[offset:offset + workers]

                def enrich_doc(doc):
                    def record_attempt(article):
                        collection.update_one(
                            {'$and': [{'_id': doc['_id']}, missing_summary_filter()]},
                            {'$set': {key: value for key, value in
                                      enrichment_updates(article).items() if key in STATUS_FIELDS}},
                        )
                    return enricher.enrich(Article(**doc), on_attempt=record_attempt)

                for doc, updated in zip(group, executor.map(enrich_doc, group)):
                    result['processed'] += 1
                    write = collection.update_one(
                        {'$and': [{'_id': doc['_id']}, missing_summary_filter()]},
                        {'$set': enrichment_updates(updated)},
                    )
                    if not write.matched_count:
                        result['skipped'] += 1
                        continue
                    if updated.enrichment_status == 'complete':
                        result['completed'] += 1
                        consecutive_failures = 0
                    else:
                        result['failed'] += 1
                        consecutive_failures += 1
                        if not updated.enrichment_retryable or consecutive_failures >= 3:
                            result['halted'] = True
                if progress:
                    progress(dict(result))
                if result['halted']:
                    return result
    return result
