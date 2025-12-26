
# Pipeline for fetching articles and saving them in the database
from __future__ import annotations

import time
from datetime import datetime, timezone
from bson import ObjectId

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional


from models import Article
from services.scrapers import (
    BaseScraper,
    MarketWatchScraper,
    YahooScraper,
    DdgScraper,
    NewsApiScraper,
    SeekingAlphaScraper,
    BenzingaNewsScraper,
    France24Scraper,
)
from utils import function_timer, logger, function_timer2
from services.llm import gemini_client
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.database import article_service, pipeline_execution_service
logger.setLevel("DEBUG")

class ArticlePipelineController:
    def __init__(self, article_limit=2, asynchronous=True, llm_summary=True, verify_db=True):
        self.asynchronous = asynchronous
        self.gemini_client = gemini_client
        self.article_db_service = article_service
        self.pipeline_db_service = pipeline_execution_service
        llm = self.gemini_client if llm_summary else None

        self.configs = {
            "limit": article_limit,
            "async_scrape": asynchronous,
            "gemini_client": llm,
            "article_db_service": self.article_db_service,
            "verify_db": verify_db,
        }

        self.yahoo_scraper = YahooScraper(**self.configs)
        self.marketwatch_scraper = MarketWatchScraper(**self.configs)
        self.base_scraper = BaseScraper(**self.configs)
        # self.ddg_scraper = DdgScraper(**self.configs)
        self.newsapi_scraper = NewsApiScraper(**self.configs)
        self.seekingalpha_scraper = SeekingAlphaScraper(**self.configs)
        self.benzinga_scraper = BenzingaNewsScraper(**self.configs)
        self.france24_scraper = France24Scraper(**self.configs)
        self.active_scrapers = [
            self.yahoo_scraper,
            self.marketwatch_scraper,
            # self.newsapi_scraper,
            self.seekingalpha_scraper,
            self.benzinga_scraper,
            # self.france24_scraper,
        ]
    def fetch_latest_news_articles(self):
        articles : list[Article] = []
        if not self.asynchronous:
            for s in self.active_scrapers:
                articles.extend(s.scrape())
        else:
            with ThreadPoolExecutor() as executor:
                future_map = {executor.submit(s.scrape): s for s in self.active_scrapers}
                for fut in as_completed(future_map):
                    s = future_map[fut]
                    try:
                        result = fut.result()
                        if result:
                            articles.extend(result)
                    except Exception as e:
                        logger.exception("Scraper %s failed async: %s", type(s).__name__, e)
        # final_list = [article for article in articles if article.tickers is not None and len(article.tickers)>0]
        sorted_articles = sorted(articles, key=lambda article: article.publish_date or "", reverse=True)
        return sorted_articles

    def chunked(items: List, chunk_size: int) -> Iterable[List]:
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]
    def _process_one(self, article):
        """
        Returns an updated Article model, or None if summarization fails.
        """
        try:
            llm_fields = self.gemini_client.summarize_article(article)
            if not llm_fields:
                return None

            # Merge LLM fields into the existing article (keeps url/title/content/etc.)
            updated = article.model_copy(update=llm_fields)
            return updated

        except Exception as e:
            logger.warning(f"Worker failed for {getattr(article, 'url', 'unknown url')}: {e}")
            return None

    def backfill_articles(
        self,
        batch_size: int = 1000,
        max_workers: int = 8,
        per_call_timeout: Optional[float] = None,  # placeholder if you add timeouts
        limit_total: Optional[int] = None,         # for testing
    ):
        """
        Reads existing articles from DB in batches, runs LLM enrichment concurrently,
        and upserts them back to DB.

        Assumptions:
        - article_db_service.get_articles supports pagination or skip/last_id.
        - article_db_service.upsert_many_articles(...) exists or insert_many does upsert.
        """

        processed_total = 0
        updated_total = 0
        failed_total = 0

        # IMPORTANT: you should paginate by a stable cursor rather than skip for large collections.
        # This is a generic cursor pattern (last_id). Adjust to your DB service.
        last_id = None

        while True:
            if limit_total is not None and processed_total >= limit_total:
                break

            # Fetch next batch. Implement get_articles_batch in your service (see notes below).
            batch = self.article_db_service.get_articles_batch(
                limit=batch_size,
                last_id=last_id,
            )

            if not batch:
                break

            # Advance cursor (assumes Mongo ObjectId or monotonic _id)
            last_id = getattr(batch[-1], "id", None) or getattr(batch[-1], "_id", None) or None

            # Run LLM workers
            updated_batch = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_one, a): a for a in batch}

                for fut in as_completed(futures):
                    original = futures[fut]
                    try:
                        updated = fut.result(timeout=per_call_timeout) if per_call_timeout else fut.result()
                    except Exception as e:
                        logger.warning(f"Worker failed for {getattr(original, 'url', 'unknown url')}: {e}")
                        failed_total += 1
                        continue

                    processed_total += 1
                    if updated is None:
                        failed_total += 1
                        continue

                    updated_total += 1
                    updated_batch.append(updated)

            if updated_batch:
                # You almost certainly want upsert here, not insert_many (or you’ll duplicate).
                # Implement bulk upsert by url (unique) or by _id.
                self.article_db_service.upsert_many_articles(updated_batch)

            logger.info(
                f"Batch done. processed={processed_total} updated={updated_total} failed={failed_total}"
            )

        return {
            "processed": processed_total,
            "updated": updated_total,
            "failed": failed_total,
        }
    
    def run_articles_v2(self):
        articles = self.article_db_service.get_articles(limit=3)
        updated_articles = []
        for article in articles:
            art = self.gemini_client.summarize_article(article)
            # art["url"] = article.url
            art = article.model_copy(update=art)
            if art:
                updated_articles.append(art)
            
            pass
        
        self.article_db_service.insert_many_articles(updated_articles)
        return articles
    @function_timer
    def run(self):
        start_time = time.perf_counter()  # Use perf_counter for high-resolution timing
        pipeline_run_id = ObjectId()
        start_date =  datetime.now(timezone.utc),

        logger.info("Starting article pipeline run with pipeline_run_id: {}".format(
            pipeline_run_id
        ))
        attempted = self.configs["limit"] * len(self.active_scrapers)
        logger.info(f"Attempting to fetch {attempted} articles.")
        
        articles_fetched : list[Article] = self.fetch_latest_news_articles()
        logger.info(f"Total articles fetched: {len(articles_fetched)}")
        insert_result = {}
        
        if articles_fetched:
            insert_result = self.article_db_service.insert_many_articles(articles_fetched, pipeline_run_id=pipeline_run_id)
            logger.info(f"Result: {insert_result}")
        else:
            logger.info("No new articles fetched.")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        run_metadata = {
            "_id": pipeline_run_id,
            "pipeline_run_id": pipeline_run_id,
            "start_time": start_date,
            "end_time": datetime.now(timezone.utc),
            "limit": attempted,
            "duration": elapsed_time,
            "status": "success",
            "config": {"limit": self.configs["limit"], "async_scrape": self.configs["async_scrape"],
                       "verify_db": self.configs["verify_db"]},
            "sources": [
                {
                    "name": s.scraper_name,
                    **s.scrape_metadata,
                    # "logs":s.scrape_metadata.get("logs", []),
                } for s in self.active_scrapers],
            "articles_fetched": len(articles_fetched),
            "articles_inserted": insert_result.get("inserted_count", 0),

        }
        
        self.pipeline_db_service.insert_one_execution(run_metadata)
        logger.info(
            f"Article pipeline run completed with pipeline_run_id: {pipeline_run_id} in {elapsed_time:.2f} seconds."
        )

        now = datetime.now(timezone.utc)
        

def main():
    pipeline = ArticlePipelineController(article_limit=20, verify_db=True)

    # pipeline.run()
    pipeline.backfill_articles(limit_total=2)

if __name__ == "__main__":
    main()