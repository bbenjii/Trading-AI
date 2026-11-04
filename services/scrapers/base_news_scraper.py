# base_scraper.py
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed

import re
import time
from typing import List, Optional, Dict, Iterable, Tuple
from urllib.parse import urlsplit, urlunsplit

import newspaper
import requests
from newspaper.network import session

from bs4 import BeautifulSoup
from pydantic import BaseModel

from models import Article

from utils.logger import logger

class BaseScraper:
    """
    A reusable base for news scrapers.
    Subclasses should implement `get_article_links` and can optionally override `extract_article`.
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(
        self,
        limit: int = 10,
        max_workers: int = 6,
        async_scrape: bool = False,        
        base_url: Optional[str] = None,
        scraper_name: Optional[str] = None,
        gemini_client = None,
        headers: Optional[Dict[str, str]] = None,
        article_db_service = None,
        verify_db = False,
    ):
        self.gemini_client = gemini_client
        self.scraper_name = scraper_name
        self.limit = limit
        self.max_workers = max_workers
        self.async_scrape = async_scrape
        self.headers = {**(headers or self.DEFAULT_HEADERS)}
        self.base_url = base_url  # Optional hint for robots and link building
        self.article_db_service = article_db_service
        self.verify_db = verify_db
        self.scrape_metadata = {}
    
    def _log(self, msg: str):
        self.scrape_metadata.setdefault("logs", []).append(msg)
    
    def scrape(self, urls=None, manual_fetch=False, no_limit=False) -> List[Article]:
        started = time.perf_counter()
        self.scrape_metadata = {
            "status": "failed", "fetched_count": 0, "scraped_count": 0,
            "duplicate_count": 0, "extraction_failed_count": 0,
        }
        try:
            links = urls if manual_fetch and urls else self.get_article_links()
            self.scrape_metadata["fetched_count"] = len(links)
            if not links:
                empty_feed = (self.scrape_metadata.get("feed_valid")
                              and self.scrape_metadata.get("feed_item_count") == 0)
                self.scrape_metadata["status"] = "no_new" if empty_feed else "failed"
                if not empty_feed:
                    self.scrape_metadata["error"] = "No usable article links discovered"
                return []

            # Check all discovered candidates before applying the per-run extraction limit.
            candidates, seen = [], set()
            for link in links:
                url = self.normalize_url(link)
                if url in seen:
                    continue
                seen.add(url)
                if (self.verify_db and self.article_db_service
                        and self.article_db_service.url_exists_in_db(url)):
                    self.scrape_metadata["duplicate_count"] += 1
                    continue
                candidates.append(url)
            limit = len(candidates) if no_limit else self.limit
            selected = candidates[:limit]
            self.scrape_metadata["attempted_count"] = len(selected)
            if not selected:
                self.scrape_metadata["status"] = "no_new"
                return []

            if self.async_scrape:
                articles = self._extract_many_threaded(selected, limit=limit)
            else:
                articles = []
                for link in selected:
                    try:
                        article = self.extract_article(link)
                        if article and article.content.strip():
                            articles.append(article)
                    except Exception as exc:
                        self._log(f"Extraction failed: {type(exc).__name__}")
            self.scrape_metadata["scraped_count"] = len(articles)
            failed = len(selected) - len(articles)
            self.scrape_metadata["extraction_failed_count"] = failed
            self.scrape_metadata["status"] = (
                "success" if not failed else "partial" if articles else "failed"
            )
            return articles
        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            self.scrape_metadata["error"] = type(exc).__name__ + (
                f" (HTTP {code})" if code else ""
            )
            self._log(self.scrape_metadata["error"])
            return []
        finally:
            self.scrape_metadata["duration_sec"] = round(time.perf_counter() - started, 3)
            logger.info("%s: %s", self.scraper_name, self.scrape_metadata)

    # ---------- Methods for subclasses to implement or override ----------

    def get_article_links(self) -> List[str]:
        """
        Return a list of absolute article URLs. The base class enforces limit.
        Subclasses must implement this.
        """
        raise NotImplementedError

    def extract_article(self, url: str) -> Optional[Article]:
        """
        Extract one article. You can override this to use site-specific selectors.
        The default calls `extract_article_default` which tries common patterns.
        """
        return self.extract_article_default(url)

    # ---------- Helpful utilities for subclasses ----------

    # HTML fetch with retries, timeouts, robots, and per-host delay
    def fetch_html(self, url: str) -> Optional[str]:

        try:
            r = session.get(url, headers=self.headers, timeout=15)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            self._log(f"Request failed for {url}: {e}")
            return None

    @staticmethod
    def normalize_url(url: str) -> str:
        parts = urlsplit(url)
        # Strip query and fragment
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    @staticmethod
    def soup(html: str) -> BeautifulSoup:
        # lxml is faster if installed, html.parser is fine too
        return BeautifulSoup(html, "lxml")

    @staticmethod
    def is_video_url(url: str) -> bool:
        return "/video/" in url or "/videos/" in url

    # ---------- Default extractor (override for per-site logic) ----------

    def extract_article_default(self, url: str) -> Optional[Article]:
        html = self.fetch_html(url)
        if not html:
            return None

        news_article = newspaper.article(url=url, input_html=html)

        news_article.nlp()
        content = news_article.text.strip()
        single_line = re.sub(r"\s+", " ", content).strip()
        content = single_line
        title = news_article.title
        publish_date = news_article.publish_date
        authors = news_article.authors

        article = Article(
            url=self.normalize_url(url),
            title=title or "",
            content=content or "",
            publish_date=str(publish_date) if publish_date else None,
            authors=authors or None,
            summary=None,
        )

        try:
            update = self.gemini_client.summarize_article(article)
            if update:
                article = article.model_copy(update=update)
        except Exception as e:
            article.summary = news_article.summary

        return article

    # ---------- Internals ----------

    def _extract_many_threaded(self, links: List[str], limit=None) -> List[Article]:
        if limit is None:
            limit = self.limit
            
        links = links[: limit]
        out_by_index: List[Optional[Article]] = [None] * len(links)
        
        def extract(url):
            return self.extract_article(url)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            fut_map = {pool.submit(extract, link): (i, link) for i, link in enumerate(links)}
            for fut in as_completed(fut_map):
                i, link = fut_map[fut]
                try:
                    art = fut.result()
                    if art and art.content:
                        out_by_index[i] = art
                except Exception as e:
                    logger.warning(f"Worker failed for {link}: {e}")
                    self._log(f"Worker failed for {link}: {e}")
        return [a for a in out_by_index if a]

if __name__ == "__main__":
    scraper = BaseScraper()
    url = "https://www.benzinga.com/analyst-stock-ratings/analyst-color/25/11/48845509/ai-bubble-nasdaq-100-data-centers-us-market-coreweave-outlook-stocks-analysis"

    articles = scraper.scrape(urls=[url], manual_fetch=True)
    for art in articles:
        print(f"Title: {art.title}\nURL: {art.url}\nPublish Date: {art.publish_date}\nAuthors: {art.authors}\n Content: {art.content}")
    pass
    
    