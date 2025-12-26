import os

from bson import ObjectId
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
from typing import Dict, List

from pymongo import MongoClient, UpdateOne, DESCENDING, ASCENDING
from models import Article
from services.llm import gemini_client
from utils import function_timer

class ArticleService:
    def __init__(self, db = None):
        if db is None:
            uri = os.getenv("MONGO_URI")
            db_client = MongoClient(uri)
            db = db_client["dev"]
        
        db.articles.create_index([("url", ASCENDING)], name="uniq_url", unique=True)

        db.articles.create_index(
            [
                ("title", "text"),
                ("content", "text"),
                ("summary", "text"),
            ],
            name="article_text_index",
        )
        
        self.collection = db.articles
        
    @staticmethod
    def _get_embedding(text: str) -> List[float]:
        return gemini_client.generate_embeddings(text)
    
    @staticmethod
    def build_article_text(article: Article) -> str:
        parts = [
            article.title or "",
            article.summary or "",
            (article.content or "")[:2000],
        ]
        return "\n\n".join(p for p in parts if p)

    def get_articles_batch(
            self,
            limit: int = 1000,
            last_id: ObjectId | None = None,
            filter: Dict | None = None,
            projection: Dict | None = None,
            sort_dir: int = ASCENDING,
    ) -> List[Article]:
        """
        Cursor-based pagination for large collections (faster than skip).
        Fetches a batch ordered by _id.

        Args:
            limit: max docs to return
            last_id: if provided, returns docs with _id > last_id (ASC) or _id < last_id (DESC)
            filter: additional Mongo filters to apply
            projection: Mongo projection; embedding is excluded by default
            sort_dir: ASCENDING (default) for forward scan, DESCENDING for reverse scan

        Returns:
            List[Article]
        """
        filter = filter or {}

        # Cursor condition
        if last_id is not None:
            if sort_dir == ASCENDING:
                filter["_id"] = {"$gt": last_id}
            else:
                filter["_id"] = {"$lt": last_id}

        projection = projection or {}
        projection["embedding"] = False  # keep consistent with get_articles

        cursor = (
            self.collection.find(filter=filter, projection=projection)
            .sort([("_id", sort_dir)])
            .limit(limit)
        )

        docs = list(cursor)
        return [Article(**doc) for doc in docs]
    def get_articles(
            self,
            filter: Dict | None = None,
            sorting: Dict | None = None,
            limit: int | None = None,
            projection: Dict | None = None
    ) -> List[Article]:
        filter = filter or {}
        sorting = sorting or {"created_at": DESCENDING, "publish_date": DESCENDING}
        projection = projection or {}
        projection["embedding"] = False
        
        sort_list = [(k, v) for k, v in sorting.items()]

        cursor = self.collection.find(
            filter=filter,
            projection=projection
        ).sort(sort_list)

        if limit is not None:
            cursor = cursor.limit(limit)

        docs = list(cursor)
        return [Article(**doc) for doc in docs]
    
    def url_exists_in_db(self, url: str):
        try:
            count = self.collection.count_documents({"url": url})
            return count > 0
        except Exception as e:
            raise Exception(f"Error verifying if url is already present: {e}")
    
    def insert_one_article(self, article: Article):
        doc = article.model_dump(exclude_none=True)
        now = datetime.now(timezone.utc)

        doc["created_at"] = now

        try:
            res = self.collection.update_one({"url": article.url}, {"$setOnInsert": doc}, upsert=True)
            
            if res.upserted_id:
                return {"inserted_id": str(res.upserted_id), "status": "success", "error": None}
            return {"inserted_id": None, "status": "exists", "error": "Article with this URL already exists"}
        except Exception as e:
            raise Exception(f"Error inserting article into database: {e}")
        
    def insert_many_articles(self, articles: list[Article], pipeline_run_id: ObjectId | None = None):
        articles_batch = {}
        now = datetime.now(timezone.utc)
        
        for article in articles:
            url = article.url
            # skip if url is duplicated  
            if url in articles_batch:
                continue
            doc = article.model_dump(exclude_none=True)
            doc["created_at"] = now
            if pipeline_run_id:
                doc["pipeline_run_id"] = pipeline_run_id
            articles_batch[url] = doc
        
        batch_urls = list(articles_batch.keys())
        total_unique = len(batch_urls)
        operations = []
        for url in batch_urls:
            doc = articles_batch[url]
            operations.append(UpdateOne({"url": url}, {"$setOnInsert": doc}, upsert=True))

        try:
            res = self.collection.bulk_write(operations)
        except Exception as e:
            raise Exception(f"Error inserting articles into database: {e}")
        
        inserted_ids = {}
        for op_idx, oid in (res.upserted_ids or {}).items():
            inserted_ids[batch_urls[op_idx]] = str(oid)

        inserted_total = res.upserted_count or 0
        existing_total = total_unique - inserted_total

        return {
            "inserted_count": inserted_total,
            "existing_count": existing_total,
            "total_unique_urls": total_unique,
            "inserted_ids": inserted_ids
        }

    def generate_missing_embeddings(self):
        # articles = list(self.collection.find({"embedding": None}))
        articles = self.get_articles(filter={"embedding": None})
        if articles is None:
            return
        print(f"Generating embeddings for {len(articles)} articles")
        texts = [self.build_article_text(art) for art in articles]
        print(f"Generated {len(texts)} article texts")
        
        embeddings = []
        count = 0
        while count <= len(texts) -1:
            texts_batch = texts[count:count+100]
            batch = gemini_client.generate_embeddings(texts_batch)
            embeddings += [embedding.values for embedding in batch]
            count += 100
            print(f"Batch {count} done")
        print(f"Generated {len(embeddings)} embeddings")

        for i, art in enumerate(articles):
            self.collection.update_one(
                {"url": art.url},
                {"$set": {"embedding": embeddings[i]}},
            )
            
    def remove_embedding_field(self):
        result = self.collection.update_many(
            {},  # Empty filter to affect all documents
            {'$unset': {'embedding': 1}}
        )

        print(f"Matched {result.matched_count} document(s) and modified {result.modified_count} document(s).")

        # for art in articles:
        #     text = build_article_text(art)
        #     if not text.strip():
        #         continue
        # 
        #     # Example with OpenAI style API, adapt to your client
        #     emb = openai_client.get_embedding(text)  # returns list[float]
        # 
        #     article_service.collection.update_one(
        #         {"url": art.url},
        #         {"$set": {"embedding": emb}},
        #     )

        

    
def main():
    def fetch_yahoo_articles():
        from services.scrapers import YahooScraper
        from services.llm import gemini_client

        llm_summary = True
        llm = gemini_client if llm_summary else None
        ARTICLE_LIMIT = 2
        ASYNCHRONOUS = True
        yahoo_scraper = YahooScraper(limit=ARTICLE_LIMIT, async_scrape=ASYNCHRONOUS, gemini_client=llm)
        # articles = 
        articles_list = yahoo_scraper.scrape()
        print(articles_list)
    
    fetched_articles = [Article(url='https://finance.yahoo.com/news/petrobras-q3-earnings-snapshot-005338427.html', title='Petrobras: Q3 Earnings Snapshot', content='RIO DE JANERIO RJ BR, Brazil (AP) — RIO DE JANERIO RJ BR, Brazil (AP) — Petroleo Brasileiro SA - Petrobras (PBR) on Thursday reported profit of $6.03 billion in its third quarter. On a per-share basis, the Rio De Janerio Rj Br, Brazil-based company said it had profit of 94 cents. Earnings, adjusted for non-recurring gains, came to 82 cents per share. The oil and gas company posted revenue of $23.48 billion in the period. _____ This story was generated byAutomated Insights(http://automatedinsights.com/ap) using data from Zacks Investment Research. Access aZacks stock report on PBRat https://www.zacks.com/ap/PBR', publish_date='2025-11-07 00:53:38+00:00', authors=['Associated Press Finance'], summary='Petrobras reported a profit of $6.03 billion in its third quarter.', keyword=None), Article(url='https://finance.yahoo.com/news/brasilagro-fiscal-q1-earnings-snapshot-004815833.html', title='BrasilAgro: Fiscal Q1 Earnings Snapshot', content='SAO PAULO (AP) — SAO PAULO (AP) — BrasilAgro Cia Brasileira De Propriedades Agricolas (LND) on Thursday reported a loss of $11.8 million in its fiscal first quarter. On a per-share basis, the Sao Paulo-based company said it had a loss of 11 cents. The agricultural company posted revenue of $55.6 million in the period. _____ This story was generated byAutomated Insights(http://automatedinsights.com/ap) using data from Zacks Investment Research. Access aZacks stock report on LNDat https://www.zacks.com/ap/LND', publish_date='2025-11-07 00:48:15+00:00', authors=['Associated Press Finance'], summary='BrasilAgro reported a loss of $11.8 million, or 11 cents per share, in its fiscal first quarter, with revenue of $55.6 million.', keyword=None)]
    # fetched_articles = fetch_yahoo_articles()

    article_service = ArticleService()

    print(article_service.insert_many_articles(fetched_articles))
    # print(article_service.insert_one_article(fetched_articles[0]))
    # print(article_service.get_articles())
        
if __name__ == "__main__":
    @function_timer
    def main():
        def fetch_yahoo_articles():
            from services.scrapers import YahooScraper
            from services.llm import gemini_client

            llm_summary = True
            llm = gemini_client if llm_summary else None
            ARTICLE_LIMIT = 2
            ASYNCHRONOUS = True
            yahoo_scraper = YahooScraper(limit=ARTICLE_LIMIT, async_scrape=ASYNCHRONOUS, gemini_client=llm)
            # articles = 
            articles_list = yahoo_scraper.scrape()
            print(articles_list)

        
        fetched_articles = [Article(url='https://finance.yahoo.com/news/petrobras-q3-earnings-snapshot-005338427.html', title='Petrobras: Q3 Earnings Snapshot', content='RIO DE JANERIO RJ BR, Brazil (AP) — RIO DE JANERIO RJ BR, Brazil (AP) — Petroleo Brasileiro SA - Petrobras (PBR) on Thursday reported profit of $6.03 billion in its third quarter. On a per-share basis, the Rio De Janerio Rj Br, Brazil-based company said it had profit of 94 cents. Earnings, adjusted for non-recurring gains, came to 82 cents per share. The oil and gas company posted revenue of $23.48 billion in the period. _____ This story was generated byAutomated Insights(http://automatedinsights.com/ap) using data from Zacks Investment Research. Access aZacks stock report on PBRat https://www.zacks.com/ap/PBR', publish_date='2025-11-07 00:53:38+00:00', authors=['Associated Press Finance'], summary='Petrobras reported a profit of $6.03 billion in its third quarter.', keyword=None), Article(url='https://finance.yahoo.com/news/brasilagro-fiscal-q1-earnings-snapshot-004815833.html', title='BrasilAgro: Fiscal Q1 Earnings Snapshot', content='SAO PAULO (AP) — SAO PAULO (AP) — BrasilAgro Cia Brasileira De Propriedades Agricolas (LND) on Thursday reported a loss of $11.8 million in its fiscal first quarter. On a per-share basis, the Sao Paulo-based company said it had a loss of 11 cents. The agricultural company posted revenue of $55.6 million in the period. _____ This story was generated byAutomated Insights(http://automatedinsights.com/ap) using data from Zacks Investment Research. Access aZacks stock report on LNDat https://www.zacks.com/ap/LND', publish_date='2025-11-07 00:48:15+00:00', authors=['Associated Press Finance'], summary='BrasilAgro reported a loss of $11.8 million, or 11 cents per share, in its fiscal first quarter, with revenue of $55.6 million.', keyword=None)]
        # fetched_articles = fetch_yahoo_articles()

        article_service = ArticleService()
        article_service.remove_embedding_field()
        pass
        
    main()