from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import math
import pandas as pd
from services.marketdata import YahooStockMarket
import numpy as np

from services.llm import gemini_client
from services.marketdata import YahooStockMarket
from services.database import article_service, UntrackedSymbolsService, untracked_symbols_service
from services.marketdata.yahoo_stock_market import Ticker
from models import Article
from utils import function_timer


# def google_search()
def current_datetime():
    """
    Provides the current utc date and time as a formatted string. 

    The function retrieves the current system date and time, formats it to a 
    specific string representation in the format "YYYY-MM-DD HH:MM:SS", and 
    returns the formatted string.

    :return: 
    The current date and time as a formatted string (YYYY-MM-DD HH:MM:SS)
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_stock_info(symbol: str) -> dict:
    """
    Fetches the stock information for a given symbol.

    This function retrieves stock data for the provided stock symbol using 
    an external data source.
    You must then structure it nicely in your response if need be. 
    Args:
        symbol (str): The stock symbol to retrieve data for.
        
    Returns:
        A dictionary containing stock information.
    """
    market = YahooStockMarket()
    stock_info: Ticker = market.get_stock_info(symbol)
    return stock_info.model_dump()


def fetch_articles(
        filter: dict | None = None,
        sorting: dict | None = None,
        limit: int | None = None,
        projection: dict | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve one or many articles from the articles database.

    An Article contains the following fields:
        url: str
        title: str
        content: str
        publish_date: Optional[str]
        authors: Optional[List[str]]
        summary: Optional[str]
        keyword: Optional[str]
        sectors: Optional[List[str]]
        keywords: Optional[List[str]]
        sentiment: Optional[str]
        tickers: Optional[List[str]]

    Args:
        filter (dict | None):
            Mongo-style query document selecting articles to return.
            Defaults to {} (no filtering).
        sorting (dict | None):
            A mapping of field → direction (1 for ascending, -1 for descending).
            Example: {"publish_date": -1}.
        limit (int | None):
            Maximum number of results to return. None means no limit.
        projection (dict | None):
            A mapping defining which fields to include/exclude.
            Example: {"content": False, "_id": False}.

    Returns:
        List[dict]: A list of article dictionaries (model_dump() of Article models).
    """

    # Normalize arguments
    filter = filter or {}
    sorting = sorting or {}
    projection = projection or {}
    # Fetch full Article objects
    articles = article_service.get_articles(filter=filter, sorting=sorting, limit=limit, projection=projection)

    # Convert Pydantic Article models → plain dicts
    return [article.model_dump() for article in articles]


def get_latest_articles(limit: int = 5) -> list:
    """
    Return the most recent articles overall.

    Args:
        limit: Maximum number of articles to return.

    Returns:
        A list of article dicts with fields:
        url, title, summary, publish_date, authors, sectors, keywords, sentiment, tickers.
    """
    articles = article_service.get_articles(
        sorting={"publish_date": -1, "created_at": -1},
        limit=limit,
        projection={"content": False},  # often you do not need full content
    )
    return [a.model_dump() for a in articles]


def get_recent_articles(days_back: int = 1, limit: int = 50) -> list:
    """
    Return recent articles in the last N days.

    Args:
        days_back: Number of days back from now (UTC) to include.
                   For example, 1 means "today", 2 means "today and yesterday".
        limit: Maximum number of articles to return. 0 means no limit.

    Returns:
        A list of article dicts.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)
    # assuming publish_date is stored as ISO string or similar
    start_str = start.strftime("%Y-%m-%d 00:00:00")
    now_str = now.strftime("%Y-%m-%d 23:59:59")

    filter_doc = {
        "publish_date": {
            "$gte": start_str,
            "$lte": now_str
        }
    }

    articles = article_service.get_articles(
        filter=filter_doc,
        sorting={"publish_date": -1},
        limit=limit,
        projection={"content": False},
    )
    return [a.model_dump() for a in articles]


def get_ticker_news(symbol: str, days_back: int = 7, limit: int = 50) -> list:
    """
    Return recent articles related to a specific stock ticker.

    Args:
        symbol: Stock ticker symbol, like "AAPL" or "NVDA".
        days_back: Number of days back from now to search.
        limit: Maximum number of articles to return.

    Returns:
        A list of article dicts.
    """
    symbol = symbol.upper()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)
    start_str = start.strftime("%Y-%m-%d 00:00:00")
    now_str = now.strftime("%Y-%m-%d 23:59:59")

    filter_doc = {
        "tickers": symbol,
        "publish_date": {
            "$gte": start_str,
            "$lte": now_str
        }
    }

    articles = article_service.get_articles(
        filter=filter_doc,
        sorting={"publish_date": -1},
        limit=limit,
        projection={"content": False},
    )
    return [a.model_dump() for a in articles]


def get_sector_news(sector: str, days_back: int = 7, limit: int = 50) -> list:
    """
    Return recent articles related to a specific sector.

    Args:
        sector: Sector name such as "Technology", "Financials", "Energy".
        days_back: Number of days back from now to search.
        limit: Maximum number of articles to return.

    Returns:
        A list of article dicts.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)
    start_str = start.strftime("%Y-%m-%d 00:00:00")
    now_str = now.strftime("%Y-%m-%d 23:59:59")

    filter_doc = {
        "sectors": sector,
        "publish_date": {
            "$gte": start_str,
            "$lte": now_str
        }
    }

    articles = article_service.get_articles(
        filter=filter_doc,
        sorting={"publish_date": -1},
        limit=limit,
        projection={"content": False},
    )
    return [a.model_dump() for a in articles]


def search_articles_by_text(query: str, days_back: int = 7, limit: int = 50) -> list:
    """
    Search articles by a free text query within a recent time window.

    Args:
        query: Text to search for in title or content.
        days_back: Number of days back from now.
        limit: Maximum number of results.

    Returns:
        A list of article dicts.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)
    start_str = start.strftime("%Y-%m-%d 00:00:00")
    now_str = now.strftime("%Y-%m-%d 23:59:59")

    filter_doc = {
        "publish_date": {
            "$gte": start_str,
            "$lte": now_str
        },
        "$text": {"$search": query}
    }

    articles = article_service.get_articles(
        filter=filter_doc,
        sorting={"publish_date": -1},
        limit=limit,
        projection={"content": False},
    )
    return [a.model_dump() for a in articles]


def rag_search_articles(query: str, days_back: int = 7, limit: int = 10) -> list:
    """
    Semantic search over the articles collection using stored embeddings.

    Args:
        query: Natural language query (e.g. "AI chip makers news", "tech sector today").
        days_back: How many days back from now (UTC) to search.
        limit: Maximum number of articles to return.

    Returns:
        A list of article dicts (without the raw embedding).
    """

    # 1. Embed the query
    query_embedding = gemini_client.generate_embeddings(query)[0].values  # -> list[float]

    # 2. Build time range filter
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)

    start_str = start.strftime("%Y-%m-%d 00:00:00")
    now_str = now.strftime("%Y-%m-%d 23:59:59")

    # 3. MongoDB vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "article_embedding_index",  # name of your vector index
                "path": "embedding",  # field holding the embedding array
                "queryVector": query_embedding,
                "numCandidates": max(limit * 5, 50),  # wider candidate set, then limit
                "limit": limit,
                "filter": {
                    "publish_date": {
                        "$gte": start_str,
                        "$lte": now_str,
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "url": 1,
                "title": 1,
                "summary": 1,
                "content": 1,
                "publish_date": 1,
                "authors": 1,
                "keyword": 1,
                "sectors": 1,
                "keywords": 1,
                "sentiment": 1,
                "tickers": 1,
                # optional: expose score for debugging or re-ranking in the LLM
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = list(article_service.collection.aggregate(pipeline))
    return results

@function_timer
def get_tickers(hours_back: int = 4):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_back)
    start_str = start.strftime("%Y-%m-%d 00:00:00")
    now_str = now.strftime("%Y-%m-%d 23:59:59")

    filter_stage = {
        '$match': {
            "created_at": {
                "$gte": start,
                "$lte": now
            }}
    }
    pipeline = [
        filter_stage,
        {
            '$unwind': '$tickers'
        }, {
            '$group': {
                '_id': '$tickers',
                'count': {
                    '$sum': 1
                },
                'articles': {
                    '$push': {
                        '_id': '$_id',
                        'title': '$title',
                        'url': '$url',
                        'created_at': '$created_at',
                        'publish_date': '$publish_date',
                        'sentiment': '$sentiment',
                        'authors': '$authors',
                        'summary': '$summary'
                    }
                },
                'positive_articles': {
                    '$push': {
                        '$cond': [
                            {
                                '$eq': [
                                    '$sentiment', 'positive'
                                ]
                            }, {
                                '_id': '$_id',
                                'title': '$title',
                                'url': '$url',
                                'created_at': '$created_at',
                                'publish_date': '$publish_date',
                                'sentiment': '$sentiment'
                            }, '$Remove'
                        ]
                    }
                },
                'negative_articles': {
                    '$push': {
                        '$cond': [
                            {
                                '$eq': [
                                    '$sentiment', 'negative'
                                ]
                            }, {
                                '_id': '$_id',
                                'title': '$title',
                                'url': '$url',
                                'created_at': '$created_at',
                                'publish_date': '$publish_date',
                                'sentiment': '$sentiment'
                            }, '$Remove'
                        ]
                    }
                },
                'neutral_articles': {
                    '$push': {
                        '$cond': [
                            {
                                '$eq': [
                                    '$sentiment', 'neutral'
                                ]
                            }, {
                                '_id': '$_id',
                                'title': '$title',
                                'url': '$url',
                                'created_at': '$created_at',
                                'publish_date': '$publish_date',
                                'sentiment': '$sentiment'
                            }, '$Remove'
                        ]
                    }
                }
            }
        }, {
            '$project': {
                'ticker': '$_id',
                'count': 1,
                'articles': 1,
                'positive_articles': 1,
                'negative_articles': 1,
                'neutral_articles': 1,
                '_id': 0,

            }
        }, {
            '$sort': {
                'count': -1
            }
        }

    ]
    # articles = article_service.collection.find()
    res = list(article_service.collection.aggregate(pipeline))
    untracked_symbols = untracked_symbols_service.get_untracked_symbols()
    res = res[:]
    tickers = [r for r in res if r.get("ticker") not in untracked_symbols]
    
    symbols = [r.get("ticker") for r in tickers]
    symbols_dfs = YahooStockMarket().get_multiple_stock_df(ticker_symbols=symbols, interval="1h", length=24)
    failed_symbols = []
    for r in tickers:
        article_count = r.get("count")
        neutral_count = len(r.get("neutral_articles"))
        positive_count = len(r.get("positive_articles"))
        negative_count = len(r.get("negative_articles"))
        
        symbol_df = symbols_dfs[r.get("ticker")]
        created_at = r.get("")
        symbol_score = 0
        coefficient = 0.02 
        for article in r.get("positive_articles"):
            created_at = article.get("created_at").replace(tzinfo=timezone.utc)
            print(created_at)
            age = (now - created_at).total_seconds() / 60
            weight = math.exp(-age * coefficient)
            
            symbol_score += weight

        # for article in r.get("neutral_articles"):
        #     created_at = article.get("created_at").replace(tzinfo=timezone.utc)
        #     print(created_at)
        #     age = (now - created_at).total_seconds() / 60
        #     weight = math.exp(-age * coefficient)*0.5
        # 
        #     symbol_score += weight
            
        for article in r.get("negative_articles"):
            created_at = article.get("created_at").replace(tzinfo=timezone.utc)
            print(created_at)
            age = (now - created_at).total_seconds() / 60
            weight = math.exp(-age * coefficient)

            symbol_score -= weight
        
        r["symbol_score"] = round(symbol_score/ article_count, 3)
        r["neutral_count"] = neutral_count
        r["positive_count"] = positive_count
        r["negative_count"] = negative_count
        r["neutral_percent"] = round(neutral_count / article_count, 3)
        r["positive_percent"] = round(positive_count / article_count, 3)
        r["negative_percent"] = round(negative_count / article_count, 3)
        try:
            r["return"] = round( (symbols_dfs[r.get("ticker")].Close.iloc[-1] / symbols_dfs[r.get("ticker")].Close.iloc[0] - 1)*100, 3)
        except Exception as e:
            r["return"] = None
            failed_symbols.append(r.get("ticker"))
            
        pass
    print(len(tickers))
    if failed_symbols:
        untracked_symbols_service.add_untracked_symbols(failed_symbols)
    df = pd.DataFrame([
        {
            "symbol": r.get("ticker"),
            "article_count": r.get("count"),
            "positive_sentiment": r.get("positive_percent"),
            "negative_sentiment": r.get("negative_percent"),
            "neutral_sentiment": r.get("neutral_percent"),
            "weighted_sentiment_score": (r.get("positive_percent") - r.get("negative_percent"))*math.log(r.get("count")),
            "symbol_score": r.get("symbol_score")*math.log(1+ r.get("count")),
            "return": r.get("return")
        }
        for r in tickers
    ])
    
    # df["score"] = df["weighted_sentiment_score"] * math.log(1+ df["return"])
    df["score"] = df["weighted_sentiment_score"] * df["return"]
    df["score2"] = df["symbol_score"] * df["return"]

    df = df.sort_values(by="score", ascending=False)

    print(df)
    return tickers



# df= YahooStockMarket().get_stock_period_return(ticker_symbol="AAPL", interval="1d", length=10)
# df_list= YahooStockMarket().get_multiple_stock_df(ticker_symbols=["AAPL", "META"], interval="1d", length=10)
tickers = get_tickers(24)
# print("more than 1", len([t for t in tickers if t.get("count")>1]))
# pass

tools = [get_stock_info, current_datetime,
         fetch_articles, 
         get_ticker_news, get_sector_news, search_articles_by_text, get_latest_articles, get_recent_articles]

gemini_client.chat_session( google_search=False)
