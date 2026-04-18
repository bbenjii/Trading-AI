from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import math
import pandas as pd
import numpy as np

from services.marketdata import YahooStockMarket
from services.database import article_service, untracked_symbols_service
from utils import function_timer


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Normalize datetimes to UTC-aware for consistent age calculations.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_tickers(
        hours_back: int = 4,
        now=None,
        symbol_filter=None,
        decay_coefficient: float = 0.02,
        importance_weight: float = 0.5,
        sentiment_weight: float = 0.4,
        symbol_score_weight: float = 0.6,
):
    if now is None:
        now = datetime.now(timezone.utc)
    now = _to_utc(now)

    start = now - timedelta(hours=hours_back)

    filter_stage = {
        '$match': {
            "created_at": {
                "$gte": start,
                "$lte": now
            }}
    }
    if symbol_filter:
        filter_stage["$match"]["tickers"] = {"$in": symbol_filter}
    pipeline = [
        filter_stage,
        {
            '$unwind': '$tickers'
        }, {
            '$group': {
                '_id': '$tickers',
                'count': {'$sum': 1},
                'articles': {
                    '$push': {
                        '_id': '$_id',
                        'title': '$title',
                        'url': '$url',
                        'created_at': '$created_at',
                        'publish_date': '$publish_date',
                        'sentiment': '$sentiment',
                        'sentiment_score': '$sentiment_score',
                        'importance_score': '$importance_score',
                        'ticker_sentiments': '$ticker_sentiments',
                        'authors': '$authors',
                        'summary': '$summary'
                    }
                },
                'positive_count': {
                    '$sum': {
                        '$cond': [{'$eq': ['$sentiment', 'positive']}, 1, 0]
                    }
                },
                'negative_count': {
                    '$sum': {
                        '$cond': [{'$eq': ['$sentiment', 'negative']}, 1, 0]
                    }
                },
                'neutral_count': {
                    '$sum': {
                        '$cond': [{'$eq': ['$sentiment', 'neutral']}, 1, 0]
                    }
                },
            }
        }, {
            '$project': {
                'ticker': '$_id',
                'count': 1,
                'articles': 1,
                'positive_articles': {
                    '$filter': {
                        'input': '$articles',
                        'as': 'article',
                        'cond': {'$eq': ['$$article.sentiment', 'positive']}
                    }
                },
                'negative_articles': {
                    '$filter': {
                        'input': '$articles',
                        'as': 'article',
                        'cond': {'$eq': ['$$article.sentiment', 'negative']}
                    }
                },
                'neutral_articles': {
                    '$filter': {
                        'input': '$articles',
                        'as': 'article',
                        'cond': {'$eq': ['$$article.sentiment', 'neutral']}
                    }
                },
                'positive_count': 1,
                'negative_count': 1,
                'neutral_count': 1,
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
    symbols_dfs = YahooStockMarket().get_multiple_stock_df(ticker_symbols=symbols, interval="1h", length=hours_back,
                                                           end_time=now, prepost=True)
    failed_symbols: List[str] = []
    for r in tickers:
        article_count = r.get("count")
        if not article_count:
            # Skip empty groups defensively
            continue

        neutral_count = r.get("neutral_count", len(r.get("neutral_articles", [])))
        positive_count = r.get("positive_count", len(r.get("positive_articles", [])))
        negative_count = r.get("negative_count", len(r.get("negative_articles", [])))

        symbol_score = 0
        # Use per-ticker sentiment magnitudes when available, otherwise fall back to label.
        for article in r.get("articles", []):
            created_at = _to_utc(article.get("created_at"))
            if created_at is None:
                continue
            age_minutes = (now - created_at).total_seconds() / 60
            decay_weight = math.exp(-age_minutes * decay_coefficient)

            ticker_sentiments = article.get("ticker_sentiments") or {}
            sentiment_value = ticker_sentiments.get(r.get("ticker"))
            if sentiment_value is None:
                sentiment_score = article.get("sentiment_score")
                sentiment_value = sentiment_score if sentiment_score is not None else {
                    'positive': 1,
                    'neutral': 0,
                    'negative': -1
                }.get(article.get("sentiment"), 0)

            importance_score = article.get("importance_score") or 0
            importance_multiplier = 1 + importance_score * importance_weight

            symbol_score += decay_weight * sentiment_value * importance_multiplier

        r["symbol_score"] = round(symbol_score / article_count, 3)
        r["neutral_count"] = neutral_count
        r["positive_count"] = positive_count
        r["negative_count"] = negative_count
        r["neutral_percent"] = round(neutral_count / article_count, 3)
        r["positive_percent"] = round(positive_count / article_count, 3)
        r["negative_percent"] = round(negative_count / article_count, 3)
        try:
            r["return"] = round(
                (symbols_dfs[r.get("ticker")].Close.iloc[-1] / symbols_dfs[r.get("ticker")].Close.iloc[0] - 1) * 100, 3)
        except Exception as e:
            r["return"] = np.nan
            failed_symbols.append(r.get("ticker"))

        pass
    if failed_symbols:
        untracked_symbols_service.add_untracked_symbols(failed_symbols)
    df = pd.DataFrame([
        {
            "symbol": r.get("ticker"),
            "article_count": r.get("count"),
            "positive_sentiment": r.get("positive_percent"),
            "negative_sentiment": r.get("negative_percent"),
            "neutral_sentiment": r.get("neutral_percent"),
            "weighted_sentiment_score": (r.get("positive_percent") - r.get("negative_percent")) * math.log(1 + r.get("count")),
            "symbol_score": r.get("symbol_score") * math.log(1 + r.get("count")),
            # This is momentum observed up to `now`, not a post-decision label.
            "recent_return_pct": r.get("return")
        }
        for r in tickers
    ])

    if df.empty:
        return df

    df["signal_score"] = (
        sentiment_weight * df["weighted_sentiment_score"] +
        symbol_score_weight * df["symbol_score"]
    )

    df = df.sort_values(by=["signal_score", "symbol_score", "recent_return_pct"], ascending=False)

    return df

if __name__ == "__main__":
    today = "2026-04-08"
    now = datetime.strptime(today, "%Y-%m-%d", )
    res = get_tickers(hours_back=24, now=now)
    print(res)
