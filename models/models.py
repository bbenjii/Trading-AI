from typing import Optional, List, Dict

from pydantic import BaseModel, Field


class Article(BaseModel):
    url: str
    title: str
    content: str
    publish_date: Optional[str] = None
    authors: Optional[List[str]] = None

    summary: Optional[str] = None

    summary_short: Optional[str] = None
    summary_bullets: Optional[List[str]] = None
    summary_extended: Optional[str] = None

    event_type: Optional[str] = None
    event_type_reasoning: Optional[str] = None

    importance_score: Optional[float] = None
    importance_reasoning: Optional[str] = None

    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_reasoning: Optional[str] = None

    ticker_sentiments: Optional[Dict[str, float]] = None
    ticker_sentiment_reasoning: Optional[Dict[str, str]] = None

    tickers: Optional[List[str]] = None
    primary_ticker: Optional[str] = None
    primary_ticker_reasoning: Optional[str] = None

    sectors: Optional[List[str]] = None
    sector_reasoning: Optional[str] = None

    industry: Optional[List[str]] = None
    industry_reasoning: Optional[str] = None

    keywords: Optional[List[str]] = None
    keyword_map: Optional[Dict[str, List[str]]] = None
    keyword_reasoning: Optional[str] = None

    entities: Optional[List[str]] = None

    market_session: Optional[str] = None
    market_session_reasoning: Optional[str] = None

    source: Optional[str] = None



from typing import List, Optional
from pydantic import BaseModel, Field


class TickerSentiment(BaseModel):
    ticker: str = Field(description="Uppercase ticker, e.g. AAPL")
    score: float = Field(description="Sentiment score between -1 and 1")
    reasoning: Optional[str] = Field(default=None, description="One sentence justification")


class KeywordGroup(BaseModel):
    category: str = Field(description="e.g. companies, events, products, macro, financial_terms")
    items: List[str] = Field(default_factory=list)


class LlmSummary(BaseModel):
    summary_short: Optional[str] = None
    summary_bullets: Optional[List[str]] = None
    summary_extended: Optional[str] = None

    event_type: Optional[str] = None
    event_type_reasoning: Optional[str] = None

    importance_score: Optional[float] = None
    importance_reasoning: Optional[str] = None

    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_reasoning: Optional[str] = None

    tickers: Optional[List[str]] = None
    primary_ticker: Optional[str] = None
    primary_ticker_reasoning: Optional[str] = None

    sectors: Optional[List[str]] = None
    sector_reasoning: Optional[str] = None

    industry: Optional[List[str]] = None
    industry_reasoning: Optional[str] = None

    keywords: Optional[List[str]] = None
    keyword_groups: Optional[List[KeywordGroup]] = None
    keyword_reasoning: Optional[str] = None

    entities: Optional[List[str]] = None

    market_session: Optional[str] = None
    market_session_reasoning: Optional[str] = None

    source: Optional[str] = None

    # replaces ticker_sentiments + ticker_sentiment_reasoning dicts
    ticker_sentiment_items: Optional[List[TickerSentiment]] = None
