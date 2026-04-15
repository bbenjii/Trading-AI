from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse
import re

import certifi
from bson import ObjectId
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from services.llm import gemini_client as default_gemini_client

load_dotenv()


class CatalystClusterService:
    def __init__(self, db=None, gemini_client=None):
        if db is None:
            uri = os.getenv("MONGO_URI")
            db_client = MongoClient(uri, tlsCAFile=certifi.where())
            db = db_client["dev"]
            
        self.gemini_client = gemini_client if gemini_client is not None else default_gemini_client
        self.collection = db.catalyst_clusters
        self.article_collection = db.articles

        self.collection.create_index([("cluster_id", ASCENDING)], name="cluster_id_uniq", unique=True)
        self.collection.create_index([("primary_ticker", ASCENDING), ("last_updated_at", DESCENDING)], name="ticker_last_updated")
        self.collection.create_index([("tickers", ASCENDING), ("last_updated_at", DESCENDING)], name="tickers_last_updated")
        self.collection.create_index([("event_type", ASCENDING), ("last_updated_at", DESCENDING)], name="event_type_last_updated")
        self.collection.create_index([("article_ids", ASCENDING)], name="article_ids_idx")
        self.collection.create_index([("last_updated_at", DESCENDING)], name="last_updated_desc")

    def cluster_articles_by_urls(self, urls: List[str]) -> Dict[str, int]:
        if not urls:
            return {
                "processed": 0,
                "clustered": 0,
                "created": 0,
                "attached": 0,
                "skipped": 0,
                "skip_reasons": {},
            }

        articles = list(
            self.article_collection.find(
                {"url": {"$in": urls}},
                sort=[("created_at", ASCENDING), ("publish_date", ASCENDING), ("_id", ASCENDING)],
            )
        )

        processed = 0
        clustered = 0
        created = 0
        attached = 0
        skipped = 0
        skip_reasons: Counter[str] = Counter()

        for article in articles:
            processed += 1
            result = self._cluster_single_article(article)
            action = result.get("action")
            reason = result.get("reason")
            if action == "created":
                clustered += 1
                created += 1
            elif action == "attached":
                clustered += 1
                attached += 1
            else:
                skipped += 1
                if reason:
                    skip_reasons[reason] += 1

        return {
            "processed": processed,
            "clustered": clustered,
            "created": created,
            "attached": attached,
            "skipped": skipped,
            "skip_reasons": dict(skip_reasons),
        }

    def cluster_recent_articles(self, days: int, unclustered_only: bool = True) -> Dict[str, int]:
        if days <= 0:
            return {
                "processed": 0,
                "clustered": 0,
                "created": 0,
                "attached": 0,
                "skipped": 0,
                "skip_reasons": {},
            }

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        query: Dict[str, Any] = {"created_at": {"$gte": cutoff}}
        if unclustered_only:
            query["cluster_id"] = {"$exists": False}

        urls = [
            doc["url"]
            for doc in self.article_collection.find(
                query,
                projection={"url": True},
                sort=[("created_at", ASCENDING), ("_id", ASCENDING)],
            )
            if doc.get("url")
        ]

        return self.cluster_articles_by_urls(urls)

    def _cluster_single_article(self, article: Dict[str, Any]) -> Dict[str, Optional[str]]:
        clusterability = self._clusterability_reason(article)
        if clusterability is not None:
            self._mark_article_unclustered(article["_id"])
            return {"action": "skipped", "reason": clusterability}

        candidates = self._find_candidate_clusters(article)
        scored_candidates = []
        best_cluster = None
        best_score = 0.0

        for cluster in candidates:
            score = self._score_article_cluster_match(article, cluster)
            scored_candidates.append((cluster, score))
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster and best_score >= 0.55:
            self._attach_article_to_cluster(article, best_cluster, match_score=best_score)
            return {"action": "attached", "reason": None}

        llm_cluster = self._resolve_ambiguous_match_with_gemini(article, scored_candidates)
        if llm_cluster is not None:
            self._attach_article_to_cluster(article, llm_cluster, match_score=best_score)
            return {"action": "attached", "reason": None}

        if self._should_create_cluster(article):
            self._create_cluster(article)
            return {"action": "created", "reason": None}

        self._mark_article_unclustered(article["_id"])
        if candidates:
            return {"action": "skipped", "reason": "match_below_threshold"}
        return {"action": "skipped", "reason": "no_candidate_clusters"}

    def _find_candidate_clusters(self, article: Dict[str, Any], lookback_hours: int = 168) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        article_tickers = self._clean_upper_list(article.get("tickers"))
        primary_ticker = self._clean_upper(article.get("primary_ticker"))
        event_type = self._normalize_text(article.get("event_type"))

        conditions: List[Dict[str, Any]] = []
        if primary_ticker:
            conditions.append({"primary_ticker": primary_ticker})
        if article_tickers:
            conditions.append({"tickers": {"$in": article_tickers}})
        if event_type and event_type != "other":
            conditions.append({"event_type": event_type})

        query: Dict[str, Any] = {"last_updated_at": {"$gte": cutoff}}
        if conditions:
            query["$or"] = conditions
        candidates = list(self.collection.find(query).sort([("last_updated_at", DESCENDING)]).limit(40))
        if candidates:
            return candidates

        fallback_candidates = list(
            self.collection.find({"last_updated_at": {"$gte": cutoff}}).sort([("last_updated_at", DESCENDING)]).limit(60)
        )
        article_terms = self._article_theme_terms(article)
        if not article_terms:
            return fallback_candidates[:15]

        ranked = []
        for cluster in fallback_candidates:
            cluster_terms = self._cluster_theme_terms(cluster)
            if not cluster_terms:
                continue
            overlap = len(article_terms & cluster_terms)
            if overlap > 0:
                ranked.append((overlap, cluster))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [cluster for _, cluster in ranked[:15]]

    def _score_article_cluster_match(self, article: Dict[str, Any], cluster: Dict[str, Any]) -> float:
        score = 0.0

        primary_ticker = self._clean_upper(article.get("primary_ticker"))
        cluster_primary = self._clean_upper(cluster.get("primary_ticker"))
        article_tickers = set(self._clean_upper_list(article.get("tickers")))
        cluster_tickers = set(self._clean_upper_list(cluster.get("tickers")))
        if primary_ticker and cluster_primary and primary_ticker == cluster_primary:
            score += 0.35
        elif primary_ticker and primary_ticker in cluster_tickers:
            score += 0.20

        if article_tickers and cluster_tickers:
            overlap = len(article_tickers & cluster_tickers) / max(len(article_tickers | cluster_tickers), 1)
            score += 0.25 * overlap

        event_type = self._normalize_text(article.get("event_type"))
        cluster_event_type = self._normalize_text(cluster.get("event_type"))
        if event_type and cluster_event_type and event_type == cluster_event_type:
            score += 0.20
        elif (not event_type or event_type == "other") and cluster_event_type:
            score += 0.05

        article_terms = self._article_theme_terms(article)
        cluster_terms = self._cluster_theme_terms(cluster)
        if article_terms and cluster_terms:
            overlap = len(article_terms & cluster_terms) / max(len(article_terms | cluster_terms), 1)
            score += 0.20 * overlap

        article_title_terms = self._headline_terms(article.get("title"))
        cluster_title_terms = self._headline_terms(cluster.get("representative_title")) | self._headline_terms(cluster.get("canonical_title"))
        if article_title_terms and cluster_title_terms:
            overlap = len(article_title_terms & cluster_title_terms) / max(len(article_title_terms | cluster_title_terms), 1)
            score += 0.25 * overlap

        article_publish = self._parse_datetime(article.get("publish_date")) or self._coerce_datetime(article.get("created_at"))
        cluster_last = self._coerce_datetime(cluster.get("last_updated_at")) or self._coerce_datetime(cluster.get("last_seen_at"))
        if isinstance(article_publish, datetime) and isinstance(cluster_last, datetime):
            hours = abs((cluster_last - article_publish).total_seconds()) / 3600
            if hours <= 12:
                score += 0.10
            elif hours <= 48:
                score += 0.05

        return min(score, 1.0)

    def _create_cluster(self, article: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        article_id = str(article["_id"])
        article_url = article["url"]
        first_seen = self._parse_datetime(article.get("publish_date")) or self._coerce_datetime(article.get("created_at")) or now

        cluster_id = f"cat_{ObjectId()}"
        source = self._article_source(article)
        tickers = self._clean_upper_list(article.get("tickers"))
        importance_score = self._safe_float(article.get("importance_score"))
        sentiment_score = self._safe_float(article.get("sentiment_score"))
        representative_title = (article.get("summary_short") or article.get("title") or "").strip()
        theme_label = self._derive_theme_label(article)

        cluster_doc = {
            "cluster_id": cluster_id,
            "cluster_version": 1,
            "primary_ticker": self._clean_upper(article.get("primary_ticker")),
            "tickers": tickers,
            "event_type": self._normalize_text(article.get("event_type")),
            "canonical_title": representative_title,
            "theme_label": theme_label,
            "theme_reasoning": self._derive_theme_reasoning(article),
            "catalyst_strength_label": self._strength_label(importance_score),
            "impact_window": self._derive_impact_window(article),
            "projected_direction": self._projected_direction(sentiment_score),
            "article_ids": [article_id],
            "article_urls": [article_url],
            "article_count": 1,
            "representative_article_id": article_id,
            "representative_article_url": article_url,
            "representative_title": representative_title,
            "cluster_summary_short": self._derive_cluster_summary(article),
            "cluster_summary_bullets": self._derive_cluster_bullets([article]),
            "titles": [article.get("title") or ""],
            "sources": [source] if source else [],
            "source_count": 1 if source else 0,
            "importance_score_agg": importance_score,
            "sentiment_agg": sentiment_score,
            "first_seen_at": first_seen,
            "last_seen_at": first_seen,
            "last_updated_at": now,
            "updated_at": now,
        }
        cluster_doc.update(self._label_cluster_with_gemini(cluster_doc, [article]))

        self.collection.insert_one(cluster_doc)
        self._update_article_cluster_fields(article["_id"], cluster_id, 1)
        return cluster_doc

    def _attach_article_to_cluster(self, article: Dict[str, Any], cluster: Dict[str, Any], match_score: float) -> None:
        cluster_articles = list(
            self.article_collection.find({"_id": {"$in": [ObjectId(x) for x in cluster.get("article_ids", []) if ObjectId.is_valid(x)]}})
        )
        cluster_articles.append(article)

        rebuilt = self._rebuild_cluster_document(cluster, cluster_articles, match_score=match_score)

        self.collection.update_one(
            {"_id": cluster["_id"]},
            {"$set": rebuilt},
        )
        self.article_collection.update_many(
            {"_id": {"$in": [a["_id"] for a in cluster_articles]}},
            {
                "$set": {
                    "cluster_id": cluster["cluster_id"],
                    "cluster_version": rebuilt["cluster_version"],
                    "clustered_at": datetime.now(timezone.utc),
                }
            },
        )

    def _rebuild_cluster_document(self, cluster: Dict[str, Any], articles: List[Dict[str, Any]], match_score: float) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        unique_articles: Dict[str, Dict[str, Any]] = {}
        for article in articles:
            unique_articles[str(article["_id"])] = article
        articles = list(unique_articles.values())

        sources = sorted({src for src in (self._article_source(a) for a in articles) if src})
        titles = [a.get("title") or "" for a in articles if a.get("title")]
        tickers = sorted({ticker for a in articles for ticker in self._clean_upper_list(a.get("tickers"))})

        importance_values = [self._safe_float(a.get("importance_score")) for a in articles if self._safe_float(a.get("importance_score")) is not None]
        sentiment_values = [self._safe_float(a.get("sentiment_score")) for a in articles if self._safe_float(a.get("sentiment_score")) is not None]

        representative = self._select_representative_article(articles)
        first_seen = min(
            (self._parse_datetime(a.get("publish_date")) or self._coerce_datetime(a.get("created_at")) or now for a in articles),
            default=self._coerce_datetime(cluster.get("first_seen_at")) or now,
        )
        last_seen = max(
            (self._parse_datetime(a.get("publish_date")) or self._coerce_datetime(a.get("created_at")) or now for a in articles),
            default=self._coerce_datetime(cluster.get("last_seen_at")) or now,
        )

        importance_avg = round(sum(importance_values) / len(importance_values), 4) if importance_values else None
        sentiment_avg = round(sum(sentiment_values) / len(sentiment_values), 4) if sentiment_values else None
        cluster_version = int(cluster.get("cluster_version") or 1) + 1

        updated = {
            "cluster_version": cluster_version,
            "primary_ticker": self._choose_primary_ticker(articles, cluster),
            "tickers": tickers,
            "event_type": self._choose_event_type(articles, cluster),
            "canonical_title": (representative.get("summary_short") or representative.get("title") or "").strip(),
            "theme_label": self._derive_theme_label(representative, fallback=cluster.get("theme_label")),
            "theme_reasoning": self._derive_theme_reasoning(representative, match_score=match_score),
            "catalyst_strength_label": self._strength_label(importance_avg),
            "impact_window": self._derive_impact_window(representative),
            "projected_direction": self._projected_direction(sentiment_avg),
            "article_ids": [str(a["_id"]) for a in articles],
            "article_urls": [a["url"] for a in articles],
            "article_count": len(articles),
            "representative_article_id": str(representative["_id"]),
            "representative_article_url": representative["url"],
            "representative_title": (representative.get("summary_short") or representative.get("title") or "").strip(),
            "cluster_summary_short": self._derive_cluster_summary(representative),
            "cluster_summary_bullets": self._derive_cluster_bullets(articles),
            "titles": titles,
            "sources": sources,
            "source_count": len(sources),
            "importance_score_agg": importance_avg,
            "sentiment_agg": sentiment_avg,
            "first_seen_at": first_seen,
            "last_seen_at": last_seen,
            "last_updated_at": now,
            "updated_at": now,
        }
        updated.update(self._label_cluster_with_gemini({**cluster, **updated}, articles))
        return updated

    def _update_article_cluster_fields(self, article_id: ObjectId, cluster_id: str, cluster_version: int) -> None:
        self.article_collection.update_one(
            {"_id": article_id},
            {
                "$set": {
                    "cluster_id": cluster_id,
                    "cluster_version": cluster_version,
                    "clustered_at": datetime.now(timezone.utc),
                }
            },
        )

    def _mark_article_unclustered(self, article_id: ObjectId) -> None:
        self.article_collection.update_one(
            {"_id": article_id},
            {
                "$set": {
                    "clustered_at": datetime.now(timezone.utc),
                },
                "$unset": {
                    "cluster_id": "",
                    "cluster_version": "",
                },
            },
        )

    def _clusterability_reason(self, article: Dict[str, Any]) -> Optional[str]:
        tickers = self._clean_upper_list(article.get("tickers"))
        primary_ticker = self._clean_upper(article.get("primary_ticker"))
        event_type = self._normalize_text(article.get("event_type"))
        importance = self._safe_float(article.get("importance_score")) or 0.0
        title = (article.get("title") or "").strip()
        terms = self._article_theme_terms(article)

        if not tickers and not primary_ticker:
            return "missing_tickers"
        if not title:
            return "missing_title"
        if importance < 0.15 and not terms:
            return "weak_signal"
        if (not event_type or event_type == "other") and not terms and importance < 0.4:
            return "missing_theme_signal"
        return None

    def _should_create_cluster(self, article: Dict[str, Any]) -> bool:
        importance = self._safe_float(article.get("importance_score")) or 0.0
        event_type = self._normalize_text(article.get("event_type"))
        has_primary = bool(self._clean_upper(article.get("primary_ticker")))
        ticker_count = len(self._clean_upper_list(article.get("tickers")))
        theme_terms = self._article_theme_terms(article)

        if importance >= 0.40:
            return True
        if event_type in {"earnings", "guidance", "merger_acquisition", "regulation", "lawsuit", "bankruptcy", "product_launch", "geopolitical", "macro_data"}:
            return True
        if has_primary and theme_terms and importance >= 0.25:
            return True
        if ticker_count >= 2 and theme_terms:
            return True
        return False

    def _select_representative_article(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        def sort_key(article: Dict[str, Any]):
            importance = self._safe_float(article.get("importance_score")) or 0.0
            created = self._parse_datetime(article.get("publish_date")) or article.get("created_at") or datetime.min.replace(tzinfo=timezone.utc)
            title_len = len(article.get("title") or "")
            return (importance, created, title_len)

        return max(articles, key=sort_key)

    def _choose_primary_ticker(self, articles: List[Dict[str, Any]], cluster: Dict[str, Any]) -> Optional[str]:
        explicit = [self._clean_upper(a.get("primary_ticker")) for a in articles if self._clean_upper(a.get("primary_ticker"))]
        if explicit:
            return Counter(explicit).most_common(1)[0][0]

        tickers = [ticker for a in articles for ticker in self._clean_upper_list(a.get("tickers"))]
        if tickers:
            return Counter(tickers).most_common(1)[0][0]

        return self._clean_upper(cluster.get("primary_ticker"))

    def _choose_event_type(self, articles: List[Dict[str, Any]], cluster: Dict[str, Any]) -> Optional[str]:
        events = [self._normalize_text(a.get("event_type")) for a in articles if self._normalize_text(a.get("event_type")) and self._normalize_text(a.get("event_type")) != "other"]
        if events:
            return Counter(events).most_common(1)[0][0]
        return self._normalize_text(cluster.get("event_type"))

    def _derive_cluster_summary(self, article: Dict[str, Any]) -> Optional[str]:
        return (article.get("summary_short") or article.get("title") or None)

    def _derive_cluster_bullets(self, articles: List[Dict[str, Any]]) -> List[str]:
        bullets: List[str] = []
        seen = set()
        for article in articles:
            for bullet in article.get("summary_bullets") or []:
                text = (bullet or "").strip()
                key = text.lower()
                if text and key not in seen:
                    seen.add(key)
                    bullets.append(text)
                if len(bullets) >= 4:
                    return bullets
        return bullets

    def _derive_theme_label(self, article: Dict[str, Any], fallback: Optional[str] = None) -> Optional[str]:
        keywords = [k for k in (article.get("keywords") or []) if isinstance(k, str)]
        event_type = self._normalize_text(article.get("event_type"))
        primary_ticker = self._clean_upper(article.get("primary_ticker"))

        if primary_ticker and event_type and keywords:
            return f"{primary_ticker} {event_type.replace('_', ' ')}: {keywords[0]}"
        if primary_ticker and event_type:
            return f"{primary_ticker} {event_type.replace('_', ' ')}"
        if article.get("summary_short"):
            return article["summary_short"]
        return fallback or article.get("title")

    def _derive_theme_reasoning(self, article: Dict[str, Any], match_score: Optional[float] = None) -> Optional[str]:
        primary_ticker = self._clean_upper(article.get("primary_ticker"))
        event_type = self._normalize_text(article.get("event_type"))
        if primary_ticker and event_type:
            reason = f"Clustered around {primary_ticker} and the {event_type.replace('_', ' ')} event signature."
        elif primary_ticker:
            reason = f"Clustered around repeated coverage tied to {primary_ticker}."
        else:
            reason = "Clustered around repeated coverage of the same market-moving theme."
        if match_score is not None:
            reason = f"{reason} Match score {match_score:.2f}."
        return reason

    def _resolve_ambiguous_match_with_gemini(self, article: Dict[str, Any], scored_candidates: List[tuple[Dict[str, Any], float]]) -> Optional[Dict[str, Any]]:
        if not self.gemini_client or not scored_candidates:
            return None

        scored_candidates = sorted(scored_candidates, key=lambda item: item[1], reverse=True)
        best_score = scored_candidates[0][1]
        if best_score < 0.35 or best_score >= 0.55:
            return None

        article_payload = self._serialize_article_for_llm(article)
        candidate_payload = [
            self._serialize_cluster_for_llm(cluster, score)
            for cluster, score in scored_candidates[:3]
        ]
        decision = self.gemini_client.decide_article_cluster(article_payload, candidate_payload)
        if not decision:
            return None
        if decision.get("decision") != "attach":
            return None
        if float(decision.get("confidence", 0.0)) < 0.6:
            return None

        target_id = decision.get("cluster_id")
        for cluster, _ in scored_candidates[:3]:
            if cluster.get("cluster_id") == target_id:
                return cluster
        return None

    def _label_cluster_with_gemini(self, cluster_doc: Dict[str, Any], articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.gemini_client:
            return {}

        representative = self._select_representative_article(articles)
        article_samples = [
            {
                "title": article.get("title"),
                "summary_short": article.get("summary_short"),
                "summary_bullets": article.get("summary_bullets") or [],
                "tickers": self._clean_upper_list(article.get("tickers")),
                "primary_ticker": self._clean_upper(article.get("primary_ticker")),
                "event_type": self._normalize_text(article.get("event_type")),
                "source": self._article_source(article),
                "publish_date": str(article.get("publish_date") or ""),
            }
            for article in articles[:5]
        ]

        payload = {
            "cluster_id": cluster_doc.get("cluster_id"),
            "primary_ticker": cluster_doc.get("primary_ticker"),
            "tickers": cluster_doc.get("tickers") or [],
            "event_type": cluster_doc.get("event_type"),
            "article_count": len(articles),
            "representative_title": representative.get("title"),
            "existing_theme_label": cluster_doc.get("theme_label"),
            "existing_summary_short": cluster_doc.get("cluster_summary_short"),
            "articles": article_samples,
        }

        llm_update = self.gemini_client.label_catalyst_cluster(payload)
        if not llm_update:
            return {}
        return llm_update

    def _serialize_article_for_llm(self, article: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": article.get("title"),
            "summary_short": article.get("summary_short"),
            "summary_bullets": article.get("summary_bullets") or [],
            "tickers": self._clean_upper_list(article.get("tickers")),
            "primary_ticker": self._clean_upper(article.get("primary_ticker")),
            "event_type": self._normalize_text(article.get("event_type")),
            "keywords": article.get("keywords") or [],
            "entities": article.get("entities") or [],
            "publish_date": str(article.get("publish_date") or ""),
            "source": self._article_source(article),
        }

    def _serialize_cluster_for_llm(self, cluster: Dict[str, Any], score: float) -> Dict[str, Any]:
        return {
            "cluster_id": cluster.get("cluster_id"),
            "heuristic_score": round(score, 4),
            "canonical_title": cluster.get("canonical_title"),
            "theme_label": cluster.get("theme_label"),
            "cluster_summary_short": cluster.get("cluster_summary_short"),
            "cluster_summary_bullets": cluster.get("cluster_summary_bullets") or [],
            "primary_ticker": cluster.get("primary_ticker"),
            "tickers": cluster.get("tickers") or [],
            "event_type": cluster.get("event_type"),
            "article_count": cluster.get("article_count"),
            "representative_title": cluster.get("representative_title"),
            "sources": cluster.get("sources") or [],
            "last_seen_at": str(cluster.get("last_seen_at") or ""),
        }

    def _derive_impact_window(self, article: Dict[str, Any]) -> str:
        event_type = self._normalize_text(article.get("event_type"))
        market_session = self._normalize_text(article.get("market_session"))

        if event_type in {"earnings", "guidance", "analyst_commentary", "macro_data"}:
            return "intraday"
        if event_type in {"product_launch", "partnership", "downgrade_upgrade", "investigation"}:
            return "1-3_days"
        if event_type in {"merger_acquisition", "regulation", "lawsuit", "bankruptcy", "geopolitical"}:
            return "1-4_weeks"
        if market_session in {"premarket", "after_hours", "market_hours"}:
            return "intraday"
        return "unclear"

    def _projected_direction(self, sentiment_score: Optional[float]) -> str:
        if sentiment_score is None:
            return "unclear"
        if sentiment_score >= 0.2:
            return "bullish"
        if sentiment_score <= -0.2:
            return "bearish"
        return "mixed"

    def _strength_label(self, importance_score: Optional[float]) -> str:
        score = importance_score if importance_score is not None else 0.0
        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    def _article_source(self, article: Dict[str, Any]) -> Optional[str]:
        source = article.get("source")
        if source:
            return source
        url = article.get("url")
        if not url:
            return None
        hostname = urlparse(url).hostname or ""
        return hostname.replace("www.", "") or None

    def _article_theme_terms(self, article: Dict[str, Any]) -> set[str]:
        terms = set()
        for value in article.get("keywords") or []:
            terms.update(self._headline_terms(value))
        for value in article.get("entities") or []:
            terms.update(self._headline_terms(value))
        summary = article.get("summary_short") or article.get("title") or ""
        terms.update(self._headline_terms(summary))
        return terms

    def _cluster_theme_terms(self, cluster: Dict[str, Any]) -> set[str]:
        terms = set()
        for value in cluster.get("titles") or []:
            terms.update(self._headline_terms(value))
        for value in cluster.get("cluster_summary_bullets") or []:
            terms.update(self._headline_terms(value))
        terms.update(self._headline_terms(cluster.get("theme_label")))
        return terms

    def _headline_terms(self, text: Optional[str]) -> set[str]:
        if not text or not isinstance(text, str):
            return set()
        tokens = {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
            if token not in {
                "with", "from", "that", "this", "will", "have", "after", "amid",
                "into", "over", "under", "news", "says", "said", "more", "than",
                "stock", "stocks", "share", "shares", "company", "market",
            }
        }
        return tokens

    def _clean_upper_list(self, values: Optional[Iterable[Any]]) -> List[str]:
        if not values:
            return []
        out = []
        seen = set()
        for value in values:
            cleaned = self._clean_upper(value)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
        return out

    def _clean_upper(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        value = value.strip().upper()
        return value or None

    def _normalize_text(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        value = value.strip().lower()
        return value or None

    def _safe_float(self, value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return self._coerce_datetime(value)
        if not isinstance(value, str) or not value.strip():
            return None

        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return self._coerce_datetime(parsed)

    def _coerce_datetime(self, value: Any) -> Optional[datetime]:
        if not isinstance(value, datetime):
            return None
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    def main():
        result = CatalystClusterService().cluster_recent_articles(days=2, unclustered_only=False)
        print(result)


    main()
