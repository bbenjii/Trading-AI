"""Preview the recent tickerless articles newly eligible for catalyst clustering."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone

from services.database import catalyst_cluster_service as clusters
from services.database.catalyst_cluster_service import TICKERLESS_EVENT_TYPES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    published = {"$convert": {"input": "$publish_date", "to": "date", "onError": None, "onNull": None}}
    query = {"cluster_id": None, "tickers.0": {"$exists": False}, "primary_ticker": {"$in": [None, ""]},
             "event_type": {"$in": sorted(TICKERLESS_EVENT_TYPES)}, "importance_score": {"$gte": 0.4},
             "$expr": {"$and": [{"$gte": [published, start]}, {"$lte": [published, end]}]}}
    articles = [a for a in clusters.article_collection.find(query).sort("publish_date", 1)
                if a.get("url") and clusters._clusterability_reason(a) is None]
    print(f"{clusters.article_collection.full_name}: {len(articles)} eligible", flush=True)
    for a in articles[:5]:
        print(a["title"], flush=True)
    if args.apply:
        counts = Counter()
        for article in articles:
            if clusters.article_collection.find_one({"_id": article["_id"], "cluster_id": None}, {"_id": 1}):
                result = clusters._cluster_single_article(article)
                counts[result["action"]] += 1
        print(dict(counts))


if __name__ == "__main__":
    main()
