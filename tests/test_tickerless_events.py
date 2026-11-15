import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bson import ObjectId

# Import the clustering rules without bootstrapping the live Mongo/Gemini clients.
spec = importlib.util.spec_from_file_location("clusters", Path(__file__).parents[1] / "services/database/catalyst_cluster_service.py")
module = importlib.util.module_from_spec(spec)
with patch.dict("sys.modules", {"services.llm": SimpleNamespace(gemini_client=None)}):
    spec.loader.exec_module(module)


def test_tickerless_event_eligibility_and_distinct_decisions():
    service = module.CatalystClusterService(db=MagicMock())
    service.collection.find_one.return_value = None
    now = datetime.now(timezone.utc)
    title = "Canada lifts steel import tariff"
    article = {"_id": ObjectId(), "url": "https://example.test/steel", "event_type": "regulation",
               "title": title, "importance_score": .7, "summary_short": title, "publish_date": now}
    assert service._clusterability_reason(article) is None
    assert service._clusterability_reason({**article, "event_type": "other"}) == "missing_tickers"
    assert service._clusterability_reason({**article, "importance_score": .1}) == "missing_tickers"
    cluster = {"event_type": "regulation", "titles": ["Canada lifts fishing quotas"], "last_seen_at": now}
    with patch.object(service, "_find_candidate_clusters", return_value=[cluster]):
        assert service._cluster_single_article(article)["action"] == "created"
    doc = service.collection.update_one.call_args.args[1]["$setOnInsert"]
    assert doc["tickers"] == [] and doc["cluster_id"] == f"cat_{article['_id']}"
    cluster["titles"] = [title]
    with patch.object(service, "_find_candidate_clusters", return_value=[cluster]), patch.object(service, "_attach_article_to_cluster") as attach:
        assert service._cluster_single_article(article)["action"] == "attached"
        attach.assert_called_once()
    service.collection.find_one.return_value = {"cluster_id": "cat_existing", "cluster_version": 2}
    assert service._cluster_single_article(article)["action"] == "attached"
    assert service.collection.update_one.call_count == 1
