# Ingestion repair

Keep the four existing publishers, scheduled frequency, per-source article limit, and
scraper concurrency. Replace rejected discovery endpoints with public publisher feeds;
repair Yahoo selectors and use parsed page text without optional local NLP resources.
Feed excerpts are labelled in stored article metadata. No paywall or challenge bypass.

1. Offline regression tests: feed parsing/HTML cleaning, normalized URL deduplication
   before limits, next unseen article selection, transport/extraction errors, idle vs
   failed source status, and cloud handler responses.
2. Implement discovery/extraction repairs and aggregate source/enrichment status into run
   records. All-source failure returns HTTP 503; duplicates-only runs remain no_new.
3. Run focused tests, fetch at most one article per publisher without DB writes, then
   deploy from the existing deployment source plus reviewed changed files. Verify one
   production run's saved article and run counts through MongoDB and the public API.

Tradeoff: MarketWatch/Seeking Alpha may supply public excerpts rather than full text.
Benzinga's reachable public feed currently contains investing/crypto content, while its
stock-news JSON endpoint is blocked. Record this limitation and retain source links.
The broader summary backfill remains stopped. Do not alter the stock catalog or cleanup.

## Verification

- 37 focused ingestion/enrichment tests passed. Three unrelated trading-system
  test files still fail collection due to missing exported classes.
- Python 3.11 startup and one-item-per-source live fetch checks passed for Yahoo,
  MarketWatch, and Benzinga. Seeking Alpha discovery succeeds but linked pages
  can return 403; the user confirmed no authorized API access is available.
- Deployed article-pipeline-run revision article-pipeline-run-00025-miz.
  Production canary inserted three enriched articles; the next scheduled run
  inserted two more. Five saved summaries were verified through the public API.
- Scheduled ingestion stays enabled every five minutes. Summary backfill stays
  paused. No stock-catalog recovery or cleanup changes were made.
