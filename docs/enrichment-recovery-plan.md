# Enrichment recovery

Scope: fix ticker-only enrichment skips, persist status and attempts, retry transient
failures, and backfill articles missing a usable short summary. Preserve local changes.

1. Add offline regression tests for eligibility, retries, terminal errors, persistence,
   preservation of populated fields, dry-run, and resumable backfill. Verify they fail.
2. Share enrichment behavior between ingestion and backfill; record pending/complete/failed,
   attempts, timestamp, model, and sanitized error. Retry transient failures at most three
   times with exponential backoff. Verify the focused tests and available suite.
3. Preview the database scope without LLM calls or writes, run a small canary, verify stored
   summaries and website API, then process remaining candidates with bounded concurrency.

Use existing MongoDB and Gemini integration. No new service or scraper requests. A completed
article has a nonblank summary_short. Other summary formats and analysis remain optional.
Backfill fills missing fields and preserves populated data. Failed articles stay eligible
for a later explicit run. Cloud deployment is separate from the database backfill.

## Execution record

- 18 focused tests passed; the existing trading-system test files fail collection.
- The configured 2.0 model returned 404; 2.5 was also unavailable to this key.
  A live Gemini 3.5 Flash-Lite request and three-article database canary succeeded.
- User requested the backfill stop: process exited, 1,359 summaries saved and
  2,378 short summaries remain missing. Do not resume without a new request.
- Cloud function deployment was already running when the backfill was stopped.
- Cloud deployment completed: article-pipeline-run is ACTIVE (2026-10-23).
- At user request, repaired only the 100 latest ingested articles: 100 completed,
  zero failures; source fields preserved and older missing-summary count unchanged
  at 2,278. The broader backfill remains stopped. Scoped regression suite: 19 passed.
