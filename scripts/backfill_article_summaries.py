"""Preview by default; --apply enriches missing summaries without scraping or embeddings."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import certifi
from dotenv import dotenv_values
from pymongo import MongoClient

from services.article_enrichment import ArticleEnricher, backfill_missing_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    config = dotenv_values(Path(__file__).resolve().parents[1] / '.env')
    client = MongoClient(config['MONGO_URI'], tlsCAFile=certifi.where(),
                         serverSelectionTimeoutMS=15000)
    llm = None
    if args.apply:
        from services.llm.gemini_service import GeminiService
        llm = GeminiService(api_key=config['GEMINI_API_KEY'])
    try:
        result = backfill_missing_summaries(
            client.dev.articles, ArticleEnricher(llm), batch_size=args.batch_size,
            limit_total=args.limit, workers=args.workers, dry_run=not args.apply,
            progress=lambda r: print(json.dumps(r), flush=True),
        )
        print(json.dumps(result), flush=True)
        return 1 if result['halted'] or result['failed'] else 0
    finally:
        client.close()


if __name__ == '__main__':
    raise SystemExit(main())
