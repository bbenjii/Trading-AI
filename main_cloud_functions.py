from pipelines import ArticlePipelineController
from utils.logger import logger
import os
 
pipeline = ArticlePipelineController(article_limit=1, verify_db=True, asynchronous=True)
import nltk 
import os

def run_article_fetch_pipeline(request):
    """
       HTTP Cloud Function entrypoint
    """
    logger.info("Cloud Function triggered to run article pipeline.")
    try:
        result = pipeline.run()
        logger.info("Article pipeline completed: status=%s inserted=%s",
                    result["status"], result["articles_inserted"])
        return {
            "status": result["status"],
            "articles_fetched": result["articles_fetched"],
            "articles_inserted": result["articles_inserted"],
            "failed_sources": result["failed_sources"],
        }, 503 if result["status"] == "failed" else 200
    except Exception as e:
        logger.error("Article pipeline execution failed: %s", type(e).__name__)
        return {"status": "failed", "error": "Pipeline execution failed"}, 500

if __name__ == "__main__":
    
    run_article_fetch_pipeline(None)
