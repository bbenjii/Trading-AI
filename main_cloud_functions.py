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
        pipeline.run()
        logger.info("Article pipeline run completed.")
        return "ok", 200
    except Exception as e:
        logger.error("Error occurred during article pipeline run: %s", e)
        return f"error: {e}", 500

if __name__ == "__main__":
    
    run_article_fetch_pipeline(None)