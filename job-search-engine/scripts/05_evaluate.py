import logging
import json
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_set() -> List[dict]:
    return [
        {
            "query": "data analyst python sql",
            "relevant_job_ids": [1, 2],
        },
        {
            "query": "junior data analyst entry level",
            "relevant_job_ids": [1, 2],
        },
        {
            "query": "business intelligence internship",
            "relevant_job_ids": [2],
        },
        {
            "query": "python developer senior remote",
            "relevant_job_ids": [3],
        },
    ]

def evaluate():
    logger.info("Evaluating search quality...")
    test_set = load_test_set()
    logger.info(f"Loaded {len(test_set)} test queries")
    
    # TODO: Implement actual evaluation
    # Metrics to compute:
    # - Precision@10
    # - Recall@10
    # - MRR@10
    # - NDCG@10
    # - HitRate@10

if __name__ == "__main__":
    evaluate()
