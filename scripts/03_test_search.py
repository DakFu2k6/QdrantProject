import logging
from app.qdrant_client import qdrant_store
from app.embedding_service import get_embedding_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_semantic_search():
    embedding_service = get_embedding_service()
    test_queries = [
        "junior data analyst python sql",
        "business intelligence internship",
        "python developer remote",
        "fresh graduate data analysis job",
    ]
    for query in test_queries:
        logger.info(f"\nSearching for: {query}")
        query_embedding = embedding_service.embed_single(query)
        results = qdrant_store.search(query_embedding, limit=5)
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            payload = result.payload
            logger.info(
                f"  {i}. {payload['title']} at {payload['company_name']} "
                f"(score: {result.score:.3f})"
            )

if __name__ == "__main__":
    test_semantic_search()
