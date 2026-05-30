import logging
from app.qdrant_client import qdrant_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Creating Qdrant collection...")
    try:
        qdrant_store.create_collection()
        logger.info("Collection created successfully")
        logger.info("Creating payload indexes...")
        qdrant_store.create_payload_indexes()
        logger.info("Payload indexes created successfully")
        info = qdrant_store.get_collection_info()
        logger.info(f"Collection info: {info}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
