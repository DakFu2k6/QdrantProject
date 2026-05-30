from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class QdrantJobStore:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
        )
        self.collection_name = settings.collection_name

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def create_collection(self) -> None:
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def create_payload_indexes(self) -> None:
        payload_indexes = {
            "job_id": models.PayloadSchemaType.INTEGER,
            "title": models.PayloadSchemaType.TEXT,
            "company_name": models.PayloadSchemaType.KEYWORD,
            "location": models.PayloadSchemaType.KEYWORD,
            "country": models.PayloadSchemaType.KEYWORD,
            "job_type": models.PayloadSchemaType.KEYWORD,
            "career_level": models.PayloadSchemaType.KEYWORD,
            "industry": models.PayloadSchemaType.KEYWORD,
            "skills": models.PayloadSchemaType.KEYWORD,
            "salary_min": models.PayloadSchemaType.FLOAT,
            "salary_max": models.PayloadSchemaType.FLOAT,
            "posted_at": models.PayloadSchemaType.DATETIME,
            "expired_at": models.PayloadSchemaType.DATETIME,
            "status": models.PayloadSchemaType.KEYWORD,
        }
        for field_name, field_schema in payload_indexes.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
                logger.info(f"Created payload index for '{field_name}'")
            except Exception as e:
                logger.debug(f"Payload index for '{field_name}' already exists or error: {e}")
    
    def upsert_points(self, points: List[models.PointStruct]) -> None:
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(f"Upserted {len(points)} points")
        except Exception as e:
            logger.error(f"Error upserting points: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[models.Filter] = None,
    ) -> List[models.ScoredPoint]:
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def delete_points(self, point_ids: List[int]) -> None:
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )
            logger.info(f"Deleted {len(point_ids)} points")
        except Exception as e:
            logger.error(f"Error deleting points: {e}")
            raise
    
    def get_collection_info(self) -> dict:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "vectors_count": info.points_count,
                "config": info.config,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def recreate_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
            self.create_collection()
            self.create_payload_indexes()
        except Exception as e:
            logger.error(f"Error recreating collection: {e}")
            raise

qdrant_store = QdrantJobStore()
