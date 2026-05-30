import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        logger.info(f"Model loaded. Output dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = None,
        normalize: bool = True,
    ) -> np.ndarray:
        if batch_size is None:
            batch_size = settings.batch_size
        logger.debug(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings
    
    def embed_single(self, text: str, normalize: bool = True) -> List[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embedding.tolist()
    
    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
