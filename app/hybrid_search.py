import logging
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from qdrant_client.http import models
from app.qdrant_client import qdrant_store
from app.embedding_service import get_embedding_service
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class BM25SearchEngine:
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.job_texts_to_id: Dict[int, int] = {}  
        self.is_initialized = False
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokens = re.findall(r"\w+", text.lower())
        return tokens
    
    def build_index(self, job_payloads: List[Dict[str, Any]]) -> None:
        logger.info(f"Building BM25 index for {len(job_payloads)} jobs...")
        self.corpus = []
        self.job_texts_to_id = {}
        for idx, payload in enumerate(job_payloads):
            search_text = payload.get("search_text", "")
            self.corpus.append(search_text)
            self.job_texts_to_id[idx] = payload["job_id"]
        tokenized_corpus = [self.tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.is_initialized = True
        logger.info(f"BM25 index created with {len(self.corpus)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        if not self.is_initialized or self.bm25 is None:
            logger.warning("BM25 not initialized, returning empty results")
            return []
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            job_id = self.job_texts_to_id.get(idx)
            score = float(scores[idx])
            if job_id is not None and score > 0:
                results.append((job_id, score))
        return results

class HybridSearchService:
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.bm25_engine = BM25SearchEngine()
        self.job_payloads_cache: List[Dict[str, Any]] = []
    
    def initialize_bm25_index(self, job_payloads: Optional[List[Dict[str, Any]]] = None) -> None:
        try:
            if job_payloads:
                self.job_payloads_cache = job_payloads
            else:
                logger.info("Fetching jobs from Qdrant for BM25 indexing...")
                # TODO: Implement scroll/fetch all points from Qdrant
                # For now, assuming payloads are provided externally
                pass
            
            if self.job_payloads_cache:
                self.bm25_engine.build_index(self.job_payloads_cache)
                logger.info("BM25 index initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing BM25 index: {e}")
            raise
    
    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[Tuple[int, float]]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        fused_scores: Dict[int, float] = {}
        for results in result_lists:
            for rank, (job_id, _) in enumerate(results, start=1):
                rrf_score = 1.0 / (k + rank)
                fused_scores[job_id] = fused_scores.get(job_id, 0) + rrf_score
        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked
    
    def dense_search(
        self,
        query: str,
        limit: int = 50,
        query_filter: Optional[models.Filter] = None,
    ) -> List[Tuple[int, float]]:
        try:
            query_embedding = self.embedding_service.embed_single(query)
            results = qdrant_store.search(
                query_vector=query_embedding,
                limit=limit,
                query_filter=query_filter,
            )
            job_results = []
            for result in results:
                job_id = result.payload.get("job_id")
                score = float(result.score)
                if job_id is not None:
                    job_results.append((job_id, score))
            return job_results
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []
    
    def bm25_search(
        self,
        query: str,
        limit: int = 50,
    ) -> List[Tuple[int, float]]:
        try:
            if not self.bm25_engine.is_initialized:
                logger.warning("BM25 engine not initialized")
                return []
            return self.bm25_engine.search(query, top_k=limit)
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        query_filter: Optional[models.Filter] = None,
        dense_weight: float = 0.5,
        keyword_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Hybrid search for query: {query}")
            dense_results = self.dense_search(query, limit=100, query_filter=query_filter)
            logger.info(f"Dense search returned {len(dense_results)} results")
            bm25_results = self.bm25_search(query, limit=100)
            logger.info(f"BM25 search returned {len(bm25_results)} results")
            if dense_results and bm25_results:
                fused = self.reciprocal_rank_fusion([dense_results, bm25_results])
            elif dense_results:
                fused = dense_results
            elif bm25_results:
                fused = bm25_results
            else:
                logger.warning("No results from both search methods")
                return []
            top_job_ids = [job_id for job_id, _ in fused[:limit]]
            results = []
            for job_id in top_job_ids:
                dense_score = next(
                    (score for jid, score in dense_results if jid == job_id),
                    0.0
                )
                keyword_score = next(
                    (score for jid, score in bm25_results if jid == job_id),
                    0.0
                )
                payload = next(
                    (p for p in self.job_payloads_cache if p["job_id"] == job_id),
                    None
                )
                if payload:
                    result = {
                        "job_id": payload["job_id"],
                        "title": payload["title"],
                        "company_name": payload["company_name"],
                        "location": payload["location"],
                        "country": payload["country"],
                        "job_type": payload["job_type"],
                        "career_level": payload["career_level"],
                        "salary_min": payload.get("salary_min"),
                        "salary_max": payload.get("salary_max"),
                        "skills": payload.get("skills", []),
                        "industry": payload["industry"],
                        "source_url": payload["source_url"],
                        "posted_at": payload.get("posted_at"),
                        "dense_score": dense_score,
                        "keyword_score": keyword_score,
                        "combined_score": dense_score + keyword_score,
                    }
                    results.append(result)
            logger.info(f"Hybrid search returned {len(results)} final results")
            return results
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise

_hybrid_search_service = None

def get_hybrid_search_service() -> HybridSearchService:
    global _hybrid_search_service
    if _hybrid_search_service is None:
        _hybrid_search_service = HybridSearchService()
    return _hybrid_search_service
