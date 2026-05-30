import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self):
        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(
                "BAAI/bge-reranker-v2-m3",
                use_fp16=True,
            )
            self.is_available = True
            logger.info("FlagReranker loaded successfully")
        except ImportError:
            logger.warning("FlagEmbedding not available, reranking disabled")
            self.reranker = None
            self.is_available = False
        except Exception as e:
            logger.error(f"Error loading reranker: {e}")
            self.reranker = None
            self.is_available = False
    
    def compute_rerank_scores(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[float]:
        if not self.is_available or self.reranker is None:
            logger.warning("Reranker not available")
            return [0.0] * len(candidates)
        try:
            pairs = []
            for candidate in candidates:
                doc_text = candidate.get("search_text", "")
                pairs.append([query, doc_text])
            scores = self.reranker.compute_score(pairs)
            return [float(score) for score in scores]
        except Exception as e:
            logger.error(f"Error computing rerank scores: {e}")
            return [0.0] * len(candidates)
    
    def calculate_business_score(
        self,
        job_payload: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> float:
        score = 0.0
        if job_payload.get("status") == "active":
            score += 1.0
        score += 0.5
        if job_payload.get("salary_min") or job_payload.get("salary_max"):
            score += 0.3
        if user_profile and user_profile.get("skills"):
            user_skills = set(s.lower() for s in user_profile.get("skills", []))
            job_skills = set(s.lower() for s in job_payload.get("skills", []))
            overlap = len(user_skills.intersection(job_skills))
            score += min(overlap * 0.2, 1.0)  
        if user_profile and user_profile.get("location"):
            user_location = user_profile.get("location", "").lower()
            job_location = job_payload.get("location", "").lower()
            if user_location and user_location in job_location:
                score += 0.2
        if user_profile and user_profile.get("career_level"):
            user_level = user_profile.get("career_level", "").lower()
            job_level = job_payload.get("career_level", "").lower()
            if user_level and user_level in job_level:
                score += 0.2
        return min(score, 5.0) 
    
    def rerank_results(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        user_profile: Optional[Dict[str, Any]] = None,
        use_ml_reranker: bool = True,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        candidates_with_text = []
        for candidate in candidates:
            if "search_text" not in candidate:
                from app.data_processing import build_job_text
                candidate["search_text"] = build_job_text(candidate)
            candidates_with_text.append(candidate)
        rerank_scores = []
        if use_ml_reranker and self.is_available:
            rerank_scores = self.compute_rerank_scores(query, candidates_with_text)
        else:
            rerank_scores = [0.0] * len(candidates_with_text)
        business_scores = [
            self.calculate_business_score(candidate, user_profile)
            for candidate in candidates_with_text
        ]
        def normalize(scores):
            if not scores or max(scores) == 0:
                return [0.0] * len(scores)
            max_score = max(scores)
            return [s / max_score if max_score > 0 else 0.0 for s in scores]
        norm_rerank = normalize(rerank_scores)
        norm_business = normalize(business_scores)
        combined_scores = []
        for i, candidate in enumerate(candidates_with_text):
            combined_score = (
                0.55 * norm_rerank[i]
                + 0.10 * norm_business[i]
                + 0.25 * candidate.get("dense_score", 0.0) / (candidate.get("dense_score", 1.0) or 1.0)
                + 0.10 * candidate.get("keyword_score", 0.0) / (candidate.get("keyword_score", 1.0) or 1.0)
            )
            candidate["rerank_score"] = float(rerank_scores[i]) if rerank_scores else 0.0
            candidate["business_score"] = float(business_scores[i])
            candidate["final_score"] = combined_score
        ranked = sorted(
            candidates_with_text,
            key=lambda x: x["final_score"],
            reverse=True,
        )
        return ranked[:top_k]
    
_reranker_service = None

def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
