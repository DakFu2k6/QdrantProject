import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.config import settings
from app.schemas import (
    JobSearchRequest,
    JobSearchResponse,
    HealthCheckResponse,
    JobResult,
)
from app.qdrant_client import qdrant_store
from app.embedding_service import get_embedding_service
from app.hybrid_search import get_hybrid_search_service
from app.rerank_service import get_reranker_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TopJob Search Engine",
    description="Hybrid semantic search engine for job database using Qdrant",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up TopJob Search Engine...")
    if not qdrant_store.health_check():
        logger.warning("Qdrant is not available on startup")
    else:
        logger.info("Connected to Qdrant")
    try:
        embedding_service = get_embedding_service()
        logger.info(f"Embedding service initialized with dimension: {embedding_service.get_embedding_dimension()}")
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
    try:
        hybrid_search = get_hybrid_search_service()
        logger.info("Hybrid search service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize hybrid search service: {e}")
    try:
        reranker = get_reranker_service()
        if reranker.is_available:
            logger.info("Reranker service initialized")
        else:
            logger.warning("Reranker not available, will use fallback scoring")
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down TopJob Search Engine...")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    qdrant_ok = qdrant_store.health_check()
    embedding_ok = False
    try:
        embedding_service = get_embedding_service()
        embedding_ok = embedding_service is not None
    except Exception as e:
        logger.error(f"Embedding service health check failed: {e}")
    return HealthCheckResponse(
        status="healthy" if (qdrant_ok and embedding_ok) else "degraded",
        qdrant_connected=qdrant_ok,
        database_connected=False,  # TODO: Add DB health check
        embedding_service_ready=embedding_ok,
    )

@app.post("/search", response_model=JobSearchResponse)
async def search(request: JobSearchRequest):
    start_time = time.time()
    try:
        hybrid_search_service = get_hybrid_search_service()
        reranker_service = get_reranker_service()
        query_filter = None
        if any([request.location, request.country, request.job_type, 
                request.career_level, request.industry, request.skills]):
            from qdrant_client.http import models as qmodels
            must_conditions = []
            must_conditions.append(
                qmodels.FieldCondition(
                    key="status",
                    match=qmodels.MatchValue(value="active")
                )
            )
            if request.location:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="location",
                        match=qmodels.MatchValue(value=request.location)
                    )
                )
            if request.country:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="country",
                        match=qmodels.MatchValue(value=request.country)
                    )
                )
            if request.job_type:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="job_type",
                        match=qmodels.MatchValue(value=request.job_type)
                    )
                )
            if request.career_level:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="career_level",
                        match=qmodels.MatchValue(value=request.career_level)
                    )
                )
            if request.industry:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key="industry",
                        match=qmodels.MatchValue(value=request.industry)
                    )
                )
            if request.skills:
                for skill in request.skills:
                    must_conditions.append(
                        qmodels.FieldCondition(
                            key="skills",
                            match=qmodels.MatchValue(value=skill)
                        )
                    )
            if must_conditions:
                query_filter = qmodels.Filter(must=must_conditions)
        candidates = hybrid_search_service.hybrid_search(
            query=request.query,
            limit=max(20, request.limit * 2),  
            query_filter=query_filter,
        )
        if not candidates:
            return JobSearchResponse(
                query=request.query,
                total=0,
                results=[],
                search_time_ms=(time.time() - start_time) * 1000,
            )
        if request.use_reranker:
            ranked = reranker_service.rerank_results(
                query=request.query,
                candidates=candidates,
                top_k=request.limit,
                user_profile=None,
                use_ml_reranker=reranker_service.is_available,
            )
        else:
            ranked = candidates[:request.limit]
        results = []
        for job in ranked:
            result = JobResult(
                job_id=job["job_id"],
                title=job["title"],
                company_name=job["company_name"],
                location=job["location"],
                country=job["country"],
                job_type=job["job_type"],
                career_level=job["career_level"],
                salary_min=job.get("salary_min"),
                salary_max=job.get("salary_max"),
                skills=job.get("skills", []),
                industry=job["industry"],
                source_url=job["source_url"],
                posted_at=job.get("posted_at"),
                dense_score=job.get("dense_score", 0.0),
                keyword_score=job.get("keyword_score", 0.0),
                rerank_score=job.get("rerank_score", 0.0),
                business_score=job.get("business_score", 0.0),
                final_score=job.get("final_score", 0.0),
                reason=f"Matched '{request.query}' with semantic score {job.get('dense_score', 0):.3f}",
            )
            results.append(result)
        search_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Search for '{request.query}' completed in {search_time_ms:.1f}ms, returned {len(results)} results")
        return JobSearchResponse(
            query=request.query,
            total=len(results),
            results=results,
            search_time_ms=search_time_ms,
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def get_collections():
    try:
        info = qdrant_store.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/create-collection")
async def create_collection():
    try:
        qdrant_store.create_collection()
        qdrant_store.create_payload_indexes()
        return {"status": "success", "message": "Collection created"}
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/recreate-collection")
async def recreate_collection():
    try:
        qdrant_store.recreate_collection()
        return {"status": "success", "message": "Collection recreated"}
    except Exception as e:
        logger.error(f"Error recreating collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
    )
