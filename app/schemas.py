from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class JobPayload(BaseModel):
    job_id: int
    title: str
    company_name: str
    location: str
    country: str
    job_type: str
    career_level: str
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    currency: str
    skills: List[str]
    industry: str
    source_url: str
    posted_at: Optional[str] = None
    expired_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: str = "active"
    search_text: str

class JobSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    location: Optional[str] = None
    country: Optional[str] = None
    job_type: Optional[str] = None
    career_level: Optional[str] = None
    skills: Optional[List[str]] = None
    industry: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    limit: int = Field(10, ge=1, le=100, description="Number of results")
    use_hybrid: bool = True
    use_reranker: bool = True

class JobResult(BaseModel):
    job_id: int
    title: str
    company_name: str
    location: str
    country: str
    job_type: str
    career_level: str
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    skills: List[str]
    industry: str
    source_url: str
    posted_at: Optional[str] = None
    dense_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    business_score: float = 0.0
    final_score: float
    reason: Optional[str] = None

class JobSearchResponse(BaseModel):
    query: str
    total: int
    results: List[JobResult]
    search_time_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    qdrant_connected: bool
    database_connected: bool
    embedding_service_ready: bool

class DatabaseJobRecord(BaseModel):
    id: int
    title: str
    company_name: str
    description: str
    requirements: str
    responsibilities: str
    benefits: str
    location: str
    country: str
    job_type: str
    career_level: str
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    currency: str
    skills: str  
    industry: str
    source_url: str
    posted_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: str = "active"

class SyncState(BaseModel):
    last_synced_at: Optional[datetime] = None
    last_synced_count: int = 0
    total_jobs_in_db: int = 0
    total_jobs_in_qdrant: int = 0
