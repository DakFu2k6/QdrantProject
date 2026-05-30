import logging
import pandas as pd
from typing import List
from tqdm import tqdm
from qdrant_client.http import models
from app.qdrant_client import qdrant_store
from app.embedding_service import get_embedding_service
from app.data_processing import normalize_job, job_to_payload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_jobs() -> pd.DataFrame:
    sample_data = [
        {
            "id": 1,
            "title": "Junior Data Analyst",
            "company_name": "ABC Technology",
            "description": "Looking for a junior data analyst to join our team",
            "requirements": "Python, SQL, Power BI experience required",
            "responsibilities": "Analyze data and create reports",
            "benefits": "Competitive salary, health insurance",
            "location": "Hanoi",
            "country": "Vietnam",
            "job_type": "Full-time",
            "career_level": "Entry level",
            "salary_min": 10000000,
            "salary_max": 18000000,
            "currency": "VND",
            "skills": "Python, SQL, Power BI",
            "industry": "Information Technology",
            "source_url": "https://example.com/job1",
            "posted_at": "2026-05-01",
            "expired_at": "2026-06-01",
            "updated_at": "2026-05-01",
            "status": "active",
        },
        {
            "id": 2,
            "title": "Business Intelligence Intern",
            "company_name": "XYZ Corp",
            "description": "Internship in business intelligence and data analysis",
            "requirements": "Excel, SQL, basic statistics knowledge",
            "responsibilities": "Support BI team with data analysis and reporting",
            "benefits": "Flexible hours, learning opportunities",
            "location": "Ho Chi Minh City",
            "country": "Vietnam",
            "job_type": "Internship",
            "career_level": "Internship",
            "salary_min": 5000000,
            "salary_max": 8000000,
            "currency": "VND",
            "skills": "Excel, SQL, Statistics",
            "industry": "Information Technology",
            "source_url": "https://example.com/job2",
            "posted_at": "2026-05-05",
            "expired_at": "2026-06-05",
            "updated_at": "2026-05-05",
            "status": "active",
        },
        {
            "id": 3,
            "title": "Senior Python Developer",
            "company_name": "Tech Solutions",
            "description": "Experienced Python developer for backend development",
            "requirements": "5+ years Python, Django, PostgreSQL",
            "responsibilities": "Design and implement backend systems",
            "benefits": "High salary, remote work",
            "location": "Remote",
            "country": "Vietnam",
            "job_type": "Full-time",
            "career_level": "Senior",
            "salary_min": 40000000,
            "salary_max": 60000000,
            "currency": "VND",
            "skills": "Python, Django, PostgreSQL, Docker",
            "industry": "Information Technology",
            "source_url": "https://example.com/job3",
            "posted_at": "2026-05-10",
            "expired_at": "2026-07-10",
            "updated_at": "2026-05-10",
            "status": "active",
        },
    ]
    
    return pd.DataFrame(sample_data)

def ingest_jobs(batch_size: int = 16):
    logger.info("Loading embedding service...")
    embedding_service = get_embedding_service()
    logger.info("Loading sample jobs...")
    df = load_sample_jobs()
    logger.info(f"Processing {len(df)} jobs...")
    jobs = [normalize_job(row) for _, row in df.iterrows()]
    texts = [job_to_payload(job)["search_text"] for job in jobs]
    logger.info("Generating embeddings...")
    embeddings = embedding_service.embed_texts(texts, batch_size=batch_size)
    logger.info("Creating points...")
    points = []
    for job, embedding in zip(jobs, embeddings):
        payload = job_to_payload(job)
        point = models.PointStruct(
            id=job["job_id"],
            vector=embedding.tolist(),
            payload=payload,
        )
        points.append(point)
    logger.info(f"Upserting {len(points)} points...")
    qdrant_store.upsert_points(points)
    logger.info("Done!")

if __name__ == "__main__":
    ingest_jobs()
