import logging
import json
from datetime import datetime
from app.qdrant_client import qdrant_store
from app.embedding_service import get_embedding_service
from app.hybrid_search import get_hybrid_search_service
from app.rerank_service import get_reranker_service
from app.data_processing import normalize_job, job_to_payload
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sample_jobs():
    return [
        {
            "id": 1,
            "title": "Junior Data Analyst",
            "company_name": "ABC Technology",
            "description": "Looking for a junior data analyst to join our team. You will work with Python, SQL, and Power BI to analyze business data and create insights.",
            "requirements": "Python, SQL, Power BI experience required. Knowledge of statistics a plus.",
            "responsibilities": "Analyze data and create reports. Support senior analysts in data projects.",
            "benefits": "Competitive salary, health insurance, flexible work hours",
            "location": "Hanoi",
            "country": "Vietnam",
            "job_type": "Full-time",
            "career_level": "Entry level",
            "salary_min": 10000000,
            "salary_max": 18000000,
            "currency": "VND",
            "skills": "Python, SQL, Power BI, Excel",
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
            "description": "Join our BI team as an intern. You will support business intelligence initiatives and help create dashboards.",
            "requirements": "Excel, SQL, basic understanding of databases. Fresh graduates welcome.",
            "responsibilities": "Support BI team with data analysis and reporting. Create dashboards using Power BI or Tableau.",
            "benefits": "Flexible hours, mentorship, learning opportunities",
            "location": "Ho Chi Minh City",
            "country": "Vietnam",
            "job_type": "Internship",
            "career_level": "Internship",
            "salary_min": 5000000,
            "salary_max": 8000000,
            "currency": "VND",
            "skills": "Excel, SQL, Power BI, Statistics",
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
            "description": "Experienced Python developer for backend development. Build scalable systems using modern tech stack.",
            "requirements": "5+ years Python, Django or FastAPI, PostgreSQL required. Docker and Kubernetes a plus.",
            "responsibilities": "Design and implement backend systems. Lead technical decisions. Mentor junior developers.",
            "benefits": "High salary, remote work, stock options",
            "location": "Remote",
            "country": "Vietnam",
            "job_type": "Full-time",
            "career_level": "Senior",
            "salary_min": 40000000,
            "salary_max": 60000000,
            "currency": "VND",
            "skills": "Python, Django, PostgreSQL, Docker, Kubernetes",
            "industry": "Information Technology",
            "source_url": "https://example.com/job3",
            "posted_at": "2026-05-10",
            "expired_at": "2026-07-10",
            "updated_at": "2026-05-10",
            "status": "active",
        },
        {
            "id": 4,
            "title": "Data Engineer",
            "company_name": "DataFlow Inc",
            "description": "Build data pipelines and infrastructure. Work with big data technologies.",
            "requirements": "Python, SQL, Spark knowledge. Experience with Hadoop or cloud platforms.",
            "responsibilities": "Design ETL pipelines. Maintain data infrastructure. Optimize data workflows.",
            "benefits": "Competitive salary, training budget, work-life balance",
            "location": "Hanoi",
            "country": "Vietnam",
            "job_type": "Full-time",
            "career_level": "Mid-level",
            "salary_min": 20000000,
            "salary_max": 35000000,
            "currency": "VND",
            "skills": "Python, SQL, Spark, Hadoop",
            "industry": "Information Technology",
            "source_url": "https://example.com/job4",
            "posted_at": "2026-05-12",
            "expired_at": "2026-07-12",
            "updated_at": "2026-05-12",
            "status": "active",
        },
        {
            "id": 5,
            "title": "Analytics Reporting Officer",
            "company_name": "Finance Pro",
            "description": "Create financial and business reports. Work with dashboarding tools.",
            "requirements": "Excel expert. Power BI or Tableau experience. SQL basics required.",
            "responsibilities": "Create reports and dashboards. Analyze business metrics. Present findings.",
            "benefits": "Stable job, good benefits package",
            "location": "Ho Chi Minh City",
            "country": "Vietnam",
            "job_type": "Full-time",
            "career_level": "Mid-level",
            "salary_min": 15000000,
            "salary_max": 25000000,
            "currency": "VND",
            "skills": "Excel, Power BI, SQL, Reporting",
            "industry": "Finance",
            "source_url": "https://example.com/job5",
            "posted_at": "2026-05-08",
            "expired_at": "2026-06-08",
            "updated_at": "2026-05-08",
            "status": "active",
        },
    ]

def test_hybrid_search():
    logger.info("=" * 80)
    logger.info("TopJob Hybrid Search Test")
    logger.info("=" * 80)
    embedding_service = get_embedding_service()
    hybrid_search_service = get_hybrid_search_service()
    reranker_service = get_reranker_service()
    logger.info("\nPreparing sample jobs...")
    sample_jobs = get_sample_jobs()
    jobs = [normalize_job(row) for row in sample_jobs]
    payloads = [job_to_payload(job) for job in jobs]
    logger.info("Initializing BM25 index...")
    hybrid_search_service.initialize_bm25_index(payloads)
    hybrid_search_service.job_payloads_cache = payloads
    test_queries = [
        {
            "query": "junior data analyst python sql power bi",
            "description": "Looking for entry-level data analyst with specific skills",
        },
        {
            "query": "business intelligence fresh graduate",
            "description": "Fresh graduate interested in BI",
        },
        {
            "query": "python developer remote senior",
            "description": "Senior Python developer, remote work",
        },
        {
            "query": "data engineer hadoop spark",
            "description": "Data engineer with big data experience",
        },
        {
            "query": "hanoi analytics reporting",
            "description": "Analytics job in Hanoi",
        },
    ]
    for test_case in test_queries:
        query = test_case["query"]
        description = test_case["description"]
        logger.info("\n" + "=" * 80)
        logger.info(f"Test: {description}")
        logger.info(f"Query: '{query}'")
        logger.info("-" * 80)
        try:
            candidates = hybrid_search_service.hybrid_search(
                query=query,
                limit=50,
                query_filter=None,
            )
            logger.info(f"Hybrid search returned {len(candidates)} candidates")
            logger.info("\nBefore reranking (top 5):")
            for i, job in enumerate(candidates[:5], 1):
                logger.info(
                    f"  {i}. {job['title']:<30} @ {job['company_name']:<20} "
                    f"[Dense: {job.get('dense_score', 0):.3f}, "
                    f"Keyword: {job.get('keyword_score', 0):.3f}]"
                )
            ranked = reranker_service.rerank_results(
                query=query,
                candidates=candidates,
                top_k=5,
                user_profile=None,
                use_ml_reranker=reranker_service.is_available,
            )
            logger.info(f"\nAfter reranking (top 5):")
            for i, job in enumerate(ranked, 1):
                logger.info(
                    f"  {i}. {job['title']:<30} @ {job['company_name']:<20} "
                    f"[Final: {job.get('final_score', 0):.3f}, "
                    f"Rerank: {job.get('rerank_score', 0):.3f}, "
                    f"Business: {job.get('business_score', 0):.3f}]"
                )
        except Exception as e:
            logger.error(f"Error in search: {e}", exc_info=True)
    logger.info("\n" + "=" * 80)
    logger.info("Hybrid search test completed!")
    logger.info("=" * 80)

if __name__ == "__main__":
    test_hybrid_search()
