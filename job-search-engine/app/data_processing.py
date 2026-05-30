import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

def build_job_text(job: Dict[str, Any]) -> str:
    title = job.get("title", "")
    skills = job.get("skills", [])
    if isinstance(skills, list):
        skills_str = ", ".join(skills)
    else:
        skills_str = str(skills)
    requirements = job.get("requirements", "")
    responsibilities = job.get("responsibilities", "")
    description = job.get("description", "")
    location = job.get("location", "")
    industry = job.get("industry", "")
    job_type = job.get("job_type", "")
    career_level = job.get("career_level", "")
    text = f"""
Title: {title}
Title: {title}
Skills: {skills_str}
Skills: {skills_str}
Job Type: {job_type}
Career Level: {career_level}
Requirements: {requirements}
Responsibilities: {responsibilities}
Description: {description}
Location: {location}
Industry: {industry}
"""
    return text.strip()

def normalize_job(row: Dict[str, Any]) -> Dict[str, Any]:
    skills = row.get("skills", "")
    if isinstance(skills, str):
        skills_list = [s.strip() for s in skills.split(",") if s.strip()]
    elif isinstance(skills, list):
        skills_list = [str(s).strip() for s in skills if s]
    else:
        skills_list = []
    salary_min = row.get("salary_min")
    salary_max = row.get("salary_max")
    if salary_min is not None:
        try:
            salary_min = float(salary_min)
        except (ValueError, TypeError):
            salary_min = None
    if salary_max is not None:
        try:
            salary_max = float(salary_max)
        except (ValueError, TypeError):
            salary_max = None
    posted_at = row.get("posted_at")
    if posted_at:
        posted_at = str(posted_at)
    
    expired_at = row.get("expired_at")
    if expired_at:
        expired_at = str(expired_at)
    updated_at = row.get("updated_at")
    if updated_at:
        updated_at = str(updated_at)
    normalized = {
        "job_id": int(row.get("id", 0)),
        "title": row.get("title", ""),
        "company_name": row.get("company_name", ""),
        "description": row.get("description", ""),
        "requirements": row.get("requirements", ""),
        "responsibilities": row.get("responsibilities", ""),
        "benefits": row.get("benefits", ""),
        "location": row.get("location", ""),
        "country": row.get("country", ""),
        "job_type": row.get("job_type", ""),
        "career_level": row.get("career_level", ""),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "currency": row.get("currency", ""),
        "skills": skills_list,
        "industry": row.get("industry", ""),
        "source_url": row.get("source_url", ""),
        "posted_at": posted_at,
        "expired_at": expired_at,
        "updated_at": updated_at,
        "status": row.get("status", "active"),
    }
    return normalized

def job_to_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "title": job["title"],
        "company_name": job["company_name"],
        "location": job["location"],
        "country": job["country"],
        "job_type": job["job_type"],
        "career_level": job["career_level"],
        "salary_min": job["salary_min"],
        "salary_max": job["salary_max"],
        "currency": job["currency"],
        "skills": job["skills"],
        "industry": job["industry"],
        "source_url": job["source_url"],
        "posted_at": job["posted_at"],
        "expired_at": job["expired_at"],
        "updated_at": job["updated_at"],
        "status": job["status"],
        "search_text": build_job_text(job),
    }

def calculate_business_score(
    job_payload: Dict[str, Any],
    user_profile: Dict[str, Any] = None,
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
        score += overlap * 0.2
    return score
