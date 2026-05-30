"""
topjob-qdrant-search
====================

Job Search Engine using Qdrant Vector Database

This project provides a hybrid semantic search engine for job database
with support for:
- Semantic similarity search using dense embeddings
- Keyword-based search using BM25
- Hybrid search with rank fusion (RRF)
- Advanced reranking
- Metadata filtering
- Real-time incremental sync

Recommended workflow:
1. docker-compose up -d          # Start Qdrant, PostgreSQL, Redis
2. python scripts/01_create_collection.py   # Setup collection
3. python scripts/02_ingest_jobs.py         # Load jobs
4. python scripts/03_test_search.py         # Test search
5. uvicorn app.main:app --reload             # Start API
6. curl http://localhost:8000/docs          # Test via Swagger UI
"""

__version__ = "1.0.0"
