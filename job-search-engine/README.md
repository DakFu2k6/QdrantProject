# TopJob Search Engine

Hybrid semantic search engine for job database using Qdrant, built to support TopJob and AcademicGates with semantic search, keyword matching, and AI-powered ranking.

## Features

- **Semantic Search**: Find jobs by meaning, not just keywords
- **Keyword Search**: BM25-based exact keyword matching
- **Hybrid Search**: Combine semantic + keyword for best results
- **Metadata Filtering**: Filter by location, job type, skills, salary, etc.
- **Reranking**: Advanced reranking with bge-reranker-v2-m3
- **Business Scoring**: Custom ranking based on business rules
- **Incremental Sync**: Continuous database updates
- **REST API**: FastAPI-based search service

## Architecture

```
Job Database (PostgreSQL/MySQL)
    ↓
Data Processing Pipeline
    ├─ Normalize & validate
    ├─ Build searchable text
    └─ Extract metadata
    ↓
Embedding Service
    ├─ Dense embeddings (Qwen3-Embedding-4B)
    └─ BM25 sparse vectors
    ↓
Qdrant Vector Database
    ├─ Dense vector index
    ├─ Payload metadata indexes
    └─ Sparse vector support
    ↓
Hybrid Search + RRF Fusion
    ├─ Dense semantic search
    ├─ BM25 keyword search
    └─ Rank fusion
    ↓
Reranker
    ├─ bge-reranker-v2-m3
    └─ Business scoring
    ↓
Search API (FastAPI)
    ├─ Health checks
    ├─ Collection management
    └─ Job search endpoints
    ↓
Web UI / Chatbot / Recommendation Engine
```

## Installation

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- GPU (optional, for faster embeddings)

### Setup

1. **Clone and navigate to project**:
```bash
cd job-search-engine
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Copy environment file**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start services with Docker**:
```bash
docker-compose up -d
```

Check services are healthy:
```bash
curl http://localhost:6333/health  # Qdrant
curl http://localhost:5432        # PostgreSQL
```

## Quick Start

### 1. Create Qdrant Collection

```bash
python scripts/01_create_collection.py
```

### 2. Ingest Sample Jobs

```bash
python scripts/02_ingest_jobs.py
```

### 3. Test Semantic Search

```bash
python scripts/03_test_search.py
```

### 4. Start API Server

```bash
uvicorn app.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for interactive API documentation.

### 5. Test Search Endpoint

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "junior data analyst python sql",
    "location": "Hanoi",
    "limit": 10
  }'
```

## Project Structure

```
job-search-engine/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   ├── schemas.py              # Pydantic models
│   ├── qdrant_client.py        # Qdrant wrapper
│   ├── embedding_service.py    # Embedding generation
│   ├── data_processing.py      # Data utilities
│   ├── search_service.py       # Search logic (TODO)
│   ├── rerank_service.py       # Reranking (TODO)
│   └── business_ranker.py      # Business scoring (TODO)
├── scripts/
│   ├── 01_create_collection.py # Setup collection
│   ├── 02_ingest_jobs.py       # Load jobs into Qdrant
│   ├── 03_test_search.py       # Test semantic search
│   ├── 04_sync_jobs.py         # Incremental sync
│   └── 05_evaluate.py          # Evaluation metrics
├── data/
│   ├── sample_jobs.csv         # Sample data
│   ├── test_queries.json       # Test queries
│   └── relevance_labels.json   # Ground truth labels
├── notebooks/
│   ├── 01_data_cleaning.ipynb         # Data exploration
│   ├── 02_embedding_test.ipynb        # Embedding validation
│   ├── 03_search_evaluation.ipynb     # Search quality analysis
│   └── 04_error_analysis.ipynb        # Error investigation
├── docker-compose.yml          # Service composition
├── requirements.txt            # Python dependencies
├── .env                        # Configuration
└── README.md                   # This file
```

## API Endpoints

### Health Check
```
GET /health
```

### Search Jobs
```
POST /search
Content-Type: application/json

{
  "query": "data analyst python sql",
  "location": "Hanoi",
  "country": "Vietnam",
  "job_type": "Full-time",
  "career_level": "Entry level",
  "skills": ["Python", "SQL"],
  "industry": "Information Technology",
  "salary_min": 10000000,
  "salary_max": 50000000,
  "limit": 10,
  "use_hybrid": true,
  "use_reranker": true
}
```

### Get Collection Info
```
GET /collections
```

### Admin: Create Collection
```
POST /admin/create-collection
```

### Admin: Recreate Collection
```
POST /admin/recreate-collection
```

## Configuration

Edit `.env` file:

```env
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=password
DB_NAME=topjob

# Embedding
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
EMBEDDING_DIMENSION=2560
BATCH_SIZE=16

# Reranker
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
```

## Models

### Embedding Model
- **Primary**: `Qwen/Qwen3-Embedding-4B` (100+ languages, 32k context)
- **Alternative**: `BAAI/bge-m3` (lightweight, multilingual)

### Reranker Model
- **Primary**: `BAAI/bge-reranker-v2-m3` (multilingual, lightweight)
- **Alternative**: `Qwen/Qwen3-Reranker-4B`

## Development

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black app/ scripts/
```

### Type Checking
```bash
mypy app/
```

### Lint Code
```bash
flake8 app/ scripts/
```

## Performance Optimization

### Batch Processing
- Increase `batch_size` in `.env` for faster embeddings
- Use GPU for embedding generation
- Cache embeddings for common queries

### Database
- Create indexes on frequently filtered fields
- Use connection pooling
- Enable query caching

### Caching
- Redis cache for frequent queries
- Embedding cache
- Search result cache (with TTL)

## Monitoring

- Check collection status: `/collections`
- Monitor API health: `/health`
- Log all queries and search quality
- Track click-through rates and conversions

## Data Schema

### Jobs Table
```sql
CREATE TABLE jobs (
    id BIGINT PRIMARY KEY,
    title TEXT,
    company_name TEXT,
    description TEXT,
    requirements TEXT,
    responsibilities TEXT,
    benefits TEXT,
    location TEXT,
    country TEXT,
    job_type TEXT,
    career_level TEXT,
    salary_min NUMERIC,
    salary_max NUMERIC,
    currency VARCHAR(10),
    skills TEXT,
    industry TEXT,
    source_url TEXT,
    posted_at TIMESTAMP,
    expired_at TIMESTAMP,
    updated_at TIMESTAMP,
    status VARCHAR(20)
);
```

## Roadmap

- [ ] Implement full search service with hybrid search
- [ ] Add BM25 keyword search integration
- [ ] Implement reranking service
- [ ] Add business scoring logic
- [ ] Create incremental sync pipeline
- [ ] Build evaluation metrics (Precision@10, NDCG@10, etc.)
- [ ] Add web UI
- [ ] Implement RAG for job recommendations
- [ ] Add user profile matching
- [ ] Build chatbot integration

## Troubleshooting

### Qdrant Connection Issues
```bash
# Check Qdrant health
curl http://localhost:6333/health

# Check collection exists
curl http://localhost:6333/collections
```

### Embedding Service Issues
```bash
# Test embedding model
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('Qwen/Qwen3-Embedding-4B'); print(m.encode(['test']).shape)"
```

### Out of Memory
- Reduce `batch_size` in `.env`
- Use CPU instead of GPU
- Reduce embedding dimension

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [FastAPI](https://fastapi.tiangolo.com/)

## License

Proprietary - TopJob/AcademicGates Project

## Support

For issues or questions, contact the development team.
