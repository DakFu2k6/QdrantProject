# Hybrid Search Implementation Guide

## Overview

The TopJob Search Engine implements a sophisticated **hybrid search** system that combines:

1. **Dense Vector Search** - Semantic understanding using embeddings
2. **Sparse Keyword Search** - Exact keyword matching using BM25
3. **Rank Fusion** - Combining results using Reciprocal Rank Fusion (RRF)
4. **Reranking** - ML-based reranking with business rules
5. **Filtering** - Metadata-based filtering

## Architecture

```
User Query
    ↓
Dense Search                BM25 Search
(Embeddings)               (Keyword)
    ↓                           ↓
Qdrant Search          BM25 Index Search
    ↓                           ↓
Dense Results          Keyword Results
    ↓                           ↓
     └──→ RRF Fusion ←──┘
            ↓
      Fused Results
            ↓
      Reranker (ML)
            ↓
    Business Scoring
            ↓
      Final Results
```

## Components

### 1. HybridSearchService

Located in `app/hybrid_search.py`

**Responsibilities:**
- Dense semantic search using Qdrant
- BM25 keyword search using rank-bm25
- RRF (Reciprocal Rank Fusion) to merge results
- Metadata filtering support

**Key Methods:**

```python
# Dense semantic search
results = hybrid_search.dense_search(
    query="python data analysis",
    limit=50,
    query_filter=None
)

# BM25 keyword search
results = hybrid_search.bm25_search(
    query="python data analysis",
    limit=50
)

# Full hybrid search with RRF
results = hybrid_search.hybrid_search(
    query="python data analysis",
    limit=10,
    query_filter=metadata_filter
)
```

### 2. RerankerService

Located in `app/rerank_service.py`

**Responsibilities:**
- ML-based reranking using bge-reranker-v2-m3
- Business rule scoring (active status, freshness, salary, skills match)
- Combined score calculation
- User profile matching (future)

**Key Methods:**

```python
# Compute ML rerank scores
scores = reranker.compute_rerank_scores(
    query="python developer",
    candidates=jobs_list
)

# Calculate business score
score = reranker.calculate_business_score(
    job_payload=job,
    user_profile=user_profile
)

# Full reranking pipeline
ranked = reranker.rerank_results(
    query="python developer",
    candidates=candidates,
    top_k=10,
    user_profile=user_profile,
    use_ml_reranker=True
)
```

### 3. BM25SearchEngine

Located in `app/hybrid_search.py`

**Responsibilities:**
- Build BM25 index from job corpus
- Tokenization of text
- Keyword-based scoring

**Workflow:**

```python
# Initialize BM25
engine = BM25SearchEngine()

# Build index from jobs
engine.build_index(job_payloads)

# Search
results = engine.search(
    query="machine learning",
    top_k=50
)
```

## Scoring Strategy

### Dense Vector Score
- Range: [0, 1]
- Higher = more semantically similar
- Based on cosine similarity in embedding space

**Example:**
```
Query: "junior data analyst"
Result 1: "Junior Data Analyst" → 0.92 (very similar)
Result 2: "Business Intelligence Intern" → 0.78 (similar)
Result 3: "Senior Python Developer" → 0.45 (less similar)
```

### BM25 Score
- Range: [0, ∞]
- Higher = more keyword matches
- Based on term frequency and document frequency

**Example:**
```
Query: "python sql power bi"
Result 1: "Python SQL Power BI" → 8.5 (exact match)
Result 2: "Python SQL" → 4.2 (partial match)
Result 3: "JavaScript Developer" → 0.0 (no match)
```

### Reciprocal Rank Fusion (RRF)

Merges dense and BM25 results using:

```
score(doc) = Σ 1 / (k + rank)
```

**Example with k=60:**
```
Dense Rank 1: 1/(60+1) = 0.0164
Dense Rank 2: 1/(60+2) = 0.0159
BM25 Rank 3: 1/(60+3) = 0.0152
...
Final = sum of all contributions
```

**Benefits:**
- Combines both signals
- Robust to individual ranking failures
- No need to normalize heterogeneous scores

### Business Score

Factors (max 5.0):
- Active status: +1.0
- Fresh posting: +0.5
- Has salary: +0.3
- Skill match per skill: +0.2 (capped at 1.0)
- Location match: +0.2
- Career level match: +0.2

**Example:**
```
Job 1: Active + Fresh + Salary + 3 skill matches = 3.8
Job 2: Active + Old + No Salary = 1.5
```

### Final Combined Score

Weight distribution:
- Reranker (ML): 55%
- Dense semantic: 25%
- Keyword (BM25): 10%
- Business rules: 10%

```python
final_score = (
    0.55 * normalized_rerank_score +
    0.25 * normalized_dense_score +
    0.10 * normalized_bm25_score +
    0.10 * normalized_business_score
)
```

## Implementation Details

### Reciprocal Rank Fusion

The `reciprocal_rank_fusion` method in `HybridSearchService`:

```python
def reciprocal_rank_fusion(
    self,
    result_lists: List[List[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Fuse multiple ranked lists using RRF.
    
    Args:
        result_lists: List of ranked lists
        k: RRF parameter (typically 60)
    
    Returns:
        Fused ranked list
    """
    fused_scores = {}
    
    for results in result_lists:
        for rank, (job_id, _) in enumerate(results, start=1):
            rrf_score = 1.0 / (k + rank)
            fused_scores[job_id] = fused_scores.get(job_id, 0) + rrf_score
    
    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked
```

### Filtering

Metadata filtering is done at Qdrant level before search:

```python
from qdrant_client.http import models

# Build filter conditions
must_conditions = [
    models.FieldCondition(
        key="status",
        match=models.MatchValue(value="active")
    ),
    models.FieldCondition(
        key="location",
        match=models.MatchValue(value="Hanoi")
    ),
    models.FieldCondition(
        key="job_type",
        match=models.MatchValue(value="Full-time")
    ),
]

query_filter = models.Filter(must=must_conditions)

results = qdrant_search(
    query_vector=embedding,
    query_filter=query_filter,
    limit=10
)
```

## API Usage

### Endpoint: POST /search

**Request:**
```json
{
  "query": "junior data analyst python sql",
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

**Response:**
```json
{
  "query": "junior data analyst python sql",
  "total": 5,
  "results": [
    {
      "job_id": 1,
      "title": "Junior Data Analyst",
      "company_name": "ABC Technology",
      "location": "Hanoi",
      "skills": ["Python", "SQL", "Power BI"],
      "dense_score": 0.92,
      "keyword_score": 8.5,
      "rerank_score": 0.87,
      "business_score": 2.8,
      "final_score": 0.89,
      "reason": "Matched 'junior data analyst python sql' with semantic score 0.92"
    }
  ],
  "search_time_ms": 245.3
}
```

## Performance Considerations

### Latency

- Dense search: ~50-100ms (Qdrant)
- BM25 search: ~10-20ms (in-memory)
- Reranking: ~50-100ms (ML model)
- **Total: ~150-250ms** for top 10 results

### Optimization Tips

1. **Batch Query Processing**
   ```python
   # Process multiple queries together
   queries = ["python", "java", "javascript"]
   embeddings = embedding_service.embed_texts(queries, batch_size=32)
   ```

2. **Cache Embeddings**
   ```python
   # Cache query embeddings for popular searches
   cache[query] = embedding
   ```

3. **Limit Reranking**
   ```python
   # Only rerank top 50 instead of top 100
   candidates = hybrid_search(query, limit=50)
   ranked = rerank(candidates, top_k=10)
   ```

4. **Tune BM25 Parameters**
   - Adjust k1, b parameters in BM25Okapi
   - Default k1=1.5, b=0.75 work well for most cases

## Testing

### Run Hybrid Search Test

```bash
python scripts/06_test_hybrid_search.py
```

This will:
1. Load sample jobs
2. Initialize BM25 index
3. Run 5 different test queries
4. Show results before/after reranking
5. Display scores at each stage

### Expected Output

```
Test: Looking for entry-level data analyst with specific skills
Query: 'junior data analyst python sql power bi'
--------

Hybrid search returned 5 candidates

Before reranking (top 5):
  1. Junior Data Analyst          @ ABC Technology      [Dense: 0.927, Keyword: 8.534]
  2. Business Intelligence Intern @ XYZ Corp            [Dense: 0.789, Keyword: 5.123]
  ...

After reranking (top 5):
  1. Junior Data Analyst          @ ABC Technology      [Final: 0.892, Rerank: 0.871, Business: 2.800]
  2. Business Intelligence Intern @ XYZ Corp            [Final: 0.756, Rerank: 0.701, Business: 1.500]
  ...
```

## Future Enhancements

1. **Sparse Vector Search**
   - Use SPLADE or similar for sparse representation
   - Enable Qdrant's native sparse vector support

2. **Learning to Rank**
   - Collect click/apply data
   - Train LTR model to optimize ranking

3. **User Profile Matching**
   - Embed user CV/profile
   - Personalize search results

4. **Cross-Encoder Reranking**
   - Use cross-encoders for better ranking
   - Slower but more accurate than bi-encoders

5. **Query Expansion**
   - Expand queries with synonyms
   - Use query rewriting for better recall

## References

- [RRF Paper](https://dl.acm.org/doi/10.1145/1571941.1571947)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Qdrant Hybrid Search](https://qdrant.tech/documentation/concepts/hybrid-search/)
