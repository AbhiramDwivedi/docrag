# Phase 1: Retrieval Robustness (Low Risk, High Impact)

Goal: Improve dense retrieval quality without schema changes or re-embedding.

Scope
- Cosine similarity and vector normalization
- Increase candidate depth and add MMR
- Discovery query heuristics for proper nouns
- Observability and structured logging

Key Changes
- Normalize embeddings and use cosine similarity (or IP index with normalized vectors)
- Increase k (100-200) and implement MMR (λ configurable)
- Heuristics for short/proper-noun queries and metadata boosting
- Add detailed logging across retrieval stages

Files to Modify
- `ingestion/processors/embedder.py`
- `ingestion/storage/vector_store.py`
- `querying/agents/plugins/semantic_search.py`
- `shared/config.py`

Acceptance Criteria
- "what is [EntityName]?" returns target documents with citations
- No "No documents found" when relevant chunks exist
- Logs show clear pipeline decisions
- No regressions on current tests; top-20 includes ≥1 relevant chunk for proper nouns
