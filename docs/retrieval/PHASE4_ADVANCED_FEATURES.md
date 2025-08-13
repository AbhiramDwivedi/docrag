# Phase 4: Advanced Features (Optional)

Goal: Improve precision and entity-awareness.

Scope
- Cross-encoder reranking
- Query expansion for entities
- Entity-aware indexing and boosts

Key Changes
- Cross-encoder rerank top-100 → top-20 (configurable)
- Expand entity queries and merge results
- NER at ingestion; entity→document mapping and boosting

Files to Add/Modify
- `querying/agents/plugins/semantic_search.py`
- Add reranker utility and configuration
- Optional: spaCy pipeline for NER during ingestion

Acceptance Criteria
- Higher precision in final answers
- Better coverage for entity variants
- Config flags to toggle features without regressions
