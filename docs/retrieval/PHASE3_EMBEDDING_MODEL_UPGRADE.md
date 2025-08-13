# Phase 3: Embedding Model Upgrade (High Impact, Medium Risk)

Goal: Improve dense retrieval quality with a better embedding model.

Scope
- Evaluate e5-base-v2, bge-small-en-v1.5, gte-base
- Model-specific formatting (prefixes, normalization, pooling)
- Re-embedding and index migration script

Key Changes
- Prefix handling for e5 ("query:", "passage:")
- Configurable model in `shared/config.py`; version tracking
- `scripts/migrate_embeddings.py` with progress/resume

Files to Add/Modify
- `ingestion/processors/embedder.py`
- `shared/config.py`
- Add `scripts/migrate_embeddings.py`

Acceptance Criteria
- ≥20% recall@10 improvement on eval set
- Proper noun queries succeed with dense-only
- Migration without data loss and within latency budgets
