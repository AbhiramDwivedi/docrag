# Phase 2: Hybrid Lexical Search (Medium Risk, High Impact)

Goal: Add keyword/exact-text fallback when semantic search fails.

Scope
- New lexical search plugin using SQLite FTS5 (or Whoosh)
- Hybrid orchestration to merge dense + lexical
- Query intent classification to choose strategy
- Remove broken content search from metadata plugin

Key Changes
- `chunks_fts` FTS5 table; BM25 scoring and exact/prefix match
- Merge candidates with weighted scoring; deduplicate by chunk id
- Route short proper nouns to hybrid; complex to semantic-primary
- Remove `_find_files_by_content` from metadata plugin

Files to Add/Modify
- Add `querying/agents/plugins/lexical_search.py`
- Update `querying/agents/plugins/semantic_search.py`
- Update `querying/agents/plugins/metadata_commands.py`
- Update `shared/config.yaml` (enable_hybrid, weights)

Acceptance Criteria
- "find files containing [keyword]" uses lexical search
- Hybrid improves recall@20 for proper nouns by ≥50% vs dense-only
- Unit tests validate FTS5 build and query paths
- No content search remains in metadata plugin
