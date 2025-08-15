# Complex Query Analysis: Multi-Constraint Metadata Queries

## Issue Summary

Problem: Multi-constraint listing queries (e.g., latest N items of specific file type, optionally about a target term) return incorrect results:
- Returns default count instead of requested count.
- Includes file types outside the requested type synonyms.
- Does not apply content filtering when a target term is present.

## Root Cause Analysis

Current behavior (simplified):
```
Query: "List N latest <type-synonym> that mention <target-term>"
-> Intent Classification: METADATA_QUERY (correct for listing)
-> Orchestrator Plan: Single step (return_metadata)
-> Discovery Agent: get_latest_files() (no parameters)
-> Metadata Plugin: Defaults (count=10, no file_type filter)
-> Result: Default-sized mixed file list without content filtering
```

Expected behavior (simplified):
```
Query: "List N latest <type-synonym> that mention <target-term>"
-> Intent Classification: METADATA_QUERY (listing intent)
-> Orchestrator Plan: Two steps (metadata narrowing, then content filtering when target-term exists)
   Step 1: Metadata filtering (count=N, file_type mapping from synonym, recency)
   Step 2: Content filtering (semantic search restricted to Step 1 result set)
-> Result: Exactly N items of the requested type, matching the target term, most recent first
```

## Single Design: Constraint-aware orchestration with two-step narrowing

Implement a ConstraintExtractor + Orchestrator pattern that always respects explicit user constraints and adds a second semantic step only when a content term is present.

### 1) ConstraintExtractor (new utility)

Purpose: Convert natural-language constraints into a typed structure the orchestrator and plugins can use.

Contract:
- Input: `query: str`
- Output (dataclass `QueryConstraints`):
  - `count: Optional[int]` — requested number of items if present, else None (defaults applied later).
  - `file_types: list[str]` — canonical extensions, e.g., ["PPTX","PPT"] for “decks”, ["PDF"] for “pdfs”, ["DOCX","DOC"], ["XLSX","XLS"].
  - `recency: bool` — true if words like “latest”, “newest”, “recent” are present.
  - `content_terms: list[str]` — remaining meaningful terms after removing constraint tokens and stopwords.
  - `sort_by: Literal["modified_time"]` (default).
  - `sort_order: Literal["desc"]` (default).

Behavior:
- Extract numbers from digits and number words (one..twenty); bound to a safe range (e.g., 1–100).
- Map type synonyms deterministically (decks→PPTX/PPT, spreadsheets→XLSX/XLS, docs→DOCX/DOC, pdfs→PDF).
- Normalize to uppercase extensions used by the metadata schema.
- Remove constraint keywords and standard stopwords to form `content_terms` (stable order).

Placement:
- File: `backend/shared/constraints.py` (new) or `backend/shared/utils.py` if preferred.

### 2) Orchestrator: constraint-aware plan construction

Plan logic for METADATA_QUERY:
- Call `ConstraintExtractor.extract(query)`.
- If `content_terms` is empty:
  - Single metadata step: call `get_latest_files` with parameters:
    - `count`: `constraints.count or default_latest_count` (from config).
    - `file_type`: if `constraints.file_types` has one value, pass it; if multiple, pass list (see plugin note below).
  - Sorting: by modified time desc, with stable tie-breaker.
- If `content_terms` is non-empty:
  - Two-step plan:
    1) Metadata step: call `get_latest_files` with `file_type` (if any) and widened count to ensure sufficient pool for content filtering. Widen factor (e.g., 3) is configurable and bounded.
    2) Content step: semantic search limited to the metadata result set (`target_docs`), using `content_terms` joined as a query. Clip to requested count with stable tie-breaking.

Notes:
- Keep overall intent as METADATA_QUERY for such listing queries; the presence of a target content term only affects the plan shape.
- Deterministic ordering is required at each step.

### 3) Metadata plugin usage and determinism

Metadata step parameters:
- `operation`: `get_latest_files`.
- `count`: integer (requested or widened when content step follows).
- `file_type`: uppercase extension string or list of strings.

Sorting:
- Ensure SQL uses `ORDER BY modified_time DESC, file_name ASC` (stable tie-breaker).

Multiple file types:
- If enhanced schema supports `IN (...)`, use it; otherwise fetch and filter client-side deterministically.

### 4) Semantic filtering restricted to metadata results

Search parameters (for semantic plugin):
- `question`: join of `content_terms` or the original residual content phrase.
- `target_docs`: list of file paths from the metadata step (restricts search domain).
- `k`: min(configured max, len(target_docs)).
- `max_documents`: requested count.

Sorting and clipping:
- Sort by similarity desc, then by path asc (stable tie-breaker), and return exactly the requested count when available.

### 5) Configuration and logging

Configuration (via `shared.config.Config` and template):
- `default_latest_count`: default N when none provided (e.g., 10).
- `content_filter_widen_factor`: widening multiplier for Step 1 when Step 2 exists (e.g., 3; clip to reasonable max such as 50).
- `semantic_max_k`: upper bound for k in semantic step (e.g., 50).

Logging (structured, debug level):
- Extracted constraints.
- Step parameters (sanitized) and sizes in/out.
- Final count and applied ordering.

## Actionable Implementation Plan

1) Add constraint extraction utility
- File: `backend/shared/constraints.py` (new).
- Implement `@dataclass QueryConstraints` and `ConstraintExtractor.extract(query: str) -> QueryConstraints`.
- Include: number word map, type synonym map, stopwords, normalization, safe bounds.
- Unit tests: `tests/test_constraints.py` covering numbers, types, recency detection, content terms.

2) Wire constraints into orchestrator
- File: `backend/src/querying/agents/agentic/orchestrator_agent.py`.
- In the METADATA_QUERY path, call `ConstraintExtractor.extract(query)`.
- Build either a single-step metadata plan (no `content_terms`) or a two-step plan (with `content_terms`).
- Pass `count` and `file_type` into the metadata step; pass `target_docs` and `max_documents` into the semantic step.

3) Ensure discovery agent forwards parameters
- File: `backend/src/querying/agents/agentic/discovery_agent.py`.
- When executing metadata commands for latest files, include `count` and `file_type` from the plan step parameters.

4) Stabilize metadata plugin ordering and multi-type support
- File: `backend/src/querying/agents/plugins/metadata_commands.py`.
- Ensure `ORDER BY modified_time DESC, file_name ASC`.
- If multiple `file_type` values are provided:
  - Enhanced schema: use an `IN` clause with uppercase types.
  - Legacy schema: perform deterministic client-side filtering with stable ordering.

5) Restrict semantic search to target docs and clip deterministically
- File: semantic search caller (e.g., `backend/src/querying/agents/agentic/analysis_agent.py`).
- Accept `target_docs`, set `k` conservatively, clip to requested count using stable tie-breaking (similarity desc, path asc).

6) Deterministic, offline tests
- Unit tests for `ConstraintExtractor`.
- Orchestrator plan tests for:
  - Only metadata constraints (count, type, recency) -> single-step plan, exact count, correct ordering, correct type filtering.
  - Metadata + content constraints -> two-step plan with widened metadata pool and restricted semantic filtering, exact final count.
- Integration test with a small deterministic dataset (mix of extensions and known timestamps). Mock semantic ranking to avoid network/embedding variability.

## Success Criteria

- Count constraint respected: returns exactly the requested number when available.
- File type constraint respected: only requested type(s) appear in results.
- Content constraint respected when present: semantic filtering applied within metadata-filtered set.
- Deterministic ordering: metadata by modified time desc, tie-broken by file name; semantic by similarity desc, tie-broken by path.
- Backward compatible: simple listing queries behave as before with defaults.

## Related Components

- `backend/shared/constraints.py` — constraint extraction (new).
- `backend/src/querying/agents/agentic/orchestrator_agent.py` — plan construction.
- `backend/src/querying/agents/agentic/discovery_agent.py` — metadata execution with parameters.
- `backend/src/querying/agents/plugins/metadata_commands.py` — latest files query parameters and ordering.
- `backend/src/querying/agents/agentic/analysis_agent.py` — semantic filtering restricted to target docs.

## Notes

This design keeps concerns separated and testable: parsing, planning, metadata filtering, and semantic filtering are modular. It respects project constraints (no network in tests, deterministic outputs, config-driven behavior) and avoids personal or environment-specific examples.
