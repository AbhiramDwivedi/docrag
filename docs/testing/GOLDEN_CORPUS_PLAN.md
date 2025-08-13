# Golden Test Corpus Plan

Objective
- Establish a small, versioned, synthetic corpus to enable deterministic, CI-friendly retrieval testing.

Deliverables
- tests/fixtures/corpus_v1/ (10–20 small files across formats)
- tests/fixtures/queries_v1.jsonl (golden queries and expectations)
- scripts/build_test_artifacts.py (ingest corpus → artifacts)
- tests/fixtures/artifacts_v1/ (docmeta.db, vector.index, FTS5) – generated or cached
- config.test.yaml (points to artifacts; deterministic settings)
- CI wiring to build or restore artifacts, then run tests

Corpus Design (v1)
- Entities: 8–10 proper nouns (companies, products, acronyms) with controlled contexts
- Formats: txt, md, pdf (tiny), docx, pptx (1–2 slides), xlsx (sheet with a cell mentioning entity)
- Patterns covered:
  - Exact mentions, plural/singular, abbreviations
  - Noise around mentions (tables, bullets)
  - Near-duplicate mentions across files to test dedupe
  - Misspellings (1–2 edit distance) for negative/edge cases

Queries (queries_v1.jsonl)
- Structure per line:
  {"id": "q01", "query": "what is Acme?", "expects": {"doc_ids": ["acme_overview.txt"], "min_at_topk": 1, "k": 20}}
- Include types: entity definition, keyword find, semantic Q&A, abbreviations

Determinism
- Pin model: sentence-transformers/all-MiniLM-L6-v2 (initial)
- Force CPU; set seeds (numpy, torch, python)
- Normalize vectors; use cosine; stable sorting by (score desc, id asc)
- Ensure SQLite FTS5 present; fallback skip with marker if missing

Scripts
- scripts/build_test_artifacts.py
  - Inputs: tests/fixtures/corpus_v1/*
  - Outputs: tests/fixtures/artifacts_v1/{docmeta.db, vector.index, chunks_fts}
  - Steps: clean, ingest, embed, build FAISS and FTS5
  - Flags: --rebuild, --fail-on-drift (hash corpus -> ensure determinism)

Configs
- config.test.yaml
  - Paths: fixtures/artifacts_v1
  - Retrieval: k=100, use_cosine=true, mmr_lambda=0.5, hybrid.enabled=true
  - Embedding: model pinned, normalize=true

Tests
- Unit: normalization, MMR, FTS5, hybrid merge
- Integration: entity queries return expected docs; lexical path for "find files containing X"
- Perf: p95 latency bound (generous) and artifact size check

CI
- Cache artifacts by checksum of corpus + code version
- Job order: build artifacts → run tests → emit metrics JSON (recall@k, MRR)

Migration
- When changing model/corpus, create corpus_v2 and queries_v2; keep v1 for regression

Notes
- Keep files tiny and synthetic to avoid PII and large binary diffs
