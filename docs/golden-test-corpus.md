# Golden Test Corpus and Deterministic Retrieval Testing

This document describes the golden test corpus and deterministic test harness for DocRAG retrieval testing.

## Overview

The golden test corpus provides a small, versioned, reproducible test dataset that enables consistent retrieval behavior testing across CI and local environments. It consists of synthetic documents designed to test specific retrieval scenarios.

## Directory Structure

```
tests/
  fixtures/
    corpus_v1/
      text/                 # Text and markdown files
        acme_overview.txt
        bolt_spec.md
        contoso_partnership.txt
        globex_financial.md
        initech_roadmap.txt
        reltora_integration.md
      office/               # Generated office files
        contoso_brief.docx
        metrics.xlsx
        onepager.pdf
        roadmap.pptx
    queries_v1.jsonl       # Golden queries with expectations
    artifacts_v1/           # Generated test artifacts (not committed)
      docmeta.db           # SQLite database with metadata
      vector.index         # FAISS vector index
      metadata.json        # Build metadata and checksums
scripts/
  build_test_artifacts.py   # Deterministic artifact builder
config.test.yaml            # Test configuration
```

## Test Corpus Design

The corpus v1 contains:

- **Entities**: 8-10 proper nouns (companies, products, acronyms) with clear contexts
- **Formats**: txt, md (committed) + docx, pptx, pdf, xlsx (generated)
- **Size**: Small files (~1-2KB each) for fast testing
- **Patterns**: Exact mentions, abbreviations, pluralization, cross-references

### Example Entities

- **Acme Corporation**: Data platform company
- **BOLT Protocol**: High-performance streaming protocol
- **Contoso**: Enterprise consulting firm
- **Globex Financial**: Technology investment bank
- **Initech**: Workflow automation provider
- **Reltora Gateway**: API gateway solution

## Golden Queries

The `queries_v1.jsonl` file contains test queries with expected results:

```json
{"id":"q01","type":"entity","query":"what is Acme?","expects":{"doc_ids":["acme_overview.txt"],"k":20,"min_at_topk":1}}
{"id":"q02","type":"keyword","query":"find files containing Globex","expects":{"doc_ids":["globex_financial.md"],"k":20,"min_at_topk":1}}
```

Each query specifies:
- `id`: Unique query identifier
- `type`: Query type (entity, keyword, abbr, etc.)
- `query`: Natural language query text
- `expects`: Expected results with document IDs and thresholds

## Building Test Artifacts

### Prerequisites

Install additional dependencies for office file generation:

```bash
pip install reportlab openpyxl
```

### Build Script Usage

```bash
# Build artifacts (first time or after corpus changes)
python scripts/build_test_artifacts.py --rebuild --verbose

# Check for corpus drift
python scripts/build_test_artifacts.py --fail-on-drift

# CI mode with stricter validation
python scripts/build_test_artifacts.py --rebuild --ci
```

### Build Script Features

- **Deterministic**: Sets seeds, forces CPU, normalizes vectors
- **Checksum validation**: Detects corpus drift
- **Office file generation**: Creates docx, pptx, pdf, xlsx programmatically
- **FTS5 support**: Builds lexical search index if available
- **Metadata tracking**: Records model, versions, build info

## Running Tests

### Unit Tests

```bash
# Test embedding normalization
python -m pytest tests/test_embeddings_norm.py -v

# Test MMR selection logic
python -m pytest tests/test_mmr.py -v

# Test FTS5 functionality (skipped if not available)
python -m pytest tests/test_fts5.py -v

# Test hybrid search merging
python -m pytest tests/test_hybrid_merge.py -v
```

### Integration Tests

```bash
# Test golden retrieval scenarios
python -m pytest tests/test_golden_retrieval.py -v

# Performance smoke tests
python -m pytest tests/test_perf_smoke.py -v
```

### All Golden Tests

```bash
# Set test configuration
export DOC_RAG_CONFIG=config.test.yaml

# Run all golden tests
python -m pytest tests/test_embeddings_norm.py tests/test_mmr.py tests/test_fts5.py tests/test_hybrid_merge.py tests/test_golden_retrieval.py tests/test_perf_smoke.py -v
```

## Test Configuration

The `config.test.yaml` file contains deterministic test settings:

```yaml
# Retrieval settings
retrieval:
  k: 100
  mmr_lambda: 0.5
  similarity_threshold: 0.3

# Hybrid search settings  
hybrid:
  enabled: true
  dense_weight: 0.6
  lexical_weight: 0.4

# Embedding settings for deterministic behavior
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  normalize: true
  use_cosine: true
  device: "cpu"

# Deterministic settings
deterministic:
  seed: 42
  force_cpu: true
  omp_num_threads: 1
```

## CI Integration

The GitHub Actions workflow includes:

1. **Artifact caching**: Based on corpus and code checksums
2. **Automatic builds**: Regenerates artifacts when needed
3. **Test execution**: Runs all golden tests
4. **FTS5 fallback**: Skips lexical tests if SQLite lacks FTS5

### Cache Key

Artifacts are cached using:
```
artifacts-${{ hashFiles('tests/fixtures/corpus_v1/**', 'scripts/build_test_artifacts.py', 'config.test.yaml', 'backend/**/*.py') }}
```

## Deterministic Requirements

To ensure reproducible results:

- **Seeds**: Python (42), NumPy (42), Torch (42)
- **CPU only**: Forces CPU execution, disables CUDA
- **Thread control**: Sets `OMP_NUM_THREADS=1`
- **Vector normalization**: L2 normalization for cosine similarity
- **Stable sorting**: Tie-breaking by (score desc, id asc)

## Troubleshooting

### FTS5 Not Available

If SQLite was not compiled with FTS5 support:
```bash
# Install SQLite with FTS5
pip install pysqlite3-binary
```

Tests will automatically skip FTS5-dependent functionality if not available.

### Checksum Mismatch

If artifacts are out of date:
```bash
# Force rebuild
python scripts/build_test_artifacts.py --rebuild
```

### Performance Issues

Check artifact sizes and performance bounds:
```bash
# Check artifact metadata
cat tests/fixtures/artifacts_v1/metadata.json

# Run performance tests
python -m pytest tests/test_perf_smoke.py::TestPerformanceSmoke::test_artifact_size_bounds -v
```

## Local Testing Instructions

### Prerequisites

Ensure you have the required dependencies installed:

```bash
# Core dependencies
pip install -e backend[test]
```

This installs all required test dependencies including pytest, numpy, faiss-cpu, psutil, and office file generation libraries.

### Building Test Artifacts

Test artifacts are automatically built when running tests, but you can build them manually:

```bash
# Build test artifacts from corpus
python scripts/build_test_artifacts.py

# Force rebuild (ignores existing artifacts)
python scripts/build_test_artifacts.py --force

# Build with verbose logging
python scripts/build_test_artifacts.py --verbose
```

### Running Tests Locally

```bash
# Run all golden corpus tests
python -m pytest tests/test_golden_retrieval.py -v

# Run specific test categories
python -m pytest tests/test_embeddings_norm.py -v     # Embedding normalization
python -m pytest tests/test_mmr.py -v                 # MMR selection  
python -m pytest tests/test_fts5.py -v               # Full-text search
python -m pytest tests/test_hybrid_merge.py -v       # Hybrid scoring
python -m pytest tests/test_perf_smoke.py -v         # Performance tests

# Run with FTS5 tests skipped (if FTS5 unavailable)
python -m pytest tests/ -k "not fts5" -v
```

### Environment Consistency

For deterministic results across different environments:

1. **CPU-Only Mode**: Set `CUDA_VISIBLE_DEVICES=""` to force CPU execution
2. **Fixed Seeds**: All randomization uses seed 42 for reproducibility  
3. **Single Threading**: PyTorch configured for single-threaded execution
4. **Normalized Embeddings**: All vectors normalized for cosine similarity

### Debugging Test Failures

#### Common Issues and Solutions

1. **Missing Artifacts Error**
   ```bash
   python scripts/build_test_artifacts.py --force
   ```

2. **FTS5 Unavailable Warning**
   - Install SQLite with FTS5 support, or
   - Skip FTS5 tests: `pytest -k "not fts5"`

3. **Non-Deterministic Scores**
   - Verify CPU-only mode: `echo $CUDA_VISIBLE_DEVICES` should be empty
   - Check PyTorch threading: should show single thread in test logs

4. **Performance Test Failures**
   - Adjust memory/timing thresholds in `test_perf_smoke.py` for your system
   - Check system resources during test execution

#### Verification Commands

```bash
# Check artifact integrity
python -c "
import faiss, sqlite3
from pathlib import Path

artifacts_path = Path('tests/fixtures/artifacts_v1')
index = faiss.read_index(str(artifacts_path / 'vector.index'))
print(f'Vector index: {index.ntotal} vectors, {index.d} dimensions')

conn = sqlite3.connect(str(artifacts_path / 'docmeta.db'))
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM chunks')
print(f'Database chunks: {cursor.fetchone()[0]}')
conn.close()
"

# Test FTS5 availability
python -c "
import sqlite3
try:
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE VIRTUAL TABLE test USING fts5(content)')
    print('FTS5 available')
except:
    print('FTS5 not available - FTS5 tests will be skipped')
"
```

### Expected Performance Benchmarks

On modern development hardware, expect:

- **Artifact Build Time**: < 30 seconds for full rebuild
- **Test Execution Time**: < 60 seconds for full test suite
- **Vector Index Size**: < 100KB (17 chunks × 384 dimensions)
- **Database Size**: < 500KB including FTS5 index
- **Memory Usage**: < 100MB peak during tests
- **Query Latency**: < 100ms per retrieval query

If your results significantly exceed these benchmarks, check system resources and configuration.

## Extending the Corpus

### Adding New Documents

1. Add text/markdown files to `tests/fixtures/corpus_v1/text/`
2. Update golden queries in `tests/fixtures/queries_v1.jsonl`
3. Rebuild artifacts: `python scripts/build_test_artifacts.py --rebuild`
4. Update tests if needed

### Adding New Query Types

1. Design queries that test specific retrieval scenarios
2. Add to `queries_v1.jsonl` with appropriate expectations
3. Consider adding specific test cases for edge cases

### Versioning

Future corpus versions should:
- Use new directory: `corpus_v2/`, `queries_v2.jsonl`
- Update build script to support multiple versions
- Maintain backward compatibility for regression testing