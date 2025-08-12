# GitHub Copilot Project Instructions

## Project Purpose
- This is a Python 3.11 project for document retrieval and question answering (RAG) with a plugin-based architecture. The main codebase is under the `backend/` directory.
- Copilot should generate code that is correct, testable, and maintainable, following project conventions and structure.

## Project Structure
- Root: `backend/` (all core code lives here)
  - `backend/ingestion/`: document ingestion, chunking, embedding
  - `backend/ingestion/storage/`: vector store (FAISS), SQLite, FTS5
  - `backend/querying/agents/plugins/`: retrieval plugins (semantic, lexical, etc.)
  - `backend/shared/`: config, logging, utilities
  - `cli/` and `interface/cli/`: command-line interface
- Tests: `tests/` (unit, integration, perf)
- Test fixtures: `tests/fixtures/`
- Config: `shared/config.py` and `shared/config.yaml.template`
- Data: external, pointed to by config (e.g., `docquest-data/`)

## Coding Guidelines
- Use Python 3.11+ and type hints for all new/changed functions.
- Use project logging (`shared.logging_config`) and structured logs for new features.
- All configuration must be read via `shared.config.Config` or config files; never hardcode paths or credentials.
- Storage/data paths must be configurable and default outside the repo.
- When adding retrieval features:
  - Use cosine similarity with normalized vectors by default.
  - Implement hybrid/lexical retrieval as a separate plugin (FTS5), not in metadata plugin.
  - Implement MMR with configurable `mmr_lambda` and deterministic tie-breaking.
  - Add optional debug logging, controlled by config.

## Testing & CI
- All code changes must include or update tests (unit, integration, perf as appropriate).
- Tests must be deterministic: set seeds, force CPU, normalize vectors, stable sorting.
- No network access in tests.
- Validate and update CI (GitHub Actions) with every change: ensure all tests pass in CI, not just locally.
- If adding new config, update `config.yaml.template` and document usage.

## Security & Privacy
- Never commit secrets or credentials. Use environment variables for sensitive data.
- Do not send document contents to third-party services unless behind a feature flag and clearly documented.

## PR Expectations
- PRs must explain motivation, approach, and trade-offs.
- List any config changes, migrations, and rollback steps.
- Include before/after metrics (e.g., recall@k, latency) for retrieval changes when possible.

## Out of Scope for Copilot
- Major refactors of agent architecture without a design issue/plan.
- Adding heavy dependencies or external services without prior discussion.
