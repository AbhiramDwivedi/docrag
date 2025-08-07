# Repository Custom Instructions for GitHub Copilot

The goal of this repo is to build **DocQuest** - a local RAG pipeline that quests through document folders and lets users query documents in natural language. DocQuest embarks on intelligent journeys through any local document collection - personal files, team folders, cloud-synced directories, etc.

---

## General coding guidelines

1. **Target runtimes**  
   * Python ≥ 3.11
2. **Style & tooling**  
   * Use **type hints**, `dataclasses`/`pydantic`, and Google‑style docstrings.  
   * Format Python with **Black (PEP 8, 88 char line length)**.  
   * Prefer `pathlib.Path` over `os.path` for file paths.  
   * Windows is the primary dev environment—assume backslashes in examples.
3. **Testing**  
   * **All test files must be placed in the `backend/tests/` directory only**—never create test files in the root directory.
   * New Python modules should include **pytest** unit tests under `backend/tests/`.  
   * Use `pytest‑tmpdir` or `tmp_path` fixtures instead of real document data.
   * Test file naming: `test_<module_name>.py` (e.g., `backend/tests/test_extractor.py`)
   * Test resources should be placed in `backend/tests/resources/`
4. **Demo Files**  
   * **All demo files must be placed in the `backend/examples/` directory only**—never create demo files in the root directory.
   * Demo file naming: `demo_<feature_name>.py` (e.g., `backend/examples/demo_knowledge_graph.py`)
   * Demo files should include clear docstrings explaining their purpose and usage
5. **Dependencies**  
   * Keep requirements minimal—if the task can be done with the stdlib or an existing listed package, avoid new libs.  
   * For embeddings, default to **`sentence‑transformers/all‑MiniLM‑L6‑v2`** unless a task explicitly calls for OpenAI embeddings.
6. **Comments & TODOs**  
   * Leave `# TODO:` markers for incomplete sections; never commit unfinished code without a TODO.  
7. **Commit Messages**  
   * **Keep commit titles short and descriptive** (≤50 characters when possible).  
   * **Use the commit description** for detailed explanations, bullet points, and context.  
   * Follow conventional commits format: `type: short description` (e.g., `fix: resolve CI test failures`, `feat: add knowledge graph plugin`).  
   * Put implementation details, file changes, and rationale in the commit description, not the title.

---

## Python‑specific rules

* **Backend Structure**  
  * All imports must use the new backend structure: `backend.src.ingestion.*`, `backend.src.querying.*`, `backend.src.shared.*`, `backend.src.interface.cli.*`
  * Never use old import paths like `agent.*`, `ingest.*`, `cli.*`, or direct `backend.*` paths
  * Main code is located in `backend/src/` following standard Python project layout
* **Extractor (`backend/src/ingestion/extractors/`)**  
  * Use `PyMuPDF` (`fitz`) for PDF parsing.  
  * Use `python‑docx`, `python‑pptx`, and `pandas.read_excel` for DOCX, PPTX, XLSX respectively.  
  * Return `List[Tuple[str, str]]` where the first element is a **unit id** (page/slide/sheet) and the second is raw text.
* **Vector store (`backend/src/ingestion/storage/vector_store.py`)**  
  * Wrap **FAISS** + **SQLite**; ensure `upsert()` de‑dupes by `chunk_id`.  
  * Persist the FAISS index to `data/vector.index` after every write.
* **Watcher (`watcher/watch.py`)**  
  * Use `watchdog.observers.Observer` with a 2 s debounce to avoid duplicate events.  
  * Ignore temp files beginning `~$`.
* **CLI (`backend/src/interface/cli/ask.py`)**  
  * Answers must cite file path + unit id in plain text.  
  * If no context retrieved, return: *"No relevant information found."*.
* **Agent Framework (`backend/src/querying/agents/`)**  
  * Use plugin-based architecture for extensibility.  
  * Maintain backward compatibility with existing CLI interface.  
  * All plugins must implement the Plugin interface with `execute()`, `get_info()`, and `validate_params()` methods.
* **API (`backend/src/querying/api.py`)**  
  * FastAPI endpoints should integrate with the agent framework.  
  * Maintain RESTful design patterns and proper error handling.
* **Logging (`backend/src/interface/cli/ask.py`, `backend/src/querying/agents/`, etc.)**  
  * Use contextual loggers with descriptive names (e.g., `agent.classification`, `plugin.metadata`, `sql.query`, `llm.generation`).  
  * Support verbose logging levels: 0=minimal, 1=info, 2=debug, 3=trace.  
  * Use custom formatters with emoji mapping for different log contexts.  
  * Add special formatting for SQL queries (`sql_query`, `sql_params` attributes) and LLM interactions (`llm_prompt`, `llm_model` attributes).  
  * Ensure logging doesn't modify core business logic or function signatures.  
  * Test logging functionality with comprehensive test coverage.

---

## Documentation and Architecture

* **Documentation Review**  
  * Regularly review and update all documentation files including README.md, architecture diagrams, and API documentation.  
  * Ensure documentation accurately reflects current implementation and features.  
  * Update design diagrams when architectural changes are made.
  * Keep CI/CD workflows (`.github/workflows/ci.yml`) updated with current import paths and structure.
* **Architecture Documentation**  
  * Maintain clear architecture diagrams showing component relationships.  
  * Document data flow patterns and integration points.  
  * Keep plugin architecture documentation current with implemented plugins.
* **Architecture Diagram Updates**  
  * **MANDATORY**: Always update `docs/ARCHITECTURE.md` diagrams when making structural changes.
  * Update Mermaid diagrams to reflect new file paths, module names, or component relationships.
  * Verify all sequence diagrams use correct module paths (e.g., `backend.src.ingestion.*`, `backend.src.interface.*`).
  * Update component descriptions and file path references in diagram labels.
  * Test diagram rendering after updates to ensure syntax correctness.

---

## Paths to ignore

Copilot should **never edit or propose changes** to data artifacts:

```
data/**
.sync/**
```

---

## Interactions you can ask Copilot to perform

* "Implement the PDF extractor"
* "Write unit tests for chunker.py"
* "Review and update the architecture documentation"
* "Update the design diagrams to reflect current implementation"
* "Refactor ingestion pipeline to use async batching"
* "Add a new plugin to the agent framework"
* "Review and update the README.md documentation"
* "Create tests for the metadata commands plugin"
* "Review and update the CI workflow to match current structure"

---

Follow these instructions **unless a pull‑request discussion explicitly overrides them**.
