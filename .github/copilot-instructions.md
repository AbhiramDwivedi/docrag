# Repository Custom Instructions for GitHub Copilot

The goal of this repo is to build a small, local **RAG pipeline** that syncs a SharePoint folder via OneDrive (Files‑On‑Demand) and lets me query the documents in natural language.

---

## General coding guidelines

1. **Target runtimes**  
   * Python ≥ 3.11  
   * Node (LTS) for the optional Next.js UI
2. **Style & tooling**  
   * Use **type hints**, `dataclasses`/`pydantic`, and Google‑style docstrings.  
   * Format Python with **Black (PEP 8, 88 char line length)**; format JS/TS with **Prettier**.  
   * Prefer `pathlib.Path` over `os.path` for file paths.  
   * Windows is the primary dev environment—assume backslashes in examples.
3. **Testing**  
   * New Python modules should include **pytest** unit tests under `tests/`.  
   * Use `pytest‑tmpdir` or `tmp_path` fixtures instead of real SharePoint data.
4. **Dependencies**  
   * Keep requirements minimal—if the task can be done with the stdlib or an existing listed package, avoid new libs.  
   * For embeddings, default to **`sentence‑transformers/all‑MiniLM‑L6‑v2`** unless a task explicitly calls for OpenAI embeddings.
5. **Comments & TODOs**  
   * Leave `# TODO:` markers for incomplete sections; never commit unfinished code without a TODO.  

---

## Python‑specific rules

* **Extractor (`ingest/extractor.py`)**  
  * Use `PyMuPDF` (`fitz`) for PDF parsing.  
  * Use `python‑docx`, `python‑pptx`, and `pandas.read_excel` for DOCX, PPTX, XLSX respectively.  
  * Return `List[Tuple[str, str]]` where the first element is a **unit id** (page/slide/sheet) and the second is raw text.
* **Vector store (`ingest/vector_store.py`)**  
  * Wrap **FAISS** + **SQLite**; ensure `upsert()` de‑dupes by `chunk_id`.  
  * Persist the FAISS index to `data/vector.index` after every write.
* **Watcher (`watcher/watch.py`)**  
  * Use `watchdog.observers.Observer` with a 2 s debounce to avoid duplicate events.  
  * Ignore temp files beginning `~$`.
* **CLI (`cli/ask.py`)**  
  * Answers must cite file path + unit id in plain text.  
  * If no context retrieved, return: *“No relevant information found.”*.

---

## JavaScript / Next.js UI rules

* Scaffold with **`create‑next‑app`** (App Router).  
* Use **shadcn/ui** components and **TailwindCSS** (already configured in the template).  
* Fetch answers from FastAPI with `SSE` and stream them to the chat panel.

---

## Paths to ignore

Copilot should **never edit or propose changes** to data artifacts:

```
data/**
.sync/**
```

---

## Interactions you can ask Copilot to perform

* “Implement the PDF extractor”
* “Write unit tests for chunker.py”
* “Generate a React chat component that streams SSE”
* “Refactor ingest.py to use async batching”

---

Follow these instructions **unless a pull‑request discussion explicitly overrides them**.
