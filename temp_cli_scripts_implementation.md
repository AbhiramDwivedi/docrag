# DocQuest CLI Command Scripts Implementation Guide

## Overview
This document provides specifications for GitHub Copilot to implement convenient command-line entry points for DocQuest using `[project.scripts]` in pyproject.toml. This will enable simple commands like `dq-ask "query"` instead of long path specifications.

## Current Analysis

### ✅ **Existing Entry Points Found**
- `backend/src/interface/cli/ask.py` - Has `main()` function ✅
- `backend/src/ingestion/pipeline.py` - Has `main()` function ✅  
- `backend/src/querying/api.py` - FastAPI app defined, no main() ❌

### ❌ **Missing Components**
- File watcher implementation (`backend/watcher/watch.py` is placeholder)
- API server entry point function
- Knowledge graph rebuild utility
- Incremental ingestion entry point

---

## Required Implementation

### 1. **pyproject.toml Scripts Section**

Add the following to `backend/pyproject.toml`:

```toml
[project.scripts]
# Query Commands
dq-ask = "src.interface.cli.ask:main"

# Ingestion Commands  
dq-ingest = "src.ingestion.pipeline:main"
dq-watch = "src.ingestion.watcher:main"
dq-incremental = "src.ingestion.incremental:main"

# Knowledge Graph Commands
dq-kg-build = "src.ingestion.storage.knowledge_graph:main"

# Development Commands (Optional - for API server)
dq-serve = "src.querying.api:main"
```

### 2. **Missing Entry Point Functions to Create**

#### A. **API Server Entry Point** 
File: `backend/src/querying/api.py`
```python
def main():
    """Start the FastAPI development server."""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
```

#### B. **File Watcher Implementation**
File: `backend/src/ingestion/watcher.py` (new file)
```python
def main():
    """Start filesystem watching for incremental ingestion."""
    # Implement watchdog observer
    # Call incremental ingestion on file changes
    # Include 2-second debounce
    # Ignore temp files starting with ~$
```

#### C. **Incremental Ingestion Entry Point**
File: `backend/src/ingestion/incremental.py` (new file)  
```python
def main():
    """Process individual files for incremental ingestion."""
    # Accept file path as argument
    # Process single file through pipeline
    # Update vector store and knowledge graph
```

#### D. **Knowledge Graph Builder Entry Point**
File: `backend/src/ingestion/storage/knowledge_graph.py`
```python
def main():
    """Rebuild knowledge graph from existing vector store."""
    # Use existing build_from_document_collection method
    # CLI argument parsing for options
    # Progress reporting
```

### 3. **Command Functionality Specifications**

#### **dq-ask** ✅ (Already implemented)
- **Purpose**: Query documents using Phase III agent
- **Usage**: `dq-ask "who is john doe"`
- **Features**: Full agent framework with KG + vector search

#### **dq-ingest** ✅ (Already implemented) 
- **Purpose**: Process document directory/file
- **Usage**: `dq-ingest path/to/documents`
- **Features**: Extract, chunk, embed, store in vector + KG

#### **dq-watch** ❌ (Needs implementation)
- **Purpose**: Monitor directory for changes
- **Usage**: `dq-watch path/to/documents`
- **Features**: Real-time file monitoring, incremental processing

#### **dq-incremental** ❌ (Needs implementation)
- **Purpose**: Process single file incrementally  
- **Usage**: `dq-incremental path/to/document.pdf`
- **Features**: Single file processing, efficient updates

#### **dq-kg-build** ❌ (Needs implementation)
- **Purpose**: Rebuild knowledge graph
- **Usage**: `dq-kg-build`
- **Features**: Rebuild KG from existing vector data

#### **dq-serve** ❌ (Needs implementation)
- **Purpose**: Start development API server
- **Usage**: `dq-serve`
- **Features**: FastAPI server on localhost:8000

### 4. **Implementation Requirements**

#### **Environment Handling**
- All scripts should activate virtual environment automatically
- Handle Windows encoding issues (UTF-8 output)
- Proper error handling and logging

#### **Argument Parsing**
- Use `argparse` for command-line options
- Support verbose/debug logging levels
- Help text for all commands

#### **Path Management**
- Use absolute imports via sys.path manipulation
- Handle Windows vs Unix path differences
- Support relative and absolute paths in arguments

#### **Integration Points**
- All commands should use shared configuration (`src.shared.config`)
- Consistent logging format across commands
- Shared vector store and knowledge graph instances

### 5. **Installation and Usage**

After implementation:

```bash
# Install in development mode
pip install -e .

# Commands become available system-wide
dq-ask "what are the quarterly results?"
dq-ingest ./documents/
dq-watch ./documents/ 
dq-incremental ./new-document.pdf
dq-kg-build
dq-serve
```

### 6. **Testing Requirements**

Create tests for:
- All entry point functions
- Command-line argument parsing
- Error handling scenarios
- Integration with existing components

Files to create:
- `backend/tests/test_cli_scripts.py`
- `backend/tests/test_watcher.py`
- `backend/tests/test_incremental.py`

### 7. **Documentation Updates Required**

After implementation, update:

#### **README.md**
- Add "Quick Start" section with new commands
- Update installation instructions
- Add command reference table

#### **Copilot Instructions**
- Add CLI script patterns to `.github/copilot-instructions.md`
- Include entry point function standards
- Add testing requirements for CLI scripts

#### **Architecture Documentation**
- Update `docs/ARCHITECTURE.md` with CLI workflow diagrams
- Document command integration points
- Add deployment/installation section

---

## Priority Implementation Order

1. **High Priority**: `dq-serve` entry point (easy FastAPI addition)
2. **High Priority**: `dq-kg-build` entry point (uses existing method)
3. **Medium Priority**: `dq-watch` implementation (new functionality)
4. **Medium Priority**: `dq-incremental` implementation (new functionality)
5. **Low Priority**: Documentation updates
6. **Low Priority**: Comprehensive testing

---

## Expected Outcome

Users will be able to run simple, memorable commands:
- `dq-ask "query"` instead of `python backend/src/interface/cli/ask.py "query"`
- `dq-ingest docs/` instead of `python backend/src/ingestion/pipeline.py docs/`
- `dq-watch docs/` for real-time monitoring
- `dq-serve` for quick API development

This significantly improves the developer and user experience while maintaining all existing functionality.
