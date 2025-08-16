# Multi-Constraint Query Processing Requirements

## Problem Statement

**Current Issue**: Queries with multiple constraints return incorrect results:
- Query: `"List 5 latest decks that discuss project roadmaps"`
- Expected: 5 PowerPoint files, filtered by content about project roadmaps, sorted by modification date
- Actual: 10 mixed file types, no content filtering, default metadata query behavior

**Root Cause**: The system treats this as a simple metadata query, ignoring count, file type, and content constraints.

## Requirements

### R1: Parse Query Constraints
The system MUST extract and honor these constraints from natural language queries:
- **Count constraint**: "5 latest", "ten recent", "first 3" → extract number
- **File type constraint**: "decks" → PPTX/PPT, "spreadsheets" → XLSX/XLS, "docs" → DOCX/DOC  
- **Time constraint**: "latest", "recent", "newest" → sort by modification time DESC
- **Content constraint**: "about X", "containing Y", "discussing Z" → semantic search filter

### R2: Smart Orchestration Logic
- **Metadata-only queries**: If no content constraint → single metadata step
- **Multi-constraint queries**: If content constraint exists → two-step plan:
  1. Metadata filtering (count × multiplier, file type, recency)
  2. Content filtering (semantic search on step 1 results, clip to requested count)

### R3: Deterministic Results
- Metadata sorting: `ORDER BY modified_time DESC, file_name ASC` (stable tie-breaker)
- Semantic sorting: similarity DESC, then file_path ASC (stable tie-breaker)
- Results MUST be reproducible across runs

### R4: Parameter Passing
Discovery agent MUST forward extracted constraints to metadata plugin:
- `count`: integer (extracted number or config default)
- `file_type`: uppercase extension(s) ["PPTX"] or ["XLSX", "XLS"] 
- For content queries: widen count by configured multiplier (e.g., 3x) for step 1

## High-Level Design

### Component Architecture

```
Query → ConstraintExtractor → OrchestratorAgent → DiscoveryAgent → MetadataPlugin
                           ↓                  ↓
                           ExecutionPlan → AnalysisAgent → SemanticPlugin
```

### Key Components

#### 1. ConstraintExtractor (`backend/shared/constraints.py`)
**Purpose**: Parse natural language constraints into structured parameters

**Design**: Create a data structure to represent query constraints and a function to extract them from natural language. The constraint structure should include:
- Numeric constraints (count)
- File type mappings (decks → presentation files, docs → document files, etc.)
- Content filtering requirements (semantic search terms)
- Temporal sorting preferences (latest, recent, newest)

**Implementation Approach**:
- Use regular expressions and word mapping for number extraction
- Maintain a dictionary of file type synonyms to canonical extensions
- Detect content filtering by removing constraint keywords and checking for remaining meaningful terms
- Apply bounds checking for extracted numeric values

#### 2. Enhanced Orchestrator (`backend/src/querying/agents/agentic/orchestrator_agent.py`)
**Purpose**: Create constraint-aware execution plans

**Design Changes**:
- Modify the metadata query planning logic to parse constraints first
- Implement conditional planning: single-step for metadata-only queries, two-step for content-filtered queries
- Pass extracted constraints as parameters in execution steps
- Maintain backwards compatibility with existing simple queries

#### 3. Enhanced Discovery Agent (`backend/src/querying/agents/agentic/discovery_agent.py`)
**Purpose**: Forward constraints to metadata plugin

**Design Changes**:
- Modify metadata command execution to read constraint parameters from execution steps
- Build metadata plugin parameters including count and file type constraints
- Handle cases where constraints are missing or invalid with appropriate defaults

#### 4. Metadata Plugin (already supports parameters)
**Current State**: ✅ Already implemented in `_get_latest_files()`
- Supports `count` parameter
- Supports `file_type` parameter  
- Has deterministic ordering logic

**Required**: No changes needed, just parameter forwarding from Discovery Agent

### Implementation Steps

#### Step 1: Constraint Extraction
Create the constraint parsing utility that can extract structured parameters from natural language queries. This includes number parsing, file type mapping, and content term detection.

#### Step 2: Orchestrator Integration  
Integrate constraint extraction into the orchestrator's planning logic. Modify metadata query planning to parse constraints and create appropriate execution plans based on whether content filtering is needed.

#### Step 3: Discovery Agent Parameter Forwarding
Update the discovery agent to read constraints from execution step parameters and forward them to the metadata plugin. This ensures the metadata plugin receives the correct count and file type filters.

#### Step 4: Two-Step Content Filtering
Implement the logic for two-step queries where metadata filtering is followed by content filtering. This includes widening the initial metadata query and then applying semantic search with proper result clipping.

#### Step 5: Testing & Validation
Create comprehensive tests to verify the constraint parsing, plan creation, and result filtering work correctly across different query types.

### Success Criteria

| Test Query | Expected Behavior | Validation |
|------------|------------------|------------|
| `"List 5 latest decks"` | 5 PPTX files, no content filter | ✅ Count + type constraints |
| `"Show 3 recent spreadsheets about budget"` | 3 XLSX files about budget | ✅ Count + type + content |
| `"Latest 10 files"` | 10 files, any type, by mod time | ✅ Count only |
| `"Decks about strategy"` | All PPTX files about strategy | ✅ Type + content (no count) |

### Configuration

Add constraint-related configuration options:
- Default count when no count specified
- Content filtering widen factor (multiply count for step 1 when content filtering)  
- Maximum count limit for safety
- File type synonym mappings

### Error Handling

- Invalid counts (negative, >max) → use default
- Unknown file types → ignore type filter
- No content terms after parsing → single metadata step
- Empty metadata results → return appropriate message

## Implementation Notes

### File Locations
- **New file**: `backend/shared/constraints.py` (constraint extraction)
- **New file**: `tests/test_constraints.py` (unit tests)
- **Modify**: `backend/src/querying/agents/agentic/orchestrator_agent.py` (planning)
- **Modify**: `backend/src/querying/agents/agentic/discovery_agent.py` (parameter forwarding)
- **Modify**: `backend/src/querying/agents/agentic/analysis_agent.py` (target_docs support)

### Testing Strategy
- Unit tests for constraint extraction with edge cases
- Integration tests with small deterministic dataset
- Mock semantic search to avoid network calls
- Verify exact count, type filtering, and content matching

### Rollback Plan
All changes are additive and backwards-compatible:
- Simple queries continue working as before
- New constraint extraction only activates for multi-constraint queries
- Metadata plugin already supports optional parameters
