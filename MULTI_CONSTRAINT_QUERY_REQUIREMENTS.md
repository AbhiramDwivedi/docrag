# Multi-Constraint Query Processing Requirements

## Problem Statement

**Current Issue**: Queries with multiple constraints return incorrect results:
- Query: `"List 5 latest decks that talk about value propositions"`
- Expected: 5 PowerPoint files, filtered by content about value propositions, sorted by modification date
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

**Interface**:
```python
@dataclass
class QueryConstraints:
    count: Optional[int] = None           # Extracted number: "5 latest" → 5
    file_types: List[str] = field(default_factory=list)  # ["PPTX", "PPT"] 
    has_content_filter: bool = False      # True if semantic terms present
    content_terms: List[str] = field(default_factory=list)  # ["value", "propositions"]
    sort_by: str = "modified_time"        # Fixed for latest queries
    sort_order: str = "desc"              # Fixed for latest queries

def extract_constraints(query: str) -> QueryConstraints:
    """Extract structured constraints from natural language query."""
```

**Implementation Notes**:
- Number extraction: regex for digits + word map ("five" → 5, "ten" → 10)
- Type mapping: {"decks": ["PPTX", "PPT"], "spreadsheets": ["XLSX", "XLS"], "docs": ["DOCX", "DOC"]}
- Content detection: Remove constraint words, check if meaningful terms remain
- Bound extracted numbers to safe range (1-100)

#### 2. Enhanced Orchestrator (`backend/src/querying/agents/agentic/orchestrator_agent.py`)
**Purpose**: Create constraint-aware execution plans

**Changes Required**:
```python
def _create_metadata_plan(self, intent: QueryIntent, query: str) -> ExecutionPlan:
    """Create plan for metadata queries with constraint awareness."""
    constraints = extract_constraints(query)  # NEW: Parse constraints
    
    if constraints.has_content_filter:
        # Two-step plan: metadata narrowing + content filtering
        return self._create_two_step_plan(constraints, query)
    else:
        # Single-step plan: metadata only  
        return self._create_single_step_plan(constraints, query)
```

#### 3. Enhanced Discovery Agent (`backend/src/querying/agents/agentic/discovery_agent.py`)
**Purpose**: Forward constraints to metadata plugin

**Changes Required**:
```python
def _execute_metadata_command(self, step: ExecutionStep, context: AgentContext) -> StepResult:
    """Execute metadata command with constraint parameters."""
    # Extract constraints from step parameters (set by orchestrator)
    constraints = step.parameters.get("constraints")  # NEW
    
    params = {
        "operation": self._determine_operation(query),
        "count": constraints.count or self._get_default_count(),     # NEW
        "file_type": constraints.file_types[0] if constraints.file_types else None,  # NEW
    }
    
    result = metadata_plugin.execute(params)  # NOW INCLUDES CONSTRAINTS
```

#### 4. Metadata Plugin (already supports parameters)
**Current State**: ✅ Already implemented in `_get_latest_files()`
- Supports `count` parameter
- Supports `file_type` parameter  
- Has deterministic ordering logic

**Required**: No changes needed, just parameter forwarding from Discovery Agent

### Implementation Plan

#### Phase 1: Constraint Extraction (1-2 hours)
1. Create `backend/shared/constraints.py` with `QueryConstraints` and `extract_constraints()`
2. Add unit tests in `tests/test_constraints.py`
3. Test with query examples to validate parsing

#### Phase 2: Orchestrator Integration (1 hour)  
1. Import constraint extractor in orchestrator
2. Modify `_create_metadata_plan()` to call `extract_constraints()`
3. Pass constraints in step parameters

#### Phase 3: Discovery Agent Forwarding (30 min)
1. Modify `_execute_metadata_command()` to read constraints from step parameters
2. Build metadata plugin params with count/file_type

#### Phase 4: Two-Step Content Filtering (1-2 hours)
1. Implement two-step plan creation in orchestrator
2. Modify analysis agent to accept `target_docs` parameter for restricted semantic search
3. Add count clipping with deterministic tie-breaking

#### Phase 5: Testing & Validation (1 hour)
1. Integration tests with deterministic dataset
2. Verify query: `"List 5 latest decks about value propositions"` returns exactly 5 PPTX files
3. Performance smoke tests

### Success Criteria

| Test Query | Expected Behavior | Validation |
|------------|------------------|------------|
| `"List 5 latest decks"` | 5 PPTX files, no content filter | ✅ Count + type constraints |
| `"Show 3 recent spreadsheets about budget"` | 3 XLSX files about budget | ✅ Count + type + content |
| `"Latest 10 files"` | 10 files, any type, by mod time | ✅ Count only |
| `"Decks about strategy"` | All PPTX files about strategy | ✅ Type + content (no count) |

### Configuration

Add to `backend/src/shared/config.yaml.template`:
```yaml
agent:
  constraints:
    default_count: 10           # Default when no count specified
    content_widen_factor: 3     # Multiply count for step 1 when content filtering
    max_count: 100             # Safety limit for extracted counts
```

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
