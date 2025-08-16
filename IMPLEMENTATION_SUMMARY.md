"""
Multi-Constraint Query Processing Implementation Summary
======================================================

This document summarizes the complete implementation of multi-constraint query
processing as specified in issue #65.

## Problem Solved

The system now correctly handles complex queries like:
"List 5 latest decks that discuss project roadmaps"

Previously, such queries were treated as simple metadata queries, ignoring:
- Count constraints (5)
- File type constraints (decks → PPTX/PPT)
- Content constraints (project roadmaps)

## Solution Architecture

### Core Components Created/Modified

1. **ConstraintExtractor** (NEW: `backend/shared/constraints.py`)
   - Extracts count, file types, and content terms from natural language
   - Handles number parsing (digits + words: "5", "five")
   - Maps file type synonyms (decks → PPTX/PPT, docs → DOCX/DOC, etc.)
   - Detects content filtering needs by analyzing remaining terms
   - Applies safe bounds checking (1 ≤ count ≤ 100)

2. **Enhanced Orchestrator** (`orchestrator_agent.py`)
   - Modified `_create_metadata_query_plan` to use constraint extraction
   - Creates single-step plans for metadata-only queries
   - Creates two-step plans for content-filtered queries
   - Passes constraint parameters in ExecutionStep objects

3. **Enhanced Discovery Agent** (`discovery_agent.py`)
   - Added `_build_metadata_params` to read constraint parameters
   - Forwards count and file_type to metadata plugin
   - Handles count widening for content queries (multiplier = 3)

4. **Enhanced Analysis Agent** (`analysis_agent.py`)
   - Added `_prepare_constraint_content_params` for content filtering
   - Added `_get_target_docs_from_metadata` to extract file paths
   - Added `_apply_constraint_result_clipping` for exact count results
   - Supports target_docs parameter to restrict semantic search scope

5. **Configuration Support** (`config.py`)
   - Added `default_query_count`, `max_query_count`, `content_filtering_multiplier`
   - Made constraint behavior configurable

## Query Processing Flow

### Single-Step (Metadata-Only) Queries
Example: "List 5 latest decks"

1. ConstraintExtractor extracts: count=5, file_types=[PPTX,PPT], no content
2. Orchestrator creates single metadata step with parameters
3. Discovery agent forwards parameters to metadata plugin
4. Returns: 5 PowerPoint files by modification date

### Two-Step (Content-Filtered) Queries  
Example: "Show 3 recent spreadsheets about budget"

1. ConstraintExtractor extracts: count=3, file_types=[XLSX,XLS], content=["budget"]
2. Orchestrator creates two-step plan:
   - Step 1: Metadata query with widened count (3×3=9)
   - Step 2: Content filtering on metadata results
3. Discovery agent gets 9 Excel files from metadata plugin
4. Analysis agent filters by "budget" content, returns top 3 matches

## Success Criteria Validation

All examples from requirements work correctly:

✅ "List 5 latest decks" → 5 PPTX files, no content filter
✅ "Show 3 recent spreadsheets about budget" → 3 XLSX files about budget  
✅ "Latest 10 files" → 10 files, any type, by mod time
✅ "Decks about strategy" → All PPTX files about strategy

## Key Features

- **Backwards Compatible**: Simple queries work exactly as before
- **Deterministic**: Same query returns same results with stable ordering
- **Configurable**: Safe bounds and customizable behavior
- **Robust**: Handles edge cases, invalid inputs, and error conditions
- **Smart**: Optimal execution plans based on query complexity
- **Extensible**: Easy to add new file type mappings or constraint types

## Testing Coverage

1. **Unit Tests** (`test_constraints.py`)
   - Constraint extraction with all success criteria examples
   - Number parsing (digits and words)
   - File type mapping (single and multiple types)
   - Content term detection
   - Edge cases and bounds checking

2. **Integration Tests** (`test_standalone_constraints.py`)
   - End-to-end constraint extraction
   - Orchestrator plan creation logic
   - Parameter forwarding simulation
   - Success criteria validation

## File Changes Summary

### New Files
- `backend/src/shared/constraints.py` - Constraint extraction logic
- `tests/test_constraints.py` - Unit tests for constraint extraction  
- `tests/test_standalone_constraints.py` - Integration tests

### Modified Files
- `backend/src/querying/agents/agentic/orchestrator_agent.py` - Enhanced planning
- `backend/src/querying/agents/agentic/discovery_agent.py` - Parameter forwarding
- `backend/src/querying/agents/agentic/analysis_agent.py` - Content filtering
- `backend/src/shared/config.py` - Configuration parameters

## Requirements Compliance

### R1: Parse Query Constraints ✅
- Extracts count constraints: "5 latest" → count=5
- Extracts file type constraints: "decks" → [PPTX, PPT]  
- Extracts time constraints: "latest" → recency filter
- Extracts content constraints: "about X" → semantic search

### R2: Smart Orchestration Logic ✅
- Metadata-only queries → single metadata step
- Multi-constraint queries → two-step plan (metadata + content)
- Proper parameter passing between steps

### R3: Deterministic Results ✅
- Metadata sorting: ORDER BY modified_time DESC, file_name ASC
- Semantic sorting: similarity DESC, file_path ASC
- Reproducible results across runs

### R4: Parameter Passing ✅
- Discovery agent forwards count and file_type to metadata plugin
- Content queries use widened count in step 1
- Analysis agent receives target_docs from metadata results

## Error Handling

- Invalid counts (negative, >max) → use safe defaults
- Unknown file types → ignore type filter  
- No content terms → single metadata step
- Empty metadata results → appropriate messaging
- Plugin failures → graceful fallback

## Performance Considerations

- Content filtering multiplier prevents under-fetching in step 1
- Bounds checking prevents resource abuse (max 100 files)
- Deterministic sorting ensures consistent performance
- Backwards compatibility maintains existing query performance

## Security

- Safe bounds prevent resource exhaustion attacks
- Input validation prevents injection attacks
- No sensitive data exposure in constraint extraction
- Configuration-based limits for operational safety

## Future Enhancements

The implementation provides a solid foundation for future improvements:
- Additional file type mappings
- More sophisticated content term extraction
- Advanced time constraint parsing (specific dates, ranges)
- Multi-language support for constraint keywords
- Performance optimization for large result sets

## Conclusion

The multi-constraint query processing system is fully implemented and tested,
addressing all requirements from issue #65. Users can now query complex
requests and get exactly what they expect, while maintaining full backwards
compatibility for existing simple queries.
"""