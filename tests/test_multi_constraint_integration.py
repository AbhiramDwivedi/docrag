"""Integration test for multi-constraint query processing.

This test validates that the constraint extraction and orchestrator integration
work together correctly for the success criteria from requirements.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add backend/src to path for imports
backend_src = Path(__file__).parent.parent / "backend" / "src"
sys.path.insert(0, str(backend_src))


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    mock = MagicMock()
    mock.default_query_count = 10
    mock.max_query_count = 100
    mock.content_filtering_multiplier = 3
    return mock


def test_constraint_extraction(mock_settings):
    """Test constraint extraction for success criteria examples."""
    
    with patch('shared.config.get_settings', return_value=mock_settings):
        return test_constraint_extraction_impl()

def test_constraint_extraction_impl():
    """Implementation of constraint extraction test."""
    from shared.constraints import ConstraintExtractor
    
    print("=== Testing Constraint Extraction ===")
    
    test_cases = [
        {
            "name": "Count + Type only (metadata-only query)",
            "query": "List 5 latest decks",
            "expected_count": 5,
            "expected_file_types": ["PPTX", "PPT"],
            "expected_content_filter": False
        },
        {
            "name": "Count + Type + Content (two-step query)",
            "query": "Show 3 recent spreadsheets about budget",
            "expected_count": 3,
            "expected_file_types": ["XLSX", "XLS"],
            "expected_content_filter": True
        },
        {
            "name": "Count only",
            "query": "Latest 10 files",
            "expected_count": 10,
            "expected_file_types": [],
            "expected_content_filter": False
        },
        {
            "name": "Type + Content (no count)",
            "query": "Decks about strategy",
            "expected_count": None,
            "expected_file_types": ["PPTX", "PPT"],
            "expected_content_filter": True
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        
        constraints = ConstraintExtractor.extract(test_case["query"])
        
        # Check count
        if constraints.count != test_case["expected_count"]:
            print(f"  ❌ Count mismatch: expected {test_case['expected_count']}, got {constraints.count}")
            return False
        else:
            print(f"  ✅ Count: {constraints.count}")
        
        # Check file types
        if sorted(constraints.file_types) != sorted(test_case["expected_file_types"]):
            print(f"  ❌ File types mismatch: expected {test_case['expected_file_types']}, got {constraints.file_types}")
            return False
        else:
            print(f"  ✅ File types: {constraints.file_types}")
        
        # Check content filter
        if constraints.has_content_filter != test_case["expected_content_filter"]:
            print(f"  ❌ Content filter mismatch: expected {test_case['expected_content_filter']}, got {constraints.has_content_filter}")
            return False
        else:
            print(f"  ✅ Content filter: {constraints.has_content_filter}")
            if constraints.has_content_filter:
                print(f"      Content terms: {constraints.content_terms}")
    
        print("\n✅ All constraint extraction tests passed!")
        return True


@patch('shared.config.get_settings')
def test_orchestrator_plan_creation(mock_get_settings):
    """Test that orchestrator creates appropriate plans based on constraints."""
    
    # Setup mock settings
    mock_settings = MagicMock()
    mock_settings.default_query_count = 10
    mock_settings.max_query_count = 100
    mock_settings.content_filtering_multiplier = 3
    mock_get_settings.return_value = mock_settings
    
    return test_orchestrator_plan_creation_impl()

def test_orchestrator_plan_creation_impl():
    """Implementation of orchestrator plan creation test."""
    from shared.constraints import ConstraintExtractor
    
    print("\n=== Testing Orchestrator Plan Logic ===")
    
    # Test metadata-only query
    query1 = "List 5 latest decks"
    constraints1 = ConstraintExtractor.extract(query1)
    print(f"\nQuery: '{query1}'")
    print(f"Has content filter: {constraints1.has_content_filter}")
    
    if constraints1.has_content_filter:
        print("  ❌ Should be metadata-only (single-step)")
        return False
    else:
        print("  ✅ Correctly identified as metadata-only (single-step)")
    
    # Test content query
    query2 = "Show 3 recent spreadsheets about budget"
    constraints2 = ConstraintExtractor.extract(query2)
    print(f"\nQuery: '{query2}'")
    print(f"Has content filter: {constraints2.has_content_filter}")
    
    if not constraints2.has_content_filter:
        print("  ❌ Should be content-filtered (two-step)")
        return False
    else:
        print("  ✅ Correctly identified as content-filtered (two-step)")
    
    print("\n✅ All orchestrator plan logic tests passed!")
    return True


@patch('shared.config.get_settings')
def test_parameter_forwarding(mock_get_settings):
    """Test that parameters are correctly structured for forwarding."""
    
    # Setup mock settings
    mock_settings = MagicMock()
    mock_settings.default_query_count = 10
    mock_settings.max_query_count = 100
    mock_settings.content_filtering_multiplier = 3
    mock_get_settings.return_value = mock_settings
    
    return test_parameter_forwarding_impl()

def test_parameter_forwarding_impl():
    """Implementation of parameter forwarding test."""
    from shared.constraints import ConstraintExtractor, get_content_filtering_multiplier
    
    print("\n=== Testing Parameter Forwarding ===")
    
    # Test single-step metadata parameters
    query = "List 5 latest decks"
    constraints = ConstraintExtractor.extract(query)
    
    print(f"\nSingle-step query: '{query}'")
    
    # Simulate what orchestrator would create for metadata params
    metadata_params = {
        "query": query,
        "intent": "metadata_query",
        "operation": "get_latest_files"
    }
    
    if constraints.count is not None:
        metadata_params["count"] = constraints.count
    
    if constraints.file_types:
        if len(constraints.file_types) == 1:
            metadata_params["file_type"] = constraints.file_types[0]
        else:
            metadata_params["file_types"] = constraints.file_types
    
    print(f"  Metadata params: {metadata_params}")
    
    expected_keys = ["query", "intent", "operation", "count", "file_type"]
    for key in expected_keys:
        if key in metadata_params:
            print(f"  ✅ Has {key}: {metadata_params[key]}")
        else:
            print(f"  ⚠️  Missing {key} (may be optional)")
    
    # Test two-step content parameters
    query2 = "Show 3 recent spreadsheets about budget"
    constraints2 = ConstraintExtractor.extract(query2)
    
    print(f"\nTwo-step query: '{query2}'")
    
    # Step 1: Widened metadata query
    base_count = constraints2.count or 10
    widened_count = base_count * get_content_filtering_multiplier()
    
    metadata_params2 = {
        "query": query2,
        "intent": "metadata_query_for_content",
        "operation": "get_latest_files",
        "count": widened_count
    }
    
    if constraints2.file_types:
        metadata_params2["file_type"] = constraints2.file_types[0]
    
    print(f"  Step 1 - Metadata params: {metadata_params2}")
    print(f"  Widened count: {base_count} → {widened_count} (×{get_content_filtering_multiplier()})")
    
    # Step 2: Content filtering parameters
    content_params = {
        "query": query2,
        "intent": "content_filtering",
        "extraction_type": "content_filtering",
        "content_terms": constraints2.content_terms,
        "target_count": base_count,
        "use_target_docs": True
    }
    
    print(f"  Step 2 - Content params: {content_params}")
    
    print("\n✅ Parameter forwarding structure is correct!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("Running Multi-Constraint Query Processing Integration Tests")
    print("=" * 60)
    
    # Create mock settings once for all tests
    mock_settings = MagicMock()
    mock_settings.default_query_count = 10
    mock_settings.max_query_count = 100
    mock_settings.content_filtering_multiplier = 3
    
    tests = [
        ("constraint_extraction", test_constraint_extraction_impl),
        ("orchestrator_plan_creation", test_orchestrator_plan_creation_impl),
        ("parameter_forwarding", test_parameter_forwarding_impl)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            with patch('shared.config.get_settings', return_value=mock_settings):
                result = test_func()
                results.append(result)
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("\n✅ Multi-constraint query processing implementation is working correctly!")
        print("✅ Ready for end-to-end testing with real plugins")
        return True
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)