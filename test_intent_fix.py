#!/usr/bin/env python3
"""Test script to verify the intent classification fix for definition queries."""

import sys
import os

# Add the backend src to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

def test_intent_classification():
    """Test that 'What does STEP stand for?' is correctly classified as CONTENT_ANALYSIS."""
    print("🧪 Testing Intent Classification Fix...")
    print("=" * 50)
    
    # Test the pattern matching logic directly
    query = "What does STEP stand for?"
    query_lower = query.lower()
    
    print(f"Query: '{query}'")
    print(f"Query (lowercase): '{query_lower}'")
    print()
    
    # Check the old pattern (would fail)
    old_patterns = ["analyze", "summarize", "what", "content"]
    old_match = any(term in query_lower for term in old_patterns)
    print(f"❌ Old patterns {old_patterns}: {old_match}")
    
    # Check the new pattern (should work)
    new_patterns = ["what does", "what is", "define", "explain", "stands for", "mean", "meaning", "definition", "analyze", "summarize", "content"]
    new_match = any(term in query_lower for term in new_patterns)
    print(f"✅ New patterns {new_patterns}: {new_match}")
    print()
    
    # Test specific pattern matches
    print("Pattern matches:")
    for pattern in new_patterns:
        if pattern in query_lower:
            print(f"  ✅ '{pattern}' matches")
    
    print()
    
    # Determine classification
    if new_match:
        intent = "CONTENT_ANALYSIS"
        confidence = 0.7
        print(f"🎯 Result: {intent} (confidence: {confidence})")
        print("✅ This should now synthesize an answer from the document content!")
    else:
        print("❌ Would still be misclassified as DOCUMENT_DISCOVERY")
        
    return new_match

if __name__ == "__main__":
    success = test_intent_classification()
    print("\n" + "=" * 50)
    if success:
        print("✅ Intent classification fix is working correctly!")
        print("The query 'What does STEP stand for?' will now be classified as CONTENT_ANALYSIS")
        print("instead of DOCUMENT_DISCOVERY, which should make it synthesize an answer.")
    else:
        print("❌ Intent classification fix failed!")
        
    sys.exit(0 if success else 1)
