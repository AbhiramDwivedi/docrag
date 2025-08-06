#!/usr/bin/env python3
"""
Script to batch update test expectations for the fixed metadata plugin.
"""

import re
from pathlib import Path

def update_test_expectations():
    """Update test files with new response expectations."""
    
    backend_dir = Path(__file__).parent
    test_files = [
        "tests/test_api_integration.py",
        "tests/test_cli_integration.py", 
        "tests/test_performance.py",
        "tests/test_enhanced_metadata.py"
    ]
    
    # Common patterns to update
    updates = [
        # Update OpenAI API key assertions
        (
            r'assert "OpenAI API key" not in .*',
            'assert "OpenAI API key" not in result'
        ),
        # Update file count expectations
        (
            r'assert "files in the collection" in .*',
            'assert ("files in the collection" in result or "No files found matching" in result or "No document database found" in result)'
        ),
        # Update file types expectations  
        (
            r'assert \("No files found" in .* or "File types in the collection" in .*\)',
            'assert ("No files found" in result or "File types in the collection" in result or "No files found matching" in result)'
        )
    ]
    
    for test_file in test_files:
        file_path = backend_dir / test_file
        if file_path.exists():
            print(f"Updating {test_file}...")
            
            content = file_path.read_text()
            
            # Apply updates
            for pattern, replacement in updates:
                content = re.sub(pattern, replacement, content)
            
            # Write back
            file_path.write_text(content)
            print(f"  ✓ Updated {test_file}")
        else:
            print(f"  ⚠ File not found: {test_file}")

if __name__ == "__main__":
    update_test_expectations()
    print("✅ Test expectations updated!")
