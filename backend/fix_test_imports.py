#!/usr/bin/env python3
"""Script to fix all test file imports."""

import re
from pathlib import Path

def fix_test_imports(file_path):
    """Fix backend.src imports in test files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix the sys.path setup to include src directory
    if 'sys.path.insert(0, str(Path(__file__).parent.parent))' in content:
        content = content.replace(
            'sys.path.insert(0, str(Path(__file__).parent.parent))',
            'sys.path.insert(0, str(Path(__file__).parent.parent))\nsys.path.insert(0, str(Path(__file__).parent.parent / "src"))'
        )
    
    # Replace backend.src imports with direct imports
    content = re.sub(r'from backend\.src\.(\S+) import', r'from \1 import', content)
    content = re.sub(r'import backend\.src\.(\S+)', r'import \1', content)
    
    if content != original_content:
        print(f"Fixing imports in: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all test files."""
    tests_dir = Path('./tests')
    if not tests_dir.exists():
        print("Error: tests directory not found")
        return
    
    fixed_count = 0
    for test_file in tests_dir.glob('test_*.py'):
        if fix_test_imports(test_file):
            fixed_count += 1
    
    print(f"Fixed imports in {fixed_count} test files")

if __name__ == '__main__':
    main()
