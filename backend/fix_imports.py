#!/usr/bin/env python3
"""Script to fix all hardcoded backend.src imports."""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix backend.src imports in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace backend.src imports with relative imports
    original_content = content
    
    # Pattern: from backend.src.X import Y -> from X import Y
    content = re.sub(r'from backend\.src\.(\S+) import', r'from \1 import', content)
    
    # Pattern: import backend.src.X -> import X  
    content = re.sub(r'import backend\.src\.(\S+)', r'import \1', content)
    
    if content != original_content:
        print(f"Fixing imports in: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix all Python files in src directory."""
    src_dir = Path('./src')
    if not src_dir.exists():
        print("Error: src directory not found")
        return
    
    fixed_count = 0
    for py_file in src_dir.rglob('*.py'):
        if fix_imports_in_file(py_file):
            fixed_count += 1
    
    print(f"Fixed imports in {fixed_count} files")

if __name__ == '__main__':
    main()
