import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docquest.querying.agents.plugins.metadata_commands import MetadataCommandsPlugin
import sqlite3

# Test the exact parameters that would be generated
plugin = MetadataCommandsPlugin()

# Test 1: MSG files (should work)
print("=== Test 1: MSG Files ===")
params1 = {
    "operation": "get_latest_files",
    "file_type": "MSG",
    "count": 10
}

conn = sqlite3.connect('data/docmeta.db')
try:
    result1 = plugin._get_latest_files(params1, conn)
    print(f"Result: {result1['response']}")
    print(f"Count: {result1['metadata']['count']}")
finally:
    conn.close()

# Test 2: PPT files (should fail because no PPT files exist)
print("\n=== Test 2: PPT Files ===")
params2 = {
    "operation": "get_latest_files",
    "file_type": "PPT",
    "count": 5,
    "time_filter": "recent"
}

conn = sqlite3.connect('data/docmeta.db')
try:
    result2 = plugin._get_latest_files(params2, conn)
    print(f"Result: {result2['response']}")
    print(f"Count: {result2['metadata']['count']}")
finally:
    conn.close()

# Test 3: PPTX files (should work)
print("\n=== Test 3: PPTX Files ===")
params3 = {
    "operation": "get_latest_files",
    "file_type": "PPTX",
    "count": 5,
    "time_filter": "recent"
}

conn = sqlite3.connect('data/docmeta.db')
try:
    result3 = plugin._get_latest_files(params3, conn)
    print(f"Result: {result3['response']}")
    print(f"Count: {result3['metadata']['count']}")
finally:
    conn.close()
