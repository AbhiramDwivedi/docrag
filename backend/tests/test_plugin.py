import sys
import os
import pytest
import sqlite3
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.src.querying.agents.plugins.metadata_commands import MetadataCommandsPlugin


class TestMetadataCommandsPlugin:
    """Test the metadata commands plugin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = MetadataCommandsPlugin()
    
    @pytest.mark.skip(reason="Requires database setup - temporarily disabled for restructuring")
    def test_msg_files(self):
        """Test MSG file operations."""
        print("=== Test 1: MSG Files ===")
        params1 = {
            "operation": "get_latest_files",
            "file_type": "MSG",
            "count": 10
        }

        # Use temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            conn = sqlite3.connect(tmp_db.name)
        try:
            result1 = self.plugin._get_latest_files(params1, conn)
            print(f"Result: {result1['response']}")
            print(f"Count: {result1['metadata']['count']}")
        finally:
            conn.close()
    
    @pytest.mark.skip(reason="Requires database setup - temporarily disabled for restructuring") 
    def test_docx_files(self):
        """Test DOCX file operations."""
        print("=== Test 2: DOCX Files ===")
        params2 = {
            "operation": "get_latest_files",
            "file_type": "DOCX",
            "count": 3
        }

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            conn = sqlite3.connect(tmp_db.name)
        try:
            result2 = self.plugin._get_latest_files(params2, conn)
            print(f"Result: {result2['response']}")
            print(f"Count: {result2['metadata']['count']}")
        finally:
            conn.close()

    @pytest.mark.skip(reason="Requires database setup - temporarily disabled for restructuring") 
    def test_ppt_files(self):
        """Test PPT file operations."""
        print("\n=== Test 2: PPT Files ===")
        params2 = {
            "operation": "get_latest_files",
            "file_type": "PPT",
            "count": 5,
            "time_filter": "recent"
        }

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            conn = sqlite3.connect(tmp_db.name)
        try:
            result2 = self.plugin._get_latest_files(params2, conn)
            print(f"Result: {result2['response']}")
            print(f"Count: {result2['metadata']['count']}")
        finally:
            conn.close()

    @pytest.mark.skip(reason="Requires database setup - temporarily disabled for restructuring")
    def test_pptx_files(self):
        """Test PPTX file operations."""
        print("\n=== Test 3: PPTX Files ===")
        params3 = {
            "operation": "get_latest_files",
            "file_type": "PPTX",
            "count": 5,
            "time_filter": "recent"
        }

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            conn = sqlite3.connect(tmp_db.name)
        try:
            result3 = self.plugin._get_latest_files(params3, conn)
            print(f"Result: {result3['response']}")
            print(f"Count: {result3['metadata']['count']}")
        finally:
            conn.close()
