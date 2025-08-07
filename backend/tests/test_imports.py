#!/usr/bin/env python3
"""Simple test script to check imports."""

import sys
from pathlib import Path

print("=== Import Test ===")
print(f"Current working directory: {Path('.').absolute()}")
print(f"Python executable: {sys.executable}")

# Add src to path
src_path = Path('./src')
sys.path.insert(0, str(src_path))
print(f"Added to Python path: {src_path.absolute()}")

print("\n=== Testing imports ===")

# Test 1: shared.config
try:
    from shared.config import get_settings
    print("✅ shared.config.get_settings import SUCCESS")
except ImportError as e:
    print(f"❌ shared.config.get_settings FAILED: {e}")

# Test 2: querying.agents.agent
try:
    from querying.agents.agent import Agent
    print("✅ querying.agents.agent.Agent import SUCCESS")
except ImportError as e:
    print(f"❌ querying.agents.agent.Agent FAILED: {e}")

print("\n=== Done ===")
