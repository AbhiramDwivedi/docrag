#!/usr/bin/env python3
"""Backward compatibility shim for old CLI path.

This allows 'python -m cli.ask' to still work by redirecting to the new interface.
"""

import sys
import subprocess
from pathlib import Path

# Redirect to new interface
if __name__ == "__main__":
    # Replace 'cli.ask' with 'interface.cli.ask' in the command
    new_cmd = [sys.executable, "-m", "interface.cli.ask"] + sys.argv[1:]
    
    # Execute with the same environment and directory
    result = subprocess.run(new_cmd, cwd=Path.cwd())
    sys.exit(result.returncode)