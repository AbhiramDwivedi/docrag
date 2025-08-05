#!/usr/bin/env python3
"""Backward compatibility for ask.py - redirects to interface.cli.ask"""

import sys
from pathlib import Path

# Add root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the real ask module
from interface.cli.ask import *

if __name__ == "__main__":
    from interface.cli.ask import main
    main()