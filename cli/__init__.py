"""Backward compatibility shim for CLI.

DEPRECATED: This module provides backward compatibility for existing scripts.
Please migrate to the new workflow:

Old way:
    python -m cli.ask "question"

New way:
    cd backend/
    python -m interface.cli.ask "question"

This shim will be removed in a future version.
"""

import warnings
import sys
from pathlib import Path

def show_deprecation_warning():
    """Show deprecation warning for old CLI usage."""
    warnings.warn(
        "The 'cli' module is deprecated. Please migrate to the new workflow:\n"
        "Old: python -m cli.ask 'question'\n"
        "New: cd backend/ && python -m interface.cli.ask 'question'\n"
        "This shim will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

# Show warning when module is imported
show_deprecation_warning()

# Add backend to path for imports
backend_path = Path(__file__).parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Re-export the ask module for backward compatibility
try:
    from backend.interface.cli.ask import *
except ImportError:
    # Fallback error message
    print("Error: Could not import backend CLI modules.")
    print("Please ensure you're running from the project root and backend/ exists.")
    print("Migration instructions:")
    print("  cd backend/")
    print("  python -m interface.cli.ask 'your question'")
    sys.exit(1)