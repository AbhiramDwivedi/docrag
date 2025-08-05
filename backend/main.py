"""Backend entry point for DocQuest.

This module provides the main entry points for DocQuest backend operations
including ingestion and querying.
"""

import sys
from pathlib import Path

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingestion.pipeline import main as ingestion_main
from backend.querying.api import app as api_app
from backend.interface.cli.ask import main as cli_main


def run_ingestion():
    """Run the document ingestion pipeline."""
    return ingestion_main()


def run_api():
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(api_app, host="0.0.0.0", port=8000)


def run_cli():
    """Run the CLI interface."""
    return cli_main()


if __name__ == "__main__":
    # Default to CLI mode
    run_cli()