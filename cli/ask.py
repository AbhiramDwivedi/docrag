"""Backward compatibility ask module.

DEPRECATED: Please use the new workflow:
    cd backend/
    python -m interface.cli.ask "question"
"""

from . import show_deprecation_warning

# Show deprecation warning
show_deprecation_warning()

# Import from backend if available
try:
    from backend.interface.cli.ask import main, answer
except ImportError:
    def main():
        print("Error: Backend CLI not available.")
        print("Please migrate to: cd backend/ && python -m interface.cli.ask")
        return 1
    
    def answer(question: str) -> str:
        return "Error: Backend CLI not available. Please use: cd backend/ && python -m interface.cli.ask"

if __name__ == "__main__":
    main()