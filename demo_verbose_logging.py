"""
Demonstration of DocQuest CLI Verbose Logging Functionality

This script demonstrates the different verbose levels implemented for the CLI.
"""

import subprocess
import sys
from pathlib import Path

def run_cli(args):
    """Run CLI command and capture output."""
    cmd = [sys.executable, "cli/ask.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    return result.stdout, result.stderr, result.returncode

def demo_verbose_levels():
    """Demonstrate different verbose levels."""
    
    print("="*60)
    print("DocQuest CLI Verbose Logging Demonstration")
    print("="*60)
    
    test_query = "find all pdf files"
    
    print(f"\nTest Query: \"{test_query}\"\n")
    
    # Test each verbose level
    levels = [
        (0, "Minimal (default)", "No verbose logging, only final results"),
        (1, "Info Level", "Agent reasoning, plugin selection, operation names"),
        (2, "Debug Level", "LLM prompts/responses, SQL queries, parameter generation"),
        (3, "Trace Level", "Full execution traces, timing information, raw data")
    ]
    
    for level, name, description in levels:
        print(f"{'='*20} LEVEL {level}: {name} {'='*20}")
        print(f"Description: {description}")
        print(f"Command: python cli/ask.py --verbose {level} \"{test_query}\"")
        print("-" * 60)
        
        stdout, stderr, returncode = run_cli(["--verbose", str(level), test_query])
        
        print("Output:")
        if stdout:
            print(stdout)
        if stderr:
            print("STDERR:", stderr)
        
        print(f"Return code: {returncode}")
        print()

if __name__ == "__main__":
    demo_verbose_levels()