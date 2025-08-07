#!/usr/bin/env python3
"""Test script to check overall test status."""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run tests and summarize results."""
    print("=== Running All Tests ===")
    
    test_files = [
        "test_agent_framework.py",
        "test_plugin.py", 
        "test_verbose_logging.py",
        "test_performance.py",
    ]
    
    overall_results = {"passed": 0, "failed": 0, "errors": 0}
    
    for test_file in test_files:
        print(f"\n--- Testing {test_file} ---")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", f"tests/{test_file}", "-v", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse output for results
            output = result.stdout
            if "failed" in output:
                failed_count = output.count("FAILED")
                passed_count = output.count("PASSED")
                print(f"  ✅ {passed_count} passed, ❌ {failed_count} failed")
                overall_results["passed"] += passed_count
                overall_results["failed"] += failed_count
            elif "passed" in output:
                passed_count = output.count("PASSED")
                print(f"  ✅ {passed_count} passed")
                overall_results["passed"] += passed_count
            else:
                print(f"  ❓ Could not parse results")
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {test_file} timed out")
            overall_results["errors"] += 1
        except Exception as e:
            print(f"  💥 {test_file} error: {e}")
            overall_results["errors"] += 1
    
    print(f"\n=== Overall Results ===")
    print(f"✅ Passed: {overall_results['passed']}")
    print(f"❌ Failed: {overall_results['failed']}")
    print(f"💥 Errors: {overall_results['errors']}")
    
    total = sum(overall_results.values())
    if total > 0:
        success_rate = (overall_results['passed'] / total) * 100
        print(f"📊 Success Rate: {success_rate:.1f}%")

if __name__ == '__main__':
    run_tests()
