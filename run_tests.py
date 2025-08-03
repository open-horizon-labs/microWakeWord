#!/usr/bin/env python3
"""
Convenience script for running tests locally.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py unit         # Run only unit tests
    python run_tests.py integration  # Run only integration tests
    python run_tests.py fast         # Run all except slow tests
    python run_tests.py coverage     # Run with coverage report
"""

import os
import subprocess
import sys


def run_command(cmd):
    """Run a command and return exit code."""
    print(f"Running: {cmd}")
    return subprocess.call(cmd, shell=True)


def main():
    # Ensure we're in the project root
    if not os.path.exists("microwakeword"):
        print("Error: Must run from project root directory")
        sys.exit(1)

    # Parse command line arguments
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    # Install test dependencies if needed
    if not os.path.exists("tests"):
        print("Setting up test environment...")
        run_command("pip install -r requirements-test.txt")

    # Run tests based on mode
    if mode == "unit":
        print("\n=== Running unit tests ===")
        exit_code = run_command("pytest tests/unit -v")

    elif mode == "integration":
        print("\n=== Running integration tests ===")
        exit_code = run_command("pytest tests/integration -v")

    elif mode == "fast":
        print("\n=== Running fast tests only ===")
        exit_code = run_command('pytest -v -m "not slow"')

    elif mode == "coverage":
        print("\n=== Running tests with coverage ===")
        exit_code = run_command(
            "pytest --cov=microwakeword --cov-report=html --cov-report=term"
        )
        if exit_code == 0:
            print("\nCoverage report generated in htmlcov/index.html")

    elif mode == "all":
        print("\n=== Running all tests ===")
        exit_code = run_command("pytest -v")

    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)

    # Run linting if tests passed
    if exit_code == 0:
        print("\n=== Running linting ===")
        lint_code = run_command(
            "flake8 microwakeword tests --max-line-length=120 --ignore=E203,W503"
        )
        if lint_code != 0:
            print("Warning: Linting issues found")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
