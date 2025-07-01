#!/usr/bin/env python3
"""Test runner for Weaviate ingestion tests."""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all Weaviate ingestion tests."""
    test_dir = Path(__file__).parent
    
    # Test files to run
    test_files = [
        "test_ingest_to_weaviate.py",
        "test_ingest_integration.py",
        "test_dataset_loader.py"
    ]
    
    print("Running Weaviate ingestion tests...")
    print("=" * 50)
    
    all_passed = True
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if not test_path.exists():
            print(f"Warning: Test file {test_file} not found, skipping...")
            continue
            
        print(f"\nRunning {test_file}...")
        print("-" * 30)
        
        try:
            # Run pytest for each test file
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v",
                "--tb=short"
            ], capture_output=False, text=True)
            
            if result.returncode != 0:
                print(f"FAILED: {test_file}")
                all_passed = False
            else:
                print(f"PASSED: {test_file}")
                
        except Exception as e:
            print(f"ERROR running {test_file}: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests PASSED! ✅")
        return 0
    else:
        print("Some tests FAILED! ❌")
        return 1


def run_specific_test_class(test_class: str = None):
    """Run a specific test class."""
    if not test_class:
        print("Please specify a test class name")
        return 1
        
    test_dir = Path(__file__).parent
    
    print(f"Running test class: {test_class}")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(test_dir),
            "-k", test_class,
            "-v",
            "--tb=short"
        ], capture_output=False, text=True)
        
        return result.returncode
        
    except Exception as e:
        print(f"ERROR running test class {test_class}: {e}")
        return 1


def run_quick_tests():
    """Run only quick unit tests (excluding integration tests)."""
    test_dir = Path(__file__).parent
    
    print("Running quick unit tests...")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(test_dir / "test_ingest_to_weaviate.py"),
            str(test_dir / "test_dataset_loader.py"),
            "-v",
            "--tb=short",
            "-m", "not integration"  # Exclude integration tests if marked
        ], capture_output=False, text=True)
        
        return result.returncode
        
    except Exception as e:
        print(f"ERROR running quick tests: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "quick":
            exit_code = run_quick_tests()
        elif command == "class" and len(sys.argv) > 2:
            exit_code = run_specific_test_class(sys.argv[2])
        else:
            print("Usage:")
            print("  python run_tests.py              # Run all tests")
            print("  python run_tests.py quick        # Run quick tests only")
            print("  python run_tests.py class <name> # Run specific test class")
            exit_code = 1
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)
