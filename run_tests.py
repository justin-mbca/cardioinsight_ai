#!/usr/bin/env python3
"""
CardioInsight AI - Run Tests

This script runs all tests for the CardioInsight AI system.
"""

import os
import sys
import unittest
import argparse

def run_tests(verbose=False):
    """
    Run all tests.
    
    Parameters:
    -----------
    verbose : bool
        Whether to run tests in verbose mode. Default is False.
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

def main():
    """
    Main function.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CardioInsight AI tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Run tests in verbose mode')
    
    args = parser.parse_args()
    
    # Run tests
    return run_tests(verbose=args.verbose)

if __name__ == '__main__':
    sys.exit(main())

