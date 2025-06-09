#!/usr/bin/env python3
"""
CardioInsight AI - Web Application Runner

This script runs the CardioInsight AI web application.
"""

import os
import sys
import argparse

# Add parent directory to path to import CardioInsight AI modules
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


#from web_app.app import create_app
from app import create_app

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run CardioInsight AI Web Application')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the web application on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the web application on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    return parser.parse_args()

def main():
    """
    Main function to run the web application.
    """
    args = parse_args()
    
    # Create the Flask application
    app = create_app()
    
    # Run the application
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()

