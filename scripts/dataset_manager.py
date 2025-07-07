#!/usr/bin/env python3
"""
Dataset Manager CLI - Main entry point for dataset management.

Usage examples:
    python scripts/dataset_manager.py list
    python scripts/dataset_manager.py preview "my-legal-dataset"
    python scripts/dataset_manager.py add "my-legal-dataset" --auto
    python scripts/dataset_manager.py validate "my-legal-dataset"
    python scripts/dataset_manager.py ingest "my-legal-dataset" --max-docs 1000
"""

from juddges.data.cli import app

if __name__ == "__main__":
    app()
