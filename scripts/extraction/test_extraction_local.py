"""Quick test extraction using local Weaviate."""

import os

# Set environment BEFORE importing anything
os.environ["WEAVIATE_HOST"] = "localhost"
os.environ["WEAVIATE_PORT"] = "8084"
os.environ["WEAVIATE_GRPC_PORT"] = "50051"
os.environ["WEAVIATE_API_KEY"] = ""  # No auth for local

import sys
sys.path.insert(0, "/home/laugustyniak/github/legal-ai/JuDDGES")

from scripts.extraction.run_extraction_sample import main

if __name__ == "__main__":
    # Override command line args
    sys.argv = [
        "test_extraction_local.py",
        "--sample-size", "5",
        "--model", "gemini-2.5-flash",
        "--output-dir", "data/extraction_results",
    ]
    main()
