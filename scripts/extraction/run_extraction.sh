#!/bin/bash
# Helper script to run Gemini extractions without Google Cloud SDK interference
#
# Usage:
#   ./scripts/extraction/run_extraction.sh test_langfuse_simple.py
#   ./scripts/extraction/run_extraction.sh run_10_examples.py

set -e

# Load .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check for API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY not set"
    echo "Set it in .env file or export it"
    exit 1
fi

# Get script name from first argument
SCRIPT_NAME="${1:-test_langfuse_simple.py}"

# Disable Google Cloud SDK to prevent ADC authentication conflicts
# This forces LangChain to use the API key instead
export CLOUDSDK_CONFIG=/dev/null

# Run the extraction script
echo "🚀 Running: $SCRIPT_NAME"
echo "📍 API Key: ${GOOGLE_API_KEY:0:20}..."
echo ""

python "scripts/extraction/$SCRIPT_NAME"
