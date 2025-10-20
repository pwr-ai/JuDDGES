#!/bin/bash

# API Documentation Generation Script
# Generates API reference documentation from Python docstrings using MkDocs

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${GREEN}JuDDGES API Documentation Generator${NC}"
echo "========================================"
echo

# Change to project root
cd "$PROJECT_ROOT"

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo -e "${RED}Error: mkdocs is not installed${NC}"
    echo "Install with: uv pip install mkdocs mkdocs-material mkdocstrings[python]"
    exit 1
fi

# Parse command line arguments
SERVE=false
BUILD=false
STRICT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --serve|-s)
            SERVE=true
            shift
            ;;
        --build|-b)
            BUILD=true
            shift
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --serve, -s      Serve documentation locally (http://127.0.0.1:8000)"
            echo "  --build, -b      Build static site to site/ directory"
            echo "  --strict         Enable strict mode (warnings as errors)"
            echo "  --help, -h       Show this help message"
            echo
            echo "Examples:"
            echo "  $0                  # Validate documentation"
            echo "  $0 --serve          # Serve locally"
            echo "  $0 --build          # Build for production"
            echo "  $0 --build --strict # Build with strict validation"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate mkdocs.yml
echo -e "${YELLOW}Validating configuration...${NC}"
if [ ! -f "mkdocs.yml" ]; then
    echo -e "${RED}Error: mkdocs.yml not found${NC}"
    exit 1
fi

# Check if juddges package is installed
if ! python -c "import juddges" 2>/dev/null; then
    echo -e "${YELLOW}Warning: juddges package not installed${NC}"
    echo "Installing in development mode..."
    uv pip install -e .
fi

# Validate documentation structure
echo -e "${YELLOW}Checking documentation structure...${NC}"
REQUIRED_DIRS=(
    "docs"
    "docs/reference"
    "docs/reference/api"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "${RED}Error: Required directory not found: $dir${NC}"
        exit 1
    fi
done

# Count API documentation files
API_DOCS_COUNT=$(find docs/reference/api -name "*.md" | wc -l)
echo -e "${GREEN}Found $API_DOCS_COUNT API documentation files${NC}"

# Build or serve based on flags
if [ "$SERVE" = true ]; then
    echo
    echo -e "${GREEN}Starting development server...${NC}"
    echo "Documentation will be available at: http://127.0.0.1:8000"
    echo "Press Ctrl+C to stop"
    echo

    if [ "$STRICT" = true ]; then
        mkdocs serve --strict
    else
        mkdocs serve
    fi

elif [ "$BUILD" = true ]; then
    echo
    echo -e "${YELLOW}Building static site...${NC}"

    # Clean previous build
    if [ -d "site" ]; then
        echo "Cleaning previous build..."
        rm -rf site
    fi

    # Build with or without strict mode
    if [ "$STRICT" = true ]; then
        mkdocs build --strict
    else
        mkdocs build
    fi

    echo
    echo -e "${GREEN}Build complete!${NC}"
    echo "Static site generated in: site/"
    echo "To view locally:"
    echo "  cd site && python -m http.server 8000"
    echo

else
    # Default: validate only
    echo
    echo -e "${YELLOW}Validating documentation...${NC}"

    if mkdocs build --strict --site-dir /tmp/mkdocs-test 2>&1; then
        echo -e "${GREEN}✓ Documentation is valid${NC}"
        rm -rf /tmp/mkdocs-test
    else
        echo -e "${RED}✗ Documentation has errors${NC}"
        exit 1
    fi

    echo
    echo "To view documentation:"
    echo "  $0 --serve"
    echo
    echo "To build for production:"
    echo "  $0 --build"
    echo
fi

# Show statistics
echo
echo -e "${GREEN}Documentation Statistics${NC}"
echo "========================"
echo "API documentation files: $API_DOCS_COUNT"
echo "Total markdown files: $(find docs -name "*.md" | wc -l)"
echo "Python modules: $(find juddges -name "*.py" -not -name "__*" | wc -l)"
echo

# Check for missing module documentation
echo -e "${YELLOW}Checking coverage...${NC}"

MISSING_DOCS=()

# Check key modules
KEY_MODULES=(
    "juddges/data/loaders.py:docs/reference/api/data/loaders.md"
    "juddges/llm/factory.py:docs/reference/api/llm/factory.md"
    "juddges/extraction/gemini_chain.py:docs/reference/api/extraction/gemini_chain.md"
    "juddges/preprocessing/text_chunker.py:docs/reference/api/preprocessing/text_chunker.md"
    "juddges/evals/metrics.py:docs/reference/api/evals/metrics.md"
)

for mapping in "${KEY_MODULES[@]}"; do
    module="${mapping%%:*}"
    doc="${mapping##*:}"

    if [ -f "$module" ] && [ ! -f "$doc" ]; then
        MISSING_DOCS+=("$module -> $doc")
    fi
done

if [ ${#MISSING_DOCS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All key modules documented${NC}"
else
    echo -e "${YELLOW}Missing documentation for:${NC}"
    for missing in "${MISSING_DOCS[@]}"; do
        echo "  - $missing"
    done
fi

echo
echo -e "${GREEN}Done!${NC}"
