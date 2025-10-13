#!/bin/bash
# Large-scale extraction with configurable queries
# Configuration: batch_size=5, max_workers=20, sample_size=30000 per query

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Processing configuration
BATCH_SIZE=5
MAX_WORKERS=20
SAMPLE_SIZE=30000
MODEL="gemini-2.5-pro"

# Output configuration (auto-generated based on timestamp)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR="data/extraction_results/run_${TIMESTAMP}"
LOG_DIR="logs/extraction"

# ============================================================================
# QUERY LIST - Add your queries here
# ============================================================================
# Format: "query_name|search_query"
# - query_name: short identifier (no spaces, used for filenames)
# - search_query: full search query string

declare -a QUERIES=(
    "swiss_franc|kredyt hipoteczny we frankach szwajcarskich kredyt frankowy CHF"
    "ipbox|IP Box ulga podatkowa prawa własności intelektualnej"
    "employee_benefits|owocowe czwartki benefit pracowniczy karta multisport"
    "vat_tax|VAT podatek od towarów i usług"
    "cryptocurrency|kryptowaluty bitcoin ethereum podatek"
)

# ============================================================================
# EXECUTION
# ============================================================================

# Create directories
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Count queries
TOTAL_QUERIES=${#QUERIES[@]}
EXPECTED_DOCS=$((TOTAL_QUERIES * SAMPLE_SIZE))

echo "=========================================="
echo "Large-Scale Extraction Run"
echo "=========================================="
echo "Timestamp: $TIMESTAMP"
echo "Batch size: $BATCH_SIZE"
echo "Parallel workers: $MAX_WORKERS"
echo "Sample size per query: $SAMPLE_SIZE"
echo "Model: $MODEL"
echo "Output directory: $BASE_OUTPUT_DIR"
echo "Total queries: $TOTAL_QUERIES"
echo "Expected documents: ~$EXPECTED_DOCS"
echo "=========================================="
echo ""

# Process each query
QUERY_INDEX=1
declare -a QUERY_NAMES=()
declare -a QUERY_OUTPUT_DIRS=()

for QUERY_ENTRY in "${QUERIES[@]}"; do
    # Parse query entry
    IFS='|' read -r QUERY_NAME SEARCH_QUERY <<< "$QUERY_ENTRY"

    # Create query-specific output directory
    QUERY_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${QUERY_NAME}"
    mkdir -p "$QUERY_OUTPUT_DIR"

    # Store for summary
    QUERY_NAMES+=("$QUERY_NAME")
    QUERY_OUTPUT_DIRS+=("$QUERY_OUTPUT_DIR")

    # Log file path
    LOG_FILE="${LOG_DIR}/${QUERY_NAME}_${TIMESTAMP}.log"

    echo "[$QUERY_INDEX/$TOTAL_QUERIES] Processing: $QUERY_NAME"
    echo "  Query: '$SEARCH_QUERY'"
    echo "  Output: $QUERY_OUTPUT_DIR"
    echo "  Log: $LOG_FILE"
    echo ""

    # Run extraction
    python scripts/extraction/run_extraction_rest.py \
        --sample-size $SAMPLE_SIZE \
        --search-query "$SEARCH_QUERY" \
        --model $MODEL \
        --batch-size $BATCH_SIZE \
        --max-workers $MAX_WORKERS \
        --ingest-to-weaviate \
        --output-dir "$QUERY_OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"

    echo ""
    echo "  ✓ $QUERY_NAME completed"
    echo ""

    ((QUERY_INDEX++))
done

# ============================================================================
# SUMMARY REPORT
# ============================================================================

echo "=========================================="
echo "Extraction Summary Report"
echo "=========================================="
echo ""

TOTAL_DOCS=0
TOTAL_SUCCESS=0
TOTAL_FAILED=0
TOTAL_INGESTED=0
TOTAL_INGESTION_FAILED=0

for i in "${!QUERY_NAMES[@]}"; do
    QUERY_NAME="${QUERY_NAMES[$i]}"
    QUERY_OUTPUT_DIR="${QUERY_OUTPUT_DIRS[$i]}"

    echo "[$((i+1))/$TOTAL_QUERIES] $QUERY_NAME"
    echo "---"

    # Extraction summary
    if [ -f "${QUERY_OUTPUT_DIR}/extraction_summary.json" ]; then
        SUMMARY=$(cat "${QUERY_OUTPUT_DIR}/extraction_summary.json")
        echo "  Extraction:"
        echo "$SUMMARY" | jq -r '
            "    Total: \(.total_documents)",
            "    Success: \(.successful_extractions)",
            "    Failed: \(.failed_extractions)",
            "    Rate: \(.success_rate)%"
        '

        # Accumulate totals
        DOCS=$(echo "$SUMMARY" | jq -r '.total_documents // 0')
        SUCCESS=$(echo "$SUMMARY" | jq -r '.successful_extractions // 0')
        FAILED=$(echo "$SUMMARY" | jq -r '.failed_extractions // 0')
        TOTAL_DOCS=$((TOTAL_DOCS + DOCS))
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + SUCCESS))
        TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
    fi

    # Ingestion summary
    if [ -f "${QUERY_OUTPUT_DIR}/ingestion_report.json" ]; then
        INGESTION=$(cat "${QUERY_OUTPUT_DIR}/ingestion_report.json")
        echo "  Weaviate Ingestion:"
        echo "$INGESTION" | jq -r '
            "    Updated: \(.successful_updates)",
            "    Failed: \(.failed_updates)",
            "    Duration: \(.duration_seconds)s"
        '

        # Accumulate ingestion totals
        INGESTED=$(echo "$INGESTION" | jq -r '.successful_updates // 0')
        ING_FAILED=$(echo "$INGESTION" | jq -r '.failed_updates // 0')
        TOTAL_INGESTED=$((TOTAL_INGESTED + INGESTED))
        TOTAL_INGESTION_FAILED=$((TOTAL_INGESTION_FAILED + ING_FAILED))
    fi

    echo ""
done

# Overall totals
echo "=========================================="
echo "Overall Totals"
echo "=========================================="
echo "Extraction:"
echo "  Total documents: $TOTAL_DOCS"
echo "  Successful: $TOTAL_SUCCESS"
echo "  Failed: $TOTAL_FAILED"
if [ $TOTAL_DOCS -gt 0 ]; then
    OVERALL_RATE=$(awk "BEGIN {printf \"%.1f\", ($TOTAL_SUCCESS/$TOTAL_DOCS)*100}")
    echo "  Success rate: ${OVERALL_RATE}%"
fi
echo ""
echo "Weaviate Ingestion:"
echo "  Successful updates: $TOTAL_INGESTED"
echo "  Failed updates: $TOTAL_INGESTION_FAILED"
if [ $((TOTAL_INGESTED + TOTAL_INGESTION_FAILED)) -gt 0 ]; then
    ING_RATE=$(awk "BEGIN {printf \"%.1f\", ($TOTAL_INGESTED/($TOTAL_INGESTED+$TOTAL_INGESTION_FAILED))*100}")
    echo "  Success rate: ${ING_RATE}%"
fi
echo ""
echo "=========================================="
echo "Large-scale extraction completed!"
echo "Results saved to: $BASE_OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"
echo "=========================================="
