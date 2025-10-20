#!/bin/bash
# Run distributed extraction locally with multiple workers
#
# Usage:
#   ./scripts/extraction/run_distributed.sh --num-workers 20 --max-documents 10000 --query "kredyt frankowy"

set -e

# Default configuration
NUM_WORKERS=20
MAX_DOCUMENTS=1000
SEARCH_QUERIES="kredyt frankowy"
REDIS_URL="redis://localhost:6379"
BATCH_SIZE=3
MODEL="gemini-2.5-pro"
JOB_BATCH_SIZE=50

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --max-documents)
            MAX_DOCUMENTS="$2"
            shift 2
            ;;
        --query|--search-queries)
            SEARCH_QUERIES="$2"
            shift 2
            ;;
        --redis-url)
            REDIS_URL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --job-batch-size)
            JOB_BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --num-workers NUM          Number of parallel workers (default: 20)"
            echo "  --max-documents NUM        Maximum number of documents to extract (default: 1000)"
            echo "  --search-queries QUERY     Search query for documents (default: 'kredyt frankowy')"
            echo "  --redis-url URL            Redis connection URL (default: redis://localhost:6379)"
            echo "  --batch-size NUM           Extraction batch size (default: 3)"
            echo "  --model NAME               Gemini model (default: gemini-2.5-pro)"
            echo "  --job-batch-size NUM       Documents per job (default: 50)"
            echo ""
            echo "Example:"
            echo "  $0 --num-workers 50 --max-documents 100000 --query 'kredyt frankowy'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Distributed Extraction - Local Setup"
echo "=========================================="
echo "Workers: $NUM_WORKERS"
echo "Max documents: $MAX_DOCUMENTS"
echo "Search queries: $SEARCH_QUERIES"
echo "Batch size: $BATCH_SIZE"
echo "Model: $MODEL"
echo "Redis: $REDIS_URL"
echo "=========================================="
echo ""

# Check if Redis is running
echo "[1/4] Checking Redis..."
if ! redis-cli -u "$REDIS_URL" ping > /dev/null 2>&1; then
    echo "❌ Redis not running at $REDIS_URL"
    echo "Start Redis with: redis-server"
    exit 1
fi
echo "✓ Redis is running"

# Clear existing queue
echo ""
echo "[2/4] Clearing existing queue..."
redis-cli -u "$REDIS_URL" DEL extraction_queue
echo "✓ Queue cleared"

# Queue extraction jobs
echo ""
echo "[3/4] Queuing extraction jobs..."
python scripts/extraction/coordinator.py \
    --search-queries "$SEARCH_QUERIES" \
    --max-documents "$MAX_DOCUMENTS" \
    --job-batch-size "$JOB_BATCH_SIZE" \
    --redis-url "$REDIS_URL"

if [ $? -ne 0 ]; then
    echo "❌ Failed to queue jobs"
    exit 1
fi
echo "✓ Jobs queued successfully"

# Start workers in background
echo ""
echo "[4/4] Starting $NUM_WORKERS workers..."

WORKER_PIDS=()
for i in $(seq 1 $NUM_WORKERS); do
    python scripts/extraction/worker.py \
        --worker-id "$i" \
        --redis-url "$REDIS_URL" \
        --batch-size "$BATCH_SIZE" \
        --model "$MODEL" \
        --langfuse-sample-rate 0.01 \
        > "logs/worker_${i}.log" 2>&1 &

    WORKER_PIDS+=($!)
    echo "  Started worker $i (PID: $!)"
done

echo ""
echo "✓ All $NUM_WORKERS workers started"
echo ""
echo "Logs:"
for i in $(seq 1 $NUM_WORKERS); do
    echo "  Worker $i: logs/worker_${i}.log"
done

# Monitor queue
echo ""
echo "=========================================="
echo "Monitoring extraction progress"
echo "=========================================="
echo "Press Ctrl+C to stop monitoring (workers will continue)"
echo ""

trap "echo ''; echo 'Monitoring stopped. Workers still running in background.'; exit 0" INT

while true; do
    QUEUE_LENGTH=$(redis-cli -u "$REDIS_URL" LLEN extraction_queue)
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$TIMESTAMP] Queue length: $QUEUE_LENGTH jobs remaining"

    if [ "$QUEUE_LENGTH" -eq 0 ]; then
        echo ""
        echo "✓ Queue empty - extraction complete!"
        break
    fi

    sleep 10
done

# Wait for all workers to finish
echo ""
echo "Waiting for workers to finish..."
for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "=========================================="
echo "✓ Extraction Complete!"
echo "=========================================="
echo "Check logs in: logs/worker_*.log"
