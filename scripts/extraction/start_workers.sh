#!/bin/bash
# Start multiple extraction workers
#
# Usage:
#   ./scripts/extraction/start_workers.sh [NUM_WORKERS]
#
# Example:
#   ./scripts/extraction/start_workers.sh 20

set -e

# Configuration
NUM_WORKERS=${1:-20}
REDIS_URL=${REDIS_URL:-redis://localhost:6381}
QUEUE_NAME=${QUEUE_NAME:-extraction_queue}
BATCH_SIZE=${BATCH_SIZE:-10}
MAX_FETCH_THREADS=${MAX_FETCH_THREADS:-10}
MAX_EXTRACTION_THREADS=${MAX_EXTRACTION_THREADS:-3}
LOG_DIR=${LOG_DIR:-logs/workers}

# Create log directory
mkdir -p "$LOG_DIR"

# Cleanup function
cleanup() {
    echo "Stopping all workers..."
    pkill -P $$ || true
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Starting $NUM_WORKERS extraction workers..."
echo "Redis URL: $REDIS_URL"
echo "Queue: $QUEUE_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Fetch threads: $MAX_FETCH_THREADS"
echo "Extraction threads: $MAX_EXTRACTION_THREADS"
echo "Logs: $LOG_DIR"
echo ""

# Start workers
for i in $(seq 1 $NUM_WORKERS); do
    python scripts/extraction/worker.py \
        --worker-id $i \
        --redis-url "$REDIS_URL" \
        --queue-name "$QUEUE_NAME" \
        --batch-size "$BATCH_SIZE" \
        --max-fetch-threads "$MAX_FETCH_THREADS" \
        --max-extraction-threads "$MAX_EXTRACTION_THREADS" \
        > "$LOG_DIR/worker_${i}.log" 2>&1 &

    echo "Started worker $i (PID: $!)"
done

echo ""
echo "All $NUM_WORKERS workers started!"
echo ""
echo "Monitor logs:"
echo "  tail -f $LOG_DIR/worker_1.log"
echo "  tail -f $LOG_DIR/worker_*.log"
echo ""
echo "Check queue status:"
echo "  redis-cli -p 6381 LLEN $QUEUE_NAME"
echo ""
echo "Stop all workers:"
echo "  pkill -f 'worker.py'"
echo ""
echo "Press Ctrl+C to stop all workers"

# Wait for all background jobs
wait
