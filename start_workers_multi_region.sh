#!/bin/bash
# Launch distributed workers across multiple Vertex AI regions
# This distributes load and avoids rate limiting by using separate regional quotas
#
# Usage:
#   chmod +x start_workers_multi_region.sh
#   ./start_workers_multi_region.sh
#
# Configuration:
# - 18 regions across US, EU, and Asia-Pacific
# - 3 workers per region (54 total workers)
# - Batch size: 10 documents per extraction
# - Model: gemini-2.5-flash
# - Langfuse: enabled with 5% sampling (reduced to avoid 413 errors)

set -e

# Kill any existing workers
echo "=========================================="
echo "Checking for existing workers..."
echo "=========================================="

EXISTING_WORKERS=$(pgrep -f 'scripts/extraction/worker.py' | wc -l)

if [ "$EXISTING_WORKERS" -gt 0 ]; then
    echo "Found $EXISTING_WORKERS running workers. Stopping them..."
    pkill -SIGTERM -f 'scripts/extraction/worker.py'
    sleep 2

    # Force kill if still running
    REMAINING=$(pgrep -f 'scripts/extraction/worker.py' | wc -l)
    if [ "$REMAINING" -gt 0 ]; then
        echo "Force killing $REMAINING remaining workers..."
        pkill -9 -f 'scripts/extraction/worker.py'
        sleep 1
    fi

    echo "All existing workers stopped."
else
    echo "No existing workers found."
fi

echo ""

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Disable PostgreSQL cache to avoid "too many clients" error with 45 workers
# The cache is useful but causes connection pool exhaustion with many workers
export POSTGRES_CACHE_URL=""

# Check Redis URL
if [ -z "$REDIS_URL" ]; then
    echo "Error: REDIS_URL not set. Please set it in .env or environment."
    echo "Format: redis://:PASSWORD@host:port/db"
    exit 1
fi

# Configuration
BATCH_SIZE=10
MODEL="gemini-2.5-flash"
LANGFUSE_SAMPLE_RATE=0.05  # 5% sampling (reduced to avoid 413 errors)
REDIS_QUEUE="extraction_queue"

# Workers per region
WORKERS_PER_REGION=3

# Define regions and worker ranges
# gemini-2.5-flash is available in many regions worldwide
# Using 18 regions × 3 workers = 54 workers total
# Distributed across US, EU, and Asia-Pacific for optimal load balancing
declare -A REGIONS=(
    # United States (7 regions)
    ["us-central1"]="1 3"          # Workers 1-3 (Iowa)
    ["us-east1"]="4 6"             # Workers 4-6 (South Carolina)
    ["us-east4"]="7 9"             # Workers 7-9 (Virginia)
    ["us-west1"]="10 12"           # Workers 10-12 (Oregon)
    ["us-west4"]="13 15"           # Workers 13-15 (Nevada)
    ["us-east5"]="16 18"           # Workers 16-18 (Ohio)
    ["us-south1"]="19 21"          # Workers 19-21 (Dallas)
    # Europe (9 regions)
    ["europe-west1"]="22 24"       # Workers 22-24 (Belgium)
    ["europe-west2"]="25 27"       # Workers 25-27 (London)
    ["europe-west3"]="28 30"       # Workers 28-30 (Frankfurt)
    ["europe-west4"]="31 33"       # Workers 31-33 (Netherlands)
    ["europe-west6"]="34 36"       # Workers 34-36 (Zürich)
    ["europe-central2"]="37 39"    # Workers 37-39 (Warsaw, Poland!)
    ["europe-southwest1"]="40 42"  # Workers 40-42 (Madrid, Spain)
    ["europe-west8"]="43 45"       # Workers 43-45 (Milan, Italy)
    ["europe-north1"]="46 48"      # Workers 46-48 (Finland)
    # Asia-Pacific (2 regions)
    ["asia-northeast1"]="49 51"    # Workers 49-51 (Tokyo)
    ["asia-southeast1"]="52 54"    # Workers 52-54 (Singapore)
)

TOTAL_WORKERS=$((${#REGIONS[@]} * $WORKERS_PER_REGION))

echo "=========================================="
echo "Multi-Region Worker Launcher"
echo "=========================================="
echo "Configuration:"
echo "  Regions: ${#REGIONS[@]} (${!REGIONS[@]})"
echo "  Workers per region: $WORKERS_PER_REGION"
echo "  Total workers: $TOTAL_WORKERS"
echo "  Batch size: $BATCH_SIZE documents"
echo "  Model: $MODEL"
echo "  Langfuse sampling: ${LANGFUSE_SAMPLE_RATE}%"
echo "  Redis queue: $REDIS_QUEUE"
echo "=========================================="
echo ""

# Launch workers for each region
for region in "${!REGIONS[@]}"; do
    # Parse worker range
    read -r start_id end_id <<< "${REGIONS[$region]}"

    echo "Starting 3 workers in region: $region (IDs: $start_id-$end_id)"

    for worker_id in $(seq $start_id $end_id); do
        python scripts/extraction/worker.py \
            --worker-id $worker_id \
            --region "$region" \
            --redis-url "$REDIS_URL" \
            --queue-name "$REDIS_QUEUE" \
            --batch-size $BATCH_SIZE \
            --model "$MODEL" \
            --use-langfuse \
            --langfuse-sample-rate $LANGFUSE_SAMPLE_RATE \
            --max-fetch-threads 10 \
            --max-extraction-threads 3 \
            > logs/worker_${worker_id}_${region}.log 2>&1 &

        echo "  - Worker $worker_id started (PID: $!, Region: $region)"
    done

    echo ""
done

echo "=========================================="
echo "All $TOTAL_WORKERS workers started!"
echo "=========================================="
echo ""
echo "Logs: logs/worker_<id>_<region>.log"
echo ""
echo "Monitor workers:"
echo "  ps aux | grep 'worker.py' | grep -v grep"
echo ""
echo "Monitor queue:"
echo "  redis-cli -u \$REDIS_URL LLEN $REDIS_QUEUE"
echo ""
echo "Stop all workers:"
echo "  pkill -f 'worker.py'"
echo ""
echo "Regional distribution:"
for region in "${!REGIONS[@]}"; do
    read -r start_id end_id <<< "${REGIONS[$region]}"
    echo "  $region: Workers $start_id-$end_id"
done
echo "=========================================="
