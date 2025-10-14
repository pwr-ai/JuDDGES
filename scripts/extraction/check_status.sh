#!/bin/bash
# Check distributed extraction status
# Usage: ./scripts/extraction/check_status.sh [run_id]

set -e

RUN_ID=${1:-"6704c529-1a6b-46cd-a187-30dd97d3cddb"}  # Default to latest run
REDIS_PORT=${REDIS_PORT:-6381}

echo "=========================================="
echo "Distributed Extraction Status"
echo "=========================================="
echo "Run ID: $RUN_ID"
echo ""

# 1. Redis Queue Status
echo "[1] Redis Queue Status"
echo "---"
QUEUE_LENGTH=$(redis-cli -p $REDIS_PORT LLEN extraction_queue 2>/dev/null || echo "N/A")
echo "  Jobs remaining: $QUEUE_LENGTH"
echo ""

# 2. Worker Status
echo "[2] Worker Processes"
echo "---"
WORKER_COUNT=$(ps aux | grep "python.*worker.py" | grep -v grep | wc -l)
echo "  Active workers: $WORKER_COUNT"
if [ $WORKER_COUNT -gt 0 ]; then
    echo "  PIDs: $(ps aux | grep "python.*worker.py" | grep -v grep | awk '{print $2}' | tr '\n' ' ')"
fi
echo ""

# 3. Extraction Progress (from logs)
echo "[3] Extraction Progress (from worker logs)"
echo "---"
if [ -d "logs" ]; then
    SUCCESSFUL=$(grep "✓ Extracted" logs/worker_*.log 2>/dev/null | wc -l || echo "0")
    FAILED=$(grep "Failed.*:" logs/worker_*.log 2>/dev/null | wc -l || echo "0")
    TOTAL=$((SUCCESSFUL + FAILED))

    echo "  Successful extractions: $SUCCESSFUL"
    echo "  Failed extractions: $FAILED"
    echo "  Total processed: $TOTAL"

    if [ $TOTAL -gt 0 ]; then
        SUCCESS_RATE=$(awk "BEGIN {printf \"%.1f\", ($SUCCESSFUL/$TOTAL)*100}")
        echo "  Success rate: ${SUCCESS_RATE}%"
    fi
else
    echo "  No logs directory found"
fi
echo ""

# 4. Recent Errors
echo "[4] Recent Errors (last 5)"
echo "---"
if [ -d "logs" ]; then
    grep "ERROR" logs/worker_*.log 2>/dev/null | tail -5 || echo "  No errors found"
else
    echo "  No logs directory found"
fi
echo ""

# 5. Database Status (if PostgreSQL is available)
echo "[5] Database Status"
echo "---"
DB_URL="postgresql://extraction_user:extraction_pass@localhost:5434/legal_extraction"

# Try to query database
if command -v psql &> /dev/null; then
    QUERY="SELECT extraction_status, COUNT(*) as count FROM extraction_results WHERE run_id = '$RUN_ID' GROUP BY extraction_status;"

    psql "$DB_URL" -t -c "$QUERY" 2>/dev/null | grep -v "^$" || echo "  Database not accessible or no results yet"
else
    echo "  psql not available - install postgresql-client to check database"
fi
echo ""

# 6. Performance Stats
echo "[6] Performance Statistics"
echo "---"
if [ -d "logs" ]; then
    # Get first and last extraction timestamp
    FIRST_TS=$(grep "✓ Extracted" logs/worker_*.log 2>/dev/null | head -1 | awk '{print $1, $2}' || echo "")
    LAST_TS=$(grep "✓ Extracted" logs/worker_*.log 2>/dev/null | tail -1 | awk '{print $1, $2}' || echo "")

    if [ -n "$FIRST_TS" ] && [ -n "$LAST_TS" ]; then
        echo "  First extraction: $FIRST_TS"
        echo "  Last extraction: $LAST_TS"

        # Calculate duration (approximate - would need better date parsing)
        echo "  Duration: Check logs for exact timing"

        if [ $SUCCESSFUL -gt 0 ]; then
            echo "  Throughput: ~$(awk "BEGIN {printf \"%.1f\", $SUCCESSFUL/7}") docs/min (estimated)"
        fi
    else
        echo "  No completed extractions yet"
    fi
else
    echo "  No logs available"
fi
echo ""

echo "=========================================="
echo "Monitor Commands"
echo "=========================================="
echo "Watch queue:    watch -n 5 'redis-cli -p $REDIS_PORT LLEN extraction_queue'"
echo "Follow worker:  tail -f logs/worker_1.log"
echo "Count progress: watch -n 5 'grep \"✓ Extracted\" logs/worker_*.log | wc -l'"
echo "Check errors:   grep ERROR logs/worker_*.log"
echo ""
