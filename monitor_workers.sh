#!/bin/bash
# Monitor extraction workers
# Usage: ./monitor_workers.sh

echo "=================================="
echo "Extraction Workers Status"
echo "=================================="
echo ""

# Count running workers
WORKER_COUNT=$(ps aux | grep 'worker.py' | grep -v grep | wc -l)
echo "Running workers: $WORKER_COUNT"
echo ""

# Show worker processes
echo "Worker processes:"
ps aux | grep 'worker.py' | grep -v grep | awk '{print "  Worker", $12, "- PID:", $2, "- CPU:", $3"%", "- MEM:", $4"%"}'
echo ""

# Show recent logs from each worker
echo "Recent activity (last 3 lines per worker):"
for i in {1..10}; do
    LOG_FILE="logs/worker_$i.log"
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "Worker $i:"
        tail -n 3 "$LOG_FILE" | sed 's/^/  /'
    fi
done

echo ""
echo "=================================="
echo "For live monitoring, run:"
echo "  tail -f logs/worker_*.log"
echo ""
echo "To stop all workers:"
echo "  pkill -f 'worker.py'"
echo "=================================="
