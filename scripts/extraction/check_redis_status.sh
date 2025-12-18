#!/bin/bash
# Check Redis status with authentication

# Load Redis password from .env
if [ -f .env ]; then
    export $(grep REDIS_PASSWORD .env | xargs)
fi

REDIS_HOST=${REDIS_HOST:-localhost}
REDIS_PORT=${REDIS_PORT:-6381}
QUEUE_NAME=${1:-extraction_queue}

echo "========================================="
echo "Redis Queue Status"
echo "========================================="
echo "Host: $REDIS_HOST:$REDIS_PORT"
echo "Queue: $QUEUE_NAME"
echo ""

# Check queue length
QUEUE_LEN=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a "$REDIS_PASSWORD" --no-auth-warning llen $QUEUE_NAME 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "Queue Length: $QUEUE_LEN jobs"
else
    echo "ERROR: Cannot connect to Redis"
    exit 1
fi

# Check Redis stats
echo ""
echo "Redis Statistics:"
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a "$REDIS_PASSWORD" --no-auth-warning info stats 2>/dev/null | grep -E "total_commands_processed|instantaneous_ops_per_sec|total_connections_received"

# Check memory
echo ""
echo "Memory Usage:"
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a "$REDIS_PASSWORD" --no-auth-warning info memory 2>/dev/null | grep -E "used_memory_human|used_memory_peak_human"

# Estimate consumption rate
echo ""
echo "Monitoring queue for 10 seconds..."
INITIAL_LEN=$QUEUE_LEN
sleep 10
FINAL_LEN=$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a "$REDIS_PASSWORD" --no-auth-warning llen $QUEUE_NAME 2>/dev/null)

CONSUMED=$((INITIAL_LEN - FINAL_LEN))
RATE_PER_SEC=$(echo "scale=2; $CONSUMED / 10" | bc)
RATE_PER_MIN=$(echo "scale=2; $RATE_PER_SEC * 60" | bc)

echo ""
echo "========================================="
echo "Queue Consumption Rate:"
echo "  Initial: $INITIAL_LEN"
echo "  Final:   $FINAL_LEN"
echo "  Consumed: $CONSUMED jobs in 10 seconds"
echo "  Rate: $RATE_PER_SEC jobs/sec ($RATE_PER_MIN jobs/min)"
echo "========================================="

# Estimate completion
if [ "$RATE_PER_MIN" != "0" ] && [ "$RATE_PER_MIN" != "0.00" ]; then
    MINUTES=$(echo "scale=1; $FINAL_LEN / $RATE_PER_MIN" | bc)
    HOURS=$(echo "scale=2; $MINUTES / 60" | bc)
    echo ""
    echo "Estimated completion time:"
    echo "  Minutes: $MINUTES"
    echo "  Hours: $HOURS"
fi
