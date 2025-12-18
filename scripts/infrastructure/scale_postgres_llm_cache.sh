#!/bin/bash
# Scale PostgreSQL LLM Cache for High-Concurrency Workloads
#
# This script increases PostgreSQL connection limits and memory settings
# to support 45+ concurrent workers with LangChain caching.
#
# Usage:
#   ./scale_postgres_llm_cache.sh

set -e

CONTAINER_NAME="llm-postgres"
POSTGRES_USER="llm_cache"

echo "=========================================="
echo "PostgreSQL LLM Cache Scaling"
echo "=========================================="
echo ""

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "Error: Container $CONTAINER_NAME is not running"
    exit 1
fi

echo "1. Checking current configuration..."
echo ""

# Get current settings
MAX_CONN=$(docker exec $CONTAINER_NAME psql -U $POSTGRES_USER -t -c "SHOW max_connections;" | xargs)
SHARED_BUF=$(docker exec $CONTAINER_NAME psql -U $POSTGRES_USER -t -c "SHOW shared_buffers;" | xargs)
ACTIVE_CONN=$(docker exec $CONTAINER_NAME psql -U $POSTGRES_USER -t -c "SELECT count(*) FROM pg_stat_activity;" | xargs)

echo "  Current max_connections: $MAX_CONN"
echo "  Current shared_buffers: $SHARED_BUF"
echo "  Active connections: $ACTIVE_CONN"
echo ""

# Calculate recommended settings for 45 workers
# Each worker needs:
# - 1 connection for LangChain cache (worker init)
# - 3 connections for extraction threads (max_extraction_threads=3)
# - 1 connection for storage
# Total: ~5 connections per worker
# 45 workers × 5 = 225 connections
# Add 25 for overhead = 250 total
# Set to 300 for safety margin

RECOMMENDED_MAX_CONN=300
RECOMMENDED_SHARED_BUF="512MB"  # 2× current for better caching

echo "2. Recommended settings for 45 workers:"
echo "  max_connections: $RECOMMENDED_MAX_CONN (current: $MAX_CONN)"
echo "  shared_buffers: $RECOMMENDED_SHARED_BUF (current: $SHARED_BUF)"
echo ""

# Prompt for confirmation
read -p "Apply these settings? This will restart PostgreSQL container. (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "3. Applying new settings..."
echo ""

# Stop container
echo "  Stopping container..."
docker stop $CONTAINER_NAME

# Update postgres.conf inside container volumes
# We need to start container temporarily with custom command to modify config
echo "  Starting container with updated settings..."

# Get container config
IMAGE=$(docker inspect $CONTAINER_NAME | grep -m1 "Image" | cut -d'"' -f4)
ENV_VARS=$(docker inspect $CONTAINER_NAME | grep -A 100 "Env" | grep "POSTGRES" | sed 's/.*"\(.*\)".*/\1/' | sed 's/^/-e /')
PORTS=$(docker inspect $CONTAINER_NAME | grep "HostPort" | grep -v "\"\"" | cut -d'"' -f4 | head -1)

# Start with custom postgres config
docker run -d \
  --name $CONTAINER_NAME \
  -p ${PORTS}:5432 \
  $(echo $ENV_VARS | tr '\n' ' ') \
  -c max_connections=$RECOMMENDED_MAX_CONN \
  -c shared_buffers=$RECOMMENDED_SHARED_BUF \
  $IMAGE \
  postgres -c max_connections=$RECOMMENDED_MAX_CONN -c shared_buffers=$RECOMMENDED_SHARED_BUF

echo ""
echo "4. Waiting for PostgreSQL to start..."
sleep 5

# Verify settings
echo ""
echo "5. Verifying new settings..."
NEW_MAX_CONN=$(docker exec $CONTAINER_NAME psql -U $POSTGRES_USER -t -c "SHOW max_connections;" | xargs)
NEW_SHARED_BUF=$(docker exec $CONTAINER_NAME psql -U $POSTGRES_USER -t -c "SHOW shared_buffers;" | xargs)

echo "  max_connections: $NEW_MAX_CONN"
echo "  shared_buffers: $NEW_SHARED_BUF"
echo ""

if [ "$NEW_MAX_CONN" -eq "$RECOMMENDED_MAX_CONN" ]; then
    echo "✓ PostgreSQL successfully scaled!"
    echo ""
    echo "Connection capacity:"
    echo "  - 45 workers × 5 connections = 225 connections"
    echo "  - Overhead + future growth = 75 connections"
    echo "  - Total available: $NEW_MAX_CONN connections"
    echo ""
    echo "To make this permanent, update your docker-compose or Docker run command with:"
    echo "  command: postgres -c max_connections=$RECOMMENDED_MAX_CONN -c shared_buffers=$RECOMMENDED_SHARED_BUF"
else
    echo "✗ Failed to apply settings. Max connections is still $NEW_MAX_CONN"
    exit 1
fi

echo "=========================================="
