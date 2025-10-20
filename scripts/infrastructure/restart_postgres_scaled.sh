#!/bin/bash
# Restart PostgreSQL LLM Cache with Scaled Settings
#
# This script restarts the llm-postgres container with increased connection limits
# to support 45+ concurrent workers.
#
# Usage:
#   ./restart_postgres_scaled.sh

set -e

CONTAINER_NAME="llm-postgres"

echo "=========================================="
echo "Restarting PostgreSQL with Scaled Settings"
echo "=========================================="
echo ""

# Configuration for 45 workers
# Each worker: ~5 connections (cache + threads + storage)
# 45 × 5 = 225 + 75 overhead = 300 connections
MAX_CONNECTIONS=300
SHARED_BUFFERS="512MB"

echo "New settings:"
echo "  max_connections: $MAX_CONNECTIONS"
echo "  shared_buffers: $SHARED_BUFFERS"
echo ""

# Stop and remove existing container
echo "1. Stopping existing container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "2. Starting new container with scaled settings..."
docker run -d \
  --name $CONTAINER_NAME \
  --restart unless-stopped \
  -p 5555:5432 \
  -e POSTGRES_USER=llm_cache \
  -e POSTGRES_PASSWORD=xNnseZW5SjjU5j7DKGyW_2oNFRsq1vdBGpgjwzsqB-w \
  -e POSTGRES_DB=llm_cache \
  postgres:15.2 \
  postgres \
    -c max_connections=$MAX_CONNECTIONS \
    -c shared_buffers=$SHARED_BUFFERS \
    -c effective_cache_size=1GB \
    -c maintenance_work_mem=128MB \
    -c checkpoint_completion_target=0.9 \
    -c wal_buffers=16MB \
    -c default_statistics_target=100 \
    -c random_page_cost=1.1 \
    -c effective_io_concurrency=200

echo ""
echo "3. Waiting for PostgreSQL to start..."
sleep 5

# Verify settings
echo ""
echo "4. Verifying settings..."
MAX_CONN=$(docker exec $CONTAINER_NAME psql -U llm_cache -d llm_cache -t -c "SHOW max_connections;" | xargs)
SHARED_BUF=$(docker exec $CONTAINER_NAME psql -U llm_cache -d llm_cache -t -c "SHOW shared_buffers;" | xargs)

echo "  max_connections: $MAX_CONN"
echo "  shared_buffers: $SHARED_BUF"
echo ""

if [ "$MAX_CONN" -eq "$MAX_CONNECTIONS" ]; then
    echo "✓ PostgreSQL successfully scaled!"
    echo ""
    echo "Connection capacity:"
    echo "  - 45 workers × 5 connections/worker = 225 connections"
    echo "  - Overhead + future growth = 75 connections"
    echo "  - Total available: $MAX_CONN connections"
    echo ""
    echo "You can now restart your 45 workers:"
    echo "  ./start_25_workers_multi_region.sh"
else
    echo "✗ Warning: Max connections is $MAX_CONN (expected $MAX_CONNECTIONS)"
fi

echo "=========================================="
