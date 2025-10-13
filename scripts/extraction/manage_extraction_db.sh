#!/bin/bash
# Extraction Database Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEAVIATE_DIR="$PROJECT_ROOT/weaviate"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Database connection details
DB_HOST="${EXTRACTION_POSTGRES_HOST:-localhost}"
DB_PORT="${EXTRACTION_POSTGRES_PORT:-5434}"
DB_USER="${EXTRACTION_POSTGRES_USER:-extraction_user}"
DB_PASSWORD="${EXTRACTION_POSTGRES_PASSWORD:-extraction_pass}"
DB_NAME="${EXTRACTION_POSTGRES_DB:-legal_extraction}"

# Functions
function print_help() {
    echo "Extraction Database Management"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start       - Start the extraction postgres container"
    echo "  stop        - Stop the extraction postgres container"
    echo "  restart     - Restart the extraction postgres container"
    echo "  status      - Check container status"
    echo "  logs        - Show container logs"
    echo "  connect     - Connect to database with psql"
    echo "  backup      - Backup database to file"
    echo "  restore     - Restore database from backup file"
    echo "  stats       - Show database statistics"
    echo "  cleanup     - Clean up old extraction runs (>30 days)"
    echo "  help        - Show this help message"
}

function start_db() {
    echo -e "${GREEN}Starting extraction postgres...${NC}"
    cd "$WEAVIATE_DIR"
    docker compose up -d extraction-postgres
    echo -e "${GREEN}✓ Extraction postgres started on port $DB_PORT${NC}"
}

function stop_db() {
    echo -e "${YELLOW}Stopping extraction postgres...${NC}"
    cd "$WEAVIATE_DIR"
    docker compose stop extraction-postgres
    echo -e "${GREEN}✓ Extraction postgres stopped${NC}"
}

function restart_db() {
    echo -e "${YELLOW}Restarting extraction postgres...${NC}"
    cd "$WEAVIATE_DIR"
    docker compose restart extraction-postgres
    echo -e "${GREEN}✓ Extraction postgres restarted${NC}"
}

function check_status() {
    echo -e "${GREEN}Checking extraction postgres status...${NC}"
    docker ps --filter "name=legal_ai_extraction_postgres" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

function show_logs() {
    echo -e "${GREEN}Showing extraction postgres logs...${NC}"
    docker logs -f legal_ai_extraction_postgres
}

function connect_db() {
    echo -e "${GREEN}Connecting to extraction database...${NC}"
    docker exec -it legal_ai_extraction_postgres psql -U "$DB_USER" -d "$DB_NAME"
}

function backup_db() {
    BACKUP_FILE="${1:-extraction_backup_$(date +%Y%m%d_%H%M%S).sql}"
    echo -e "${GREEN}Backing up database to $BACKUP_FILE...${NC}"
    docker exec legal_ai_extraction_postgres pg_dump -U "$DB_USER" "$DB_NAME" > "$BACKUP_FILE"
    echo -e "${GREEN}✓ Backup saved to $BACKUP_FILE${NC}"
}

function restore_db() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Please provide backup file path${NC}"
        echo "Usage: $0 restore <backup_file>"
        exit 1
    fi

    echo -e "${YELLOW}Restoring database from $1...${NC}"
    cat "$1" | docker exec -i legal_ai_extraction_postgres psql -U "$DB_USER" "$DB_NAME"
    echo -e "${GREEN}✓ Database restored from $1${NC}"
}

function show_stats() {
    echo -e "${GREEN}Database Statistics:${NC}"
    docker exec -it legal_ai_extraction_postgres psql -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 'Database Size' as metric, pg_size_pretty(pg_database_size('$DB_NAME')) as value
        UNION ALL
        SELECT 'Extraction Runs', COUNT(*)::text FROM extraction_runs
        UNION ALL
        SELECT 'Extraction Results', COUNT(*)::text FROM extraction_results
        UNION ALL
        SELECT 'Successful Extractions', COUNT(*)::text FROM extraction_results WHERE extraction_status = 'success'
        UNION ALL
        SELECT 'Failed Extractions', COUNT(*)::text FROM extraction_results WHERE extraction_status = 'failed'
        UNION ALL
        SELECT 'Ingestion Logs', COUNT(*)::text FROM ingestion_logs;
    "

    echo ""
    echo -e "${GREEN}Recent Extraction Runs:${NC}"
    docker exec -it legal_ai_extraction_postgres psql -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT
            run_id,
            search_query,
            model_name,
            sample_size,
            successful_extractions,
            failed_extractions,
            started_at
        FROM extraction_runs
        ORDER BY started_at DESC
        LIMIT 5;
    "
}

function cleanup_old_runs() {
    echo -e "${YELLOW}Cleaning up extraction runs older than 30 days...${NC}"
    read -p "Are you sure? This will delete old runs and their results. [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker exec -it legal_ai_extraction_postgres psql -U "$DB_USER" -d "$DB_NAME" -c "
            DELETE FROM extraction_runs
            WHERE started_at < NOW() - INTERVAL '30 days';
        "
        echo -e "${GREEN}✓ Cleanup completed${NC}"
    else
        echo -e "${YELLOW}Cleanup cancelled${NC}"
    fi
}

# Main script
case "${1:-help}" in
    start)
        start_db
        ;;
    stop)
        stop_db
        ;;
    restart)
        restart_db
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    connect)
        connect_db
        ;;
    backup)
        backup_db "$2"
        ;;
    restore)
        restore_db "$2"
        ;;
    stats)
        show_stats
        ;;
    cleanup)
        cleanup_old_runs
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        print_help
        exit 1
        ;;
esac
