#!/bin/bash

# Create external Docker volumes for development and production if they don't exist

VOLUMES=("legal_ai_weaviate_dev" "legal_ai_weaviate_prod")

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

for VOLUME in "${VOLUMES[@]}"; do
    if ! docker volume ls --format '{{.Name}}' | grep -q "^${VOLUME}$"; then
        docker volume create "${VOLUME}"
        echo -e "${GREEN}Created volume: ${VOLUME}${NC}"
    else
        echo -e "${YELLOW}Volume already exists: ${VOLUME}${NC}"
    fi
done

echo -e "${GREEN}Volumes are ready. Do you want to run:${NC} docker compose up -d ? [y/N]"
read -r answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    docker compose up -d
fi