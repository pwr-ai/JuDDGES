#!/usr/bin/env python3
"""Setup LangChain PostgreSQL cache with proper schema.

This script ensures the cache table has the correct indexes to handle
large LLM configuration objects without hitting PostgreSQL btree limits.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

from juddges.settings import ROOT_PATH

# Load environment variables
load_dotenv(ROOT_PATH / ".env", override=True)


def setup_cache_schema(postgres_url: str):
    """Setup cache schema with proper indexes.

    Args:
        postgres_url: PostgreSQL connection URL
    """
    engine = create_engine(postgres_url)

    logger.info(f"Connecting to PostgreSQL cache at {postgres_url}...")

    with engine.connect() as conn:
        # Check if table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'full_md5_llm_cache'
            );
        """))
        table_exists = result.scalar()

        if not table_exists:
            logger.info("Cache table doesn't exist yet - will be created automatically by LangChain")
            return

        # Check if problematic btree index exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM pg_indexes
                WHERE schemaname = 'public'
                AND tablename = 'full_md5_llm_cache'
                AND indexname = 'ix_full_md5_llm_cache_llm'
            );
        """))
        btree_index_exists = result.scalar()

        if btree_index_exists:
            logger.warning("Found problematic btree index on llm column - fixing...")

            # Drop problematic btree index
            conn.execute(text("DROP INDEX IF EXISTS ix_full_md5_llm_cache_llm;"))
            logger.info("✓ Dropped btree index on llm column")

            # Create hash index instead
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_full_md5_llm_cache_llm_hash
                ON full_md5_llm_cache USING hash (llm);
            """))
            logger.info("✓ Created hash index on llm column")

            conn.commit()
            logger.info("✓ Cache schema fixed successfully")
        else:
            logger.info("Cache schema is already correct - no fix needed")


def main():
    """Main function."""
    postgres_cache_url = os.getenv("POSTGRES_CACHE_URL")

    if not postgres_cache_url:
        logger.error("POSTGRES_CACHE_URL environment variable not set")
        return 1

    try:
        setup_cache_schema(postgres_cache_url)
        logger.info("Cache setup completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Failed to setup cache schema: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
