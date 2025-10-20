-- DEPRECATED: This script is no longer needed with SQLAlchemyMd5Cache
-- The SQLAlchemyMd5Cache stores MD5 hashes instead of full text, avoiding index size limits
-- Kept for reference only

-- Legacy Fix: LangChain cache schema to handle large LLM configurations
-- Drop the problematic btree index on the llm column
DROP INDEX IF EXISTS ix_full_md5_llm_cache_llm;

-- Create a hash index instead (supports larger values)
-- Hash indexes are good for equality comparisons but not range queries
CREATE INDEX IF NOT EXISTS ix_full_md5_llm_cache_llm_hash
ON full_md5_llm_cache USING hash (llm);

-- Note: With SQLAlchemyMd5Cache, use table 'llm_cache_full_md5' instead
-- No manual schema fixes needed
