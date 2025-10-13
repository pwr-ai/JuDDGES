-- ============================================================================
-- Legal AI Extraction Database Schema
-- ============================================================================
-- This database stores all extraction inputs, outputs, and metadata for:
-- - Parallel execution safety (ACID transactions)
-- - Full audit trail (query parameters, prompts, timestamps)
-- - Easy analysis and review
-- - Ingestion support with complete traceability
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- EXTRACTION RUNS TABLE
-- Stores metadata about each extraction run
-- ============================================================================
CREATE TABLE IF NOT EXISTS extraction_runs (
    -- Primary key
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Query parameters
    search_query TEXT,                  -- Optional search query (e.g., "kredyt frankowy", "VAT")
    document_type_filter TEXT,          -- Optional document type filter ("judgment", "tax_interpretation")

    -- Model configuration
    model_name TEXT NOT NULL,           -- Gemini model used (e.g., "gemini-2.5-pro")
    vertex_project TEXT,                -- GCP project ID
    vertex_location TEXT,               -- GCP location
    temperature FLOAT,                  -- Model temperature

    -- Prompt configuration
    prompt_template TEXT,               -- Full prompt template used for extraction
    extraction_schema JSONB,            -- Complete extraction schema definition

    -- Execution parameters
    sample_size INTEGER NOT NULL,       -- Number of documents sampled
    batch_size INTEGER NOT NULL,        -- Batch size for processing
    max_workers INTEGER NOT NULL,       -- Number of parallel workers

    -- Weaviate connection
    weaviate_host TEXT NOT NULL,
    weaviate_port INTEGER NOT NULL,

    -- Results summary
    total_documents INTEGER,            -- Total documents processed
    successful_extractions INTEGER,     -- Count of successful extractions
    failed_extractions INTEGER,         -- Count of failed extractions

    -- Execution metadata
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    duration_seconds FLOAT,

    -- Random seed for reproducibility
    random_seed INTEGER,

    -- User/context
    executed_by TEXT DEFAULT CURRENT_USER,
    notes TEXT                          -- Optional notes about this run
);

-- ============================================================================
-- EXTRACTION RESULTS TABLE
-- Stores individual document extraction results with full inputs and outputs
-- ============================================================================
CREATE TABLE IF NOT EXISTS extraction_results (
    -- Primary key
    id BIGSERIAL PRIMARY KEY,

    -- Foreign key to extraction run
    run_id UUID NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE,

    -- Document identifiers (INPUTS)
    document_id TEXT NOT NULL,          -- Weaviate UUID (e.g., "/doc/C7D6AAF0BD")
    document_number TEXT,               -- Human-readable case number (e.g., "I ACa 123/23")
    document_type TEXT NOT NULL,        -- Document type from Weaviate

    -- Document content (INPUT)
    full_text TEXT NOT NULL,            -- Complete document text
    full_text_length INTEGER,           -- Length in characters
    source_language TEXT,               -- Source language

    -- Extraction output
    extraction_status TEXT NOT NULL CHECK (extraction_status IN ('success', 'failed', 'skipped')),
    extracted_data JSONB,               -- Complete extracted data as JSONB
    error_message TEXT,                 -- Error message if failed
    error_type TEXT,                    -- Error type/class if failed

    -- Processing metadata
    extracted_at TIMESTAMP NOT NULL DEFAULT NOW(),
    processing_time_seconds FLOAT,      -- Time taken for this document

    -- Prevent duplicate extractions within a run
    UNIQUE(run_id, document_id)
);

-- ============================================================================
-- INGESTION TRACKING TABLE
-- Tracks which extraction results have been ingested back to Weaviate
-- ============================================================================
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id BIGSERIAL PRIMARY KEY,

    -- Link to extraction run
    run_id UUID NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE,

    -- Ingestion metadata
    ingestion_started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    ingestion_completed_at TIMESTAMP,

    -- Ingestion parameters
    batch_size INTEGER NOT NULL,
    overwrite_existing BOOLEAN NOT NULL DEFAULT FALSE,

    -- Results
    total_documents INTEGER,
    successful_updates INTEGER,
    failed_updates INTEGER,
    skipped_documents INTEGER,
    duration_seconds FLOAT,

    -- Errors
    errors JSONB,                       -- Array of error objects

    -- Status
    status TEXT CHECK (status IN ('running', 'completed', 'failed'))
);

-- ============================================================================
-- FIELD COVERAGE TABLE
-- Tracks which fields were successfully extracted across runs
-- ============================================================================
CREATE TABLE IF NOT EXISTS field_coverage (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES extraction_runs(run_id) ON DELETE CASCADE,

    field_name TEXT NOT NULL,
    populated_count INTEGER NOT NULL DEFAULT 0,
    empty_count INTEGER NOT NULL DEFAULT 0,
    coverage_percentage FLOAT GENERATED ALWAYS AS (
        CASE
            WHEN (populated_count + empty_count) > 0
            THEN (populated_count::FLOAT / (populated_count + empty_count) * 100)
            ELSE 0
        END
    ) STORED,

    UNIQUE(run_id, field_name)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Extraction runs indexes
CREATE INDEX idx_extraction_runs_search_query ON extraction_runs(search_query) WHERE search_query IS NOT NULL;
CREATE INDEX idx_extraction_runs_document_type ON extraction_runs(document_type_filter) WHERE document_type_filter IS NOT NULL;
CREATE INDEX idx_extraction_runs_model ON extraction_runs(model_name);
CREATE INDEX idx_extraction_runs_started_at ON extraction_runs(started_at DESC);
CREATE INDEX idx_extraction_runs_status ON extraction_runs(started_at, completed_at) WHERE completed_at IS NULL;

-- Extraction results indexes
CREATE INDEX idx_extraction_results_run_id ON extraction_results(run_id);
CREATE INDEX idx_extraction_results_document_id ON extraction_results(document_id);
CREATE INDEX idx_extraction_results_document_number ON extraction_results(document_number) WHERE document_number IS NOT NULL;
CREATE INDEX idx_extraction_results_status ON extraction_results(extraction_status);
CREATE INDEX idx_extraction_results_extracted_at ON extraction_results(extracted_at DESC);

-- GIN index for JSONB fields (fast JSON queries)
CREATE INDEX idx_extraction_results_extracted_data ON extraction_results USING GIN(extracted_data);
CREATE INDEX idx_extraction_runs_schema ON extraction_runs USING GIN(extraction_schema);

-- Ingestion logs indexes
CREATE INDEX idx_ingestion_logs_run_id ON ingestion_logs(run_id);
CREATE INDEX idx_ingestion_logs_started_at ON ingestion_logs(ingestion_started_at DESC);

-- Field coverage indexes
CREATE INDEX idx_field_coverage_run_id ON field_coverage(run_id);
CREATE INDEX idx_field_coverage_field_name ON field_coverage(field_name);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Complete extraction run summary
CREATE OR REPLACE VIEW v_extraction_run_summary AS
SELECT
    er.run_id,
    er.search_query,
    er.document_type_filter,
    er.model_name,
    er.sample_size,
    er.total_documents,
    er.successful_extractions,
    er.failed_extractions,
    ROUND((er.successful_extractions::FLOAT / NULLIF(er.total_documents, 0) * 100)::NUMERIC, 2) as success_rate_pct,
    er.started_at,
    er.completed_at,
    er.duration_seconds,
    COUNT(DISTINCT res.document_id) as unique_documents_extracted,
    COUNT(DISTINCT CASE WHEN res.extraction_status = 'success' THEN res.id END) as successful_count,
    COUNT(DISTINCT CASE WHEN res.extraction_status = 'failed' THEN res.id END) as failed_count
FROM extraction_runs er
LEFT JOIN extraction_results res ON er.run_id = res.run_id
GROUP BY er.run_id;

-- View: Latest extractions by document
CREATE OR REPLACE VIEW v_latest_extraction_by_document AS
SELECT DISTINCT ON (document_id)
    document_id,
    document_number,
    document_type,
    run_id,
    extraction_status,
    extracted_data,
    extracted_at
FROM extraction_results
ORDER BY document_id, extracted_at DESC;

-- View: Extraction quality metrics
CREATE OR REPLACE VIEW v_extraction_quality_metrics AS
SELECT
    er.run_id,
    er.model_name,
    er.search_query,
    AVG(fc.coverage_percentage) as avg_field_coverage,
    COUNT(DISTINCT fc.field_name) as total_fields_tracked,
    COUNT(DISTINCT CASE WHEN fc.coverage_percentage > 80 THEN fc.field_name END) as high_coverage_fields,
    COUNT(DISTINCT CASE WHEN fc.coverage_percentage < 20 THEN fc.field_name END) as low_coverage_fields
FROM extraction_runs er
LEFT JOIN field_coverage fc ON er.run_id = fc.run_id
GROUP BY er.run_id, er.model_name, er.search_query;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function: Get extraction statistics for a run
CREATE OR REPLACE FUNCTION get_extraction_stats(p_run_id UUID)
RETURNS TABLE(
    metric TEXT,
    value NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'total_documents'::TEXT, COUNT(*)::NUMERIC FROM extraction_results WHERE run_id = p_run_id
    UNION ALL
    SELECT 'successful'::TEXT, COUNT(*)::NUMERIC FROM extraction_results WHERE run_id = p_run_id AND extraction_status = 'success'
    UNION ALL
    SELECT 'failed'::TEXT, COUNT(*)::NUMERIC FROM extraction_results WHERE run_id = p_run_id AND extraction_status = 'failed'
    UNION ALL
    SELECT 'avg_text_length'::TEXT, AVG(full_text_length)::NUMERIC FROM extraction_results WHERE run_id = p_run_id
    UNION ALL
    SELECT 'avg_processing_time'::TEXT, AVG(processing_time_seconds)::NUMERIC FROM extraction_results WHERE run_id = p_run_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SAMPLE QUERIES (for documentation)
-- ============================================================================

-- Find all extractions for a specific query
-- SELECT * FROM extraction_results WHERE run_id IN (
--     SELECT run_id FROM extraction_runs WHERE search_query = 'kredyt frankowy'
-- );

-- Get the latest successful extraction for each document
-- SELECT * FROM v_latest_extraction_by_document WHERE extraction_status = 'success';

-- Find documents that failed extraction
-- SELECT document_id, document_number, error_message
-- FROM extraction_results
-- WHERE extraction_status = 'failed'
-- ORDER BY extracted_at DESC;

-- Get extraction quality metrics
-- SELECT * FROM v_extraction_quality_metrics ORDER BY avg_field_coverage DESC;

-- Export extraction results for ingestion
-- SELECT document_id, document_number, extracted_data
-- FROM extraction_results
-- WHERE run_id = 'your-run-id' AND extraction_status = 'success';

-- ============================================================================
-- GRANTS (adjust based on your user setup)
-- ============================================================================

-- Grant permissions to extraction_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO extraction_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO extraction_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO extraction_user;

-- Default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO extraction_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO extraction_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO extraction_user;
