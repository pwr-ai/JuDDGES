# Monitor Extraction Throughput

This guide explains how to measure and monitor the processing rate during document extraction operations.

## Overview

Two tools are available for monitoring extraction throughput:

1. **`calculate_throughput.py`** - Analyzes extraction result files (simple, no database required)
2. **`monitor_extraction_throughput.py`** - Real-time monitoring from PostgreSQL logs (advanced)

## Method 1: Calculate Throughput from Result Files

### Basic Usage

Analyze all extraction results in a directory:

```bash
python scripts/extraction/calculate_throughput.py --directory data/extraction_results
```

Analyze a specific extraction file:

```bash
python scripts/extraction/calculate_throughput.py --file data/extraction_results/sample_documents_extracted.jsonl
```

### Estimate Completion Time

Calculate how long it will take to process remaining documents:

```bash
# Estimate time for 10,000 remaining documents with 95% expected success rate
python scripts/extraction/calculate_throughput.py \
  --directory data/extraction_results \
  --estimate 10000 \
  --success-rate 95.0
```

### Example Output

```
Extraction Throughput Analysis
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┓
┃ File                     ┃ Total   ┃ Success┃ Failed┃ Success   ┃ Duration   ┃ Docs/   ┃
┃                          ┃ Docs    ┃        ┃       ┃ Rate      ┃ (min)      ┃ Min     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━┩
│ sample_documents_...jsonl│  1,000  │  964   │  36   │  96.4%    │  18.5      │  54.05  │
└──────────────────────────┴─────────┴────────┴───────┴───────────┴────────────┴─────────┘

Average Statistics:
Average throughput: 54.05 docs/min

Completion Time Estimate
Remaining documents: 10,000
Current rate: 54.05 docs/min
Expected success rate: 95.0%

Estimated Time:
  • 195.1 minutes
  • 3.3 hours
  • 0.14 days

Expected completion: 2025-10-14 12:36:00
```

### Understanding the Results

- **Total Docs**: Number of documents processed
- **Success**: Documents successfully extracted
- **Failed**: Documents that failed extraction
- **Success Rate**: Percentage of successful extractions
- **Duration**: Time taken (minutes)
  - `✓` = Calculated from actual timestamps in results
  - `(estimated)` = Estimated based on document count
- **Docs/Min**: Processing rate (documents per minute)

### Typical Throughput Rates

Based on production runs:

| Model              | Typical Rate    | Notes                          |
|--------------------|-----------------|--------------------------------|
| gemini-2.5-flash   | 40-60 docs/min  | Fast, good for large batches   |
| gemini-2.5-pro     | 20-35 docs/min  | Slower, more accurate          |
| gemini-1.5-flash   | 50-70 docs/min  | Fastest, less detailed         |
| gemini-1.5-pro     | 15-25 docs/min  | Slowest, most comprehensive    |

**Note**: Actual rates depend on:
- Document length (longer texts = slower)
- Extraction schema complexity (more fields = slower)
- Batch size and parallelization settings
- Network latency to Vertex AI
- LLM cache hit rate

## Method 2: Real-Time Monitoring from PostgreSQL

### Prerequisites

Extraction storage must be configured with PostgreSQL connection details in `.env`:

```bash
# PostgreSQL for extraction logging
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=extraction_logs
```

### Basic Usage

Single snapshot of current throughput:

```bash
python scripts/extraction/monitor_extraction_throughput.py
```

Monitor specific extraction run:

```bash
python scripts/extraction/monitor_extraction_throughput.py --run-id <run_id>
```

### Live Monitoring

Real-time dashboard with auto-refresh:

```bash
# Refresh every 5 seconds (default)
python scripts/extraction/monitor_extraction_throughput.py --live

# Custom refresh interval (10 seconds)
python scripts/extraction/monitor_extraction_throughput.py --live --refresh 10
```

Press `Ctrl+C` to stop live monitoring.

### Example Live Dashboard

```
Extraction Throughput Monitor
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Current Throughput (Last 5 minutes)                                                      ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━┫
┃ Total Documents            ┃     250      ┃              ┃              ┃              ┃
┃ Successful                 ┃     242      ┃              ┃              ┃              ┃
┃ Failed                     ┃      8       ┃              ┃              ┃              ┃
┃ Rate                       ┃  50.0 docs/min              ┃              ┃              ┃
┃                            ┃              ┃              ┃              ┃              ┃
┃ Active Runs                                                                             ┃
┃ Run 3f8d2e1a...            ┃ gemini-2.5-flash ┃ Progress: 25% (250/1000) ┃ Elapsed: 5 min  ┃
┃                            ┃              ┃              ┃              ┃              ┃
┃ Recent Completed Runs                                                                   ┃
┃ Run a7b3c9d2...            ┃ gemini-2.5-pro ┃ Docs: 500 (98% success) ┃ Duration: 12 min ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┛
```

### Options

```bash
# Look back window for rate calculation (default: 5 minutes)
--minutes 10

# Number of recent completed runs to display (default: 5)
--recent 10

# Live monitoring with custom refresh
--live --refresh 3
```

## Optimizing Throughput

### Parallel Processing

Increase throughput by adjusting batch processing settings:

```bash
python scripts/extraction/run_extraction_rest.py \
  --max-documents 1000 \
  --batch-size 10 \
  --max-workers 10 \
  --model gemini-2.5-flash
```

**Guidelines**:
- **batch-size**: Number of documents per batch (5-20)
- **max-workers**: Parallel threads (5-10, max 10 to avoid rate limits)
- Higher parallelization = higher throughput but more API quota usage

### Cache Hit Rate

Enable LLM caching to speed up repeated extractions:

```bash
# Set in .env
POSTGRES_CACHE_URL=postgresql://user:password@localhost:5432/llm_cache
```

Cache hits can improve throughput by 5-10x for similar documents.

### Model Selection

Choose model based on needs:

```bash
# Fast extraction for large batches
--model gemini-2.5-flash

# Balanced accuracy and speed
--model gemini-2.5-pro

# Maximum accuracy (slower)
--model gemini-1.5-pro
```

### Document Filtering

Process only relevant documents to maximize effective throughput:

```bash
# Filter by document type
--document-type judgment

# Search for specific topic
--search-query "kredyt frankowy"

# Combine filters
--document-type tax_interpretation --search-query "VAT"
```

## Monitoring During Large Runs

For processing thousands of documents:

1. **Start extraction**:
   ```bash
   python scripts/extraction/run_extraction_rest.py \
     --max-documents 50000 \
     --batch-size 10 \
     --max-workers 10 \
     --model gemini-2.5-flash
   ```

2. **Monitor in separate terminal**:
   ```bash
   python scripts/extraction/monitor_extraction_throughput.py --live
   ```

3. **Check progress periodically**:
   ```bash
   python scripts/extraction/calculate_throughput.py \
     --directory data/extraction_results \
     --estimate 45000
   ```

## Troubleshooting

### Low Throughput

**Symptoms**: < 10 docs/min

**Possible causes**:
1. Low parallelization (increase `--max-workers`)
2. Rate limiting (reduce `--max-workers`, add delays)
3. Network latency (check connection to Vertex AI)
4. Complex extraction schema (simplify fields)
5. Very long documents (consider chunking)

### High Failure Rate

**Symptoms**: > 10% failed extractions

**Possible causes**:
1. Invalid documents (missing full_text)
2. Schema mismatch (check field definitions)
3. Token limits exceeded (reduce max_text_length)
4. API quota exceeded (reduce parallelization)

### No Throughput Data

**Symptoms**: `calculate_throughput.py` shows 0 docs/min

**Possible causes**:
1. No result files found (check directory path)
2. Empty result files (extraction failed)
3. Missing timestamps (use estimated duration)

## Related Documentation

- [Distributed Extraction](distributed-extraction.md) - Scale extraction across multiple workers
- [Monitor Extraction Errors](monitor-extraction-errors.md) - Track and debug failures
- [Optimize Weaviate Ingestion](optimize-weaviate-ingestion.md) - Speed up data ingestion

## See Also

- **Extraction Scripts**: `scripts/extraction/`
- **Batch Processor**: `juddges/extraction/batch_processor.py`
- **Statistics Module**: `juddges/extraction/statistics.py`
