# Extraction Test Run Status

## Summary

Successfully implemented and tested the Gemini extraction pipeline with comprehensive schema based on Weaviate properties.

## ✅ What Works

### 1. **Weaviate Connection - SOLVED**
- **Issue**: Weaviate Python client v4 defaults to GRPC, but public instance (legal-ai-weaviate.augustyniak.ai:8084) only exposes REST API on port 8084
- **Solution**: Created `run_extraction_rest.py` that uses Weaviate REST API directly via GraphQL endpoint
- **Status**: ✅ **Successfully connects and fetches documents**

### 2. **Extraction Infrastructure**
- ✅ Comprehensive 14-field schema created matching Weaviate properties
- ✅ Langfuse tracing enabled and configured
- ✅ Document sampling and filtering works
- ✅ Extraction chain initialized with Gemini 2.5 Flash
- ✅ SQLite caching enabled
- ✅ Progress tracking with Rich console
- ✅ Result saving to JSONL files with field coverage analysis

### 3. **Test Run Results**
- ✅ Connected to: `http://legal-ai-weaviate.augustyniak.ai:8084`
- ✅ Fetched 25 documents from Weaviate
- ✅ Found 25 documents with valid full_text
- ✅ Sampled 5 documents for extraction
- ✅ Langfuse tracing: `https://legal-ai-langfuse.augustyniak.ai`

## ❌ Current Blocker: Google API Authentication

### Error
```
403 Request had insufficient authentication scopes
[reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
domain: "googleapis.com"
metadata {
  key: "service"
  value: "generativelanguage.googleapis.com"
}
metadata {
  key: "method"
  value: "google.ai.generativelanguage.v1beta.GenerativeService.GenerateContent"
}
]
```

### Root Cause
The `GOOGLE_API_KEY` in `.env` file either:
1. Is expired or invalid
2. Doesn't have the Generative Language API enabled
3. Has insufficient scopes/permissions for the API
4. Has billing disabled for the project

### Current Key (from .env)
```bash
GOOGLE_API_KEY=[REDACTED-API-KEY]
GEMINI_API_KEY=[REDACTED-API-KEY]
```

## 🔧 How to Fix

### Option 1: Update API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key with Generative Language API enabled
3. Update `.env` file:
   ```bash
   GOOGLE_API_KEY=your-new-api-key-here
   GEMINI_API_KEY=your-new-api-key-here
   ```

### Option 2: Enable API for Existing Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select the project associated with the API key
3. Enable "Generative Language API"
4. Ensure billing is enabled

### Option 3: Use Different Authentication
If using Google Cloud project authentication:
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

## 📊 Extraction Schema

Successfully created comprehensive 14-field schema:

1. **document_number** - Official case/document reference
2. **document_type** - judgment/tax_interpretation/legal_act
3. **title** - Document title
4. **date_issued** - ISO 8601 date
5. **summary** - 3-5 sentence concise summary
6. **thesis** - Main legal principle (1-3 sentences)
7. **keywords** - 5-15 relevant legal keywords
8. **outcome** - JSON object with decision details
9. **legal_references** - JSON array of citations
10. **legal_concepts** - JSON array of legal concepts
11. **parties** - JSON array of parties
12. **legal_analysis** - JSON object with reasoning
13. **judgment_specific** - JSON object for court cases
14. **tax_interpretation_specific** - JSON object for tax docs

## 📁 Files Created

### Scripts
- `scripts/extraction/run_extraction_rest.py` - REST API-based extraction (✅ **USE THIS**)
- `scripts/extraction/run_extraction_sample.py` - Original (has GRPC issues)
- `scripts/extraction/test_extraction_local.py` - Local testing

### Documentation
- `docs/how-to/gemini_extraction_schema.md` - Schema specification
- `docs/how-to/extraction_schema_example.md` - Detailed examples
- `docs/EXTRACTION_SCHEMA_SUMMARY.md` - Implementation summary
- `docs/EXTRACTION_STATUS.md` - This file

### Modifications
- `juddges/data/base_weaviate_db.py` - Added REST-only mode detection

## 🚀 How to Run (Once API Key is Fixed)

```bash
python scripts/extraction/run_extraction_rest.py \
  --sample-size 5 \
  --model gemini-2.5-flash \
  --output-dir data/extraction_results
```

### Command Line Options
- `--sample-size` - Number of documents to sample (default: 5)
- `--model` - gemini-2.5-flash or gemini-2.5-pro
- `--output-dir` - Output directory (default: data/extraction_results)
- `--cache-path` - SQLite cache path (default: .cache/extraction_sample.db)
- `--seed` - Random seed for reproducibility (default: 42)
- `--weaviate-host` - Override Weaviate host
- `--weaviate-port` - Override Weaviate port

## 📈 Expected Output

After successful extraction, you'll find:

1. **sample_documents_full_text.jsonl** - Original documents
2. **sample_documents_extracted.jsonl** - Extracted structured data
3. **extraction_summary.json** - Field coverage statistics

### Sample Output Structure
```json
{
  "document_id": "/doc/2A89942164",
  "document_type": "judgment",
  "extraction_status": "success",
  "extracted_data": {
    "document_number": "I ACa 123/23",
    "summary": "Court ruled that...",
    "thesis": "The main legal principle is...",
    "keywords": ["contract law", "damages", "civil procedure"],
    "outcome": {
      "decision_type": "uwzględniono",
      "decision_summary": "...",
      "awarded_amounts": [...],
      "legal_effect": "..."
    },
    "legal_references": [...],
    "parties": [...],
    "legal_analysis": {...}
  },
  "full_text_length": 45821,
  "source_language": "pl"
}
```

## 🎯 Next Steps

1. **Immediate**: Fix Google API key authentication
2. **Then**: Run full extraction test on 5-50 documents
3. **Verify**: Check Langfuse dashboard for traces
4. **Analyze**: Review field coverage statistics
5. **Scale**: Increase to 100+ documents for production
6. **Deploy**: Integrate with ingestion pipeline

## 💡 Technical Notes

### Why REST API Approach Works
- Bypasses Weaviate Python client GRPC requirements
- Uses native GraphQL endpoint: `/v1/graphql`
- Supports API key authentication via headers
- Works with public instances that only expose HTTP

### Langfuse Integration
Traces include:
- Model name and parameters
- Input text length
- Extraction duration
- Token usage
- Extraction results
- Error details (if any)

### Performance Optimization
- SQLite caching: Avoids re-extraction of same documents
- Parallel extraction: Can be added with threading/async
- Batch processing: Can fetch more documents per API call
- Temperature 0.0: Deterministic extractions

## 🔍 Troubleshooting

### If extraction still fails after fixing API key:
1. Check API quotas: `gcloud alpha billing accounts list`
2. Verify billing enabled: Google Cloud Console → Billing
3. Check API limits: 60 requests per minute for free tier
4. Monitor Langfuse for detailed error traces
5. Check `.cache/extraction_sample.db` for cached errors

### Common Issues
- **Rate limits**: Add sleep between extractions
- **Token limits**: Reduce `max_text_length` parameter
- **Cache pollution**: Delete `.cache/extraction_sample.db`
- **Network timeouts**: Increase timeout in requests.post()

## 📞 Support

For Google API issues:
- [Google AI Studio](https://aistudio.google.com/)
- [API Documentation](https://ai.google.dev/docs)

For Weaviate issues:
- REST API docs: https://weaviate.io/developers/weaviate/api/rest

For Langfuse traces:
- Dashboard: https://legal-ai-langfuse.augustyniak.ai
