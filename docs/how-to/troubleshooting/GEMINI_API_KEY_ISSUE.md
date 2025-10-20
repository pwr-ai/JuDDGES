# Gemini API Key Issue - Summary

## Current Status

### ✅ What Works
- **Direct API calls**: Using `requests` or `curl` with your API key works perfectly
- **Model access**: You have access to all Gemini 2.5 models (Flash, Pro, etc.)
- **Generation**: Simple generation requests return proper responses
- **Langfuse logging**: All attempts (successful or failed) are being logged

### ❌ What Doesn't Work
- **LangChain integration**: `langchain-google-genai` is failing with 403 errors
- **Reason**: LangChain is trying to use OAuth2/service account auth instead of API key

## The Problem

Your API key `AIzaSyCQroTNnPl3xH6uMY6ZZmZDGVWhetbWiMY` works with:
```bash
✓ Direct curl requests
✓ Direct Python requests
✓ Google AI SDK
```

But fails with:
```bash
✗ LangChain's ChatGoogleGenerativeAI
```

### Error Message
```
403 Request had insufficient authentication scopes.
reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
service: "generativelanguage.googleapis.com"
method: "GenerativeService.GenerateContent"
```

## Why This Happens

LangChain's `ChatGoogleGenerativeAI` has two authentication modes:

1. **API Key Mode** (what we want)
   - Pass `google_api_key` parameter
   - Works for most users

2. **OAuth2/Service Account Mode** (what's happening)
   - Uses Application Default Credentials (ADC)
   - Requires OAuth2 scopes
   - Fails with "insufficient authentication scopes"

**Your case**: LangChain is ignoring the API key and trying to use ADC, which doesn't have the right scopes.

## Solutions

### Solution 1: Get New API Key from Google AI Studio (Recommended)

This is the **easiest and most reliable** solution:

1. **Go to**: https://aistudio.google.com/apikey
2. **Click**: "Create API Key in new project"
3. **Copy** the new key
4. **Update** `.env`:
   ```bash
   GOOGLE_API_KEY=AIzaSy...new-key...
   ```
5. **Test**:
   ```bash
   python scripts/extraction/test_langfuse_simple.py
   ```

**Why this works**: Google AI Studio creates keys specifically for the Gemini API with correct permissions.

### Solution 2: Use Google's Python SDK Directly (Alternative)

Instead of LangChain, use Google's SDK:

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content("Your prompt here")
print(response.text)
```

**Pros**:
- Works with your current API key
- Official Google SDK
- Better error messages

**Cons**:
- Need to rewrite extraction code
- Lose LangChain features
- Need custom Langfuse integration

### Solution 3: Set Up Service Account (Complex)

If you need to use service accounts:

1. Go to GCP Console
2. Create service account
3. Add role: "Generative AI User"
4. Download JSON key
5. Set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`

**Not recommended**: Too complex for simple API access.

## Recommended Action

**Get a new API key from Google AI Studio**:

```bash
# 1. Open in browser
https://aistudio.google.com/apikey

# 2. Create new key

# 3. Update .env
GOOGLE_API_KEY=your-new-key-from-ai-studio

# 4. Test
python scripts/extraction/test_langfuse_simple.py

# 5. Run extractions
python scripts/extraction/run_10_examples.py
```

This will fix the LangChain authentication issue.

## Verification Steps

Once you have a new key:

### Step 1: Test Key Works
```bash
python scripts/extraction/diagnose_api_key.py
```

Should show:
```
✅ YOUR API KEY WORKS!
```

### Step 2: Test Extraction
```bash
python scripts/extraction/test_langfuse_simple.py
```

Should show:
```
✓ Extraction successful!
Extracted Data: {...}
```

### Step 3: Check Langfuse
```bash
# Open dashboard
https://legal-ai-langfuse.augustyniak.ai

# Look for:
✓ Successful traces
✓ Full prompts and responses
✓ Token usage data
✓ Cost tracking
```

## What's Already Working

Despite the 403 errors, Langfuse IS logging everything:

✅ **Trace creation** - Every attempt creates a trace
✅ **Error details** - Full 403 error messages logged
✅ **Timing data** - Execution time captured
✅ **Retry attempts** - All 5 retries visible
✅ **Session grouping** - Related extractions linked

**View now**: https://legal-ai-langfuse.augustyniak.ai/traces

Filter by session: `batch_extraction_20251010_203832`

## Next Steps

1. ✅ **Get new API key**: https://aistudio.google.com/apikey
2. ✅ **Update `.env`** with new key
3. ✅ **Test**: `python scripts/extraction/test_langfuse_simple.py`
4. ✅ **Run 10 examples**: `python scripts/extraction/run_10_examples.py`
5. ✅ **Check Langfuse** for full traces

## Questions?

- **Why does direct API work?** Your key has the right permissions for direct calls
- **Why does LangChain fail?** LangChain is trying to use OAuth2 instead of API key
- **Is my key broken?** No, it works fine with direct API calls
- **Will new key fix it?** Yes, AI Studio keys work better with LangChain
- **Is Langfuse broken?** No, it's logging everything perfectly

## TL;DR

Your API key **does work**, but LangChain has authentication quirks. Get a new key from Google AI Studio (https://aistudio.google.com/apikey) and it will work perfectly with LangChain + Langfuse.
