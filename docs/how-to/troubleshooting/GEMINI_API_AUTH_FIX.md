# Gemini API Authentication Fix

## Problem

When Google Cloud SDK (gcloud) is installed, LangChain's `ChatGoogleGenerativeAI` may try to use Application Default Credentials (ADC) instead of your API key, causing 403 authentication errors:

```
403 Request had insufficient authentication scopes.
reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
```

**Why this happens:**
- LangChain checks for credentials in this order:
  1. Google Cloud SDK Application Default Credentials (ADC)
  2. Service account JSON files
  3. API key (last resort)
- If gcloud is installed, it tries ADC first, which doesn't have the right scopes for Gemini API

## Solution

### Quick Fix (Recommended)

Use the helper script that automatically disables Google Cloud SDK:

```bash
# Run simple test
./scripts/extraction/run_extraction.sh test_langfuse_simple.py

# Run 10 examples
./scripts/extraction/run_extraction.sh run_10_examples.py
```

### Manual Fix

When running extraction scripts directly, disable Google Cloud SDK:

```bash
CLOUDSDK_CONFIG=/dev/null python scripts/extraction/test_langfuse_simple.py
CLOUDSDK_CONFIG=/dev/null python scripts/extraction/run_10_examples.py
```

### Permanent Fix (Code)

All extraction scripts now **explicitly pass the API key** to avoid relying on LangChain's credential discovery:

```python
import os

chain = GeminiExtractionChain(
    model_name="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),  # ✅ Explicitly pass API key
    cache_path=".cache/extraction.db",
    temperature=0.0,
)
```

**Without explicit API key:**
```python
chain = GeminiExtractionChain(
    model_name="gemini-2.5-flash",
    # ❌ Relies on LangChain credential discovery (may fail with gcloud)
)
```

## Verification

### 1. Test API Key Works Directly

```bash
python scripts/extraction/diagnose_api_key.py
```

Should show:
```
✅ YOUR API KEY WORKS!
```

### 2. Test Extraction

```bash
./scripts/extraction/run_extraction.sh test_langfuse_simple.py
```

Should show:
```
✓ Extraction successful!
```

### 3. Check Langfuse Dashboard

Visit: https://legal-ai-langfuse.augustyniak.ai

Look for successful traces with:
- ✅ Full prompts and responses
- ✅ Token usage (input/output)
- ✅ Extraction results as JSON
- ✅ No 403 errors

## Understanding the Issue

### What is Application Default Credentials (ADC)?

ADC is Google Cloud's automatic credential discovery system that checks:
1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable (service account JSON)
2. gcloud CLI configuration (`~/.config/gcloud`)
3. GCE/GKE/Cloud Run metadata server
4. Finally, checks for `GOOGLE_API_KEY` environment variable

### Why ADC Fails for Gemini

ADC credentials (service accounts, user OAuth2) use **OAuth2 scopes**:
- `https://www.googleapis.com/auth/cloud-platform`
- `https://www.googleapis.com/auth/generative-language`

These scopes must be **explicitly granted** when creating credentials. Your gcloud user account doesn't have these scopes by default.

### Why API Keys Work

API keys from Google AI Studio (https://aistudio.google.com/apikey) have **built-in Gemini access** - no scopes required.

## Environment Check

Check if you have gcloud installed:

```bash
which gcloud
# /snap/google-cloud-sdk/current/bin/gcloud

echo $CLOUDSDK_HOME
# /snap/google-cloud-sdk/current
```

If gcloud is installed, you **must** use the fix above.

## Alternative Solutions

### Option 1: Uninstall gcloud (Not Recommended)

If you don't need gcloud SDK:
```bash
sudo snap remove google-cloud-sdk
```

### Option 2: Use Service Account with Proper Scopes

Create a service account with Generative Language API access:

1. Go to GCP Console → IAM & Admin → Service Accounts
2. Create service account
3. Grant role: "Generative AI User"
4. Download JSON key
5. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   ```

**Note:** This is more complex than using API keys.

## Troubleshooting

### Still getting 403 errors?

1. **Check API key is set:**
   ```bash
   echo $GOOGLE_API_KEY | head -c 20
   # Should show: AIzaSy...
   ```

2. **Verify API key works directly:**
   ```bash
   curl -H "Content-Type: application/json" \
        -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
        "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=$GOOGLE_API_KEY"
   ```

3. **Check for conflicting credentials:**
   ```bash
   env | grep -i google
   # Should NOT show GOOGLE_APPLICATION_CREDENTIALS
   ```

4. **Try with fresh terminal:**
   ```bash
   # New terminal session
   source .env  # Or reload environment
   CLOUDSDK_CONFIG=/dev/null python scripts/extraction/test_langfuse_simple.py
   ```

### Cache issues?

Clear LangChain cache:
```bash
rm -rf .cache/*.db
```

## Summary

✅ **Use helper script:** `./scripts/extraction/run_extraction.sh test_langfuse_simple.py`

✅ **Or manually disable gcloud:** `CLOUDSDK_CONFIG=/dev/null python ...`

✅ **Always explicitly pass API key** in code: `api_key=os.getenv("GOOGLE_API_KEY")`

✅ **Verify with:** `python scripts/extraction/diagnose_api_key.py`

This ensures LangChain uses your Gemini API key instead of trying Google Cloud SDK credentials.
