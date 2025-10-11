# Gemini API Troubleshooting Guide

Complete guide for resolving Gemini API authentication and integration issues with JuDDGES.

## Table of Contents

- [Quick Solutions](#quick-solutions)
- [Common Issues](#common-issues)
- [Authentication Deep Dive](#authentication-deep-dive)
- [Verification Steps](#verification-steps)
- [Alternative Solutions](#alternative-solutions)
- [FAQ](#faq)

---

## Quick Solutions

### Use the Helper Script (Recommended)

The helper script automatically handles authentication issues:

```bash
# Run simple test
./scripts/extraction/run_extraction.sh test_langfuse_simple.py

# Run batch extraction
./scripts/extraction/run_extraction.sh run_10_examples.py

# Run any extraction script
./scripts/extraction/run_extraction.sh your_script.py
```

This script automatically:
- Disables Google Cloud SDK interference
- Sets proper environment variables
- Ensures API key authentication

### Manual Quick Fix

If running scripts directly:

```bash
# Disable Google Cloud SDK temporarily
CLOUDSDK_CONFIG=/dev/null python scripts/extraction/your_script.py
```

## Common Issues

### Issue 1: 403 Authentication Errors

**Error Message**:
```
403 Request had insufficient authentication scopes.
reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
service: "generativelanguage.googleapis.com"
```

**Cause**: LangChain is using Google Cloud SDK credentials instead of your API key.

**Solution**:

1. **Immediate Fix**: Use helper script
   ```bash
   ./scripts/extraction/run_extraction.sh test_langfuse_simple.py
   ```

2. **Code Fix**: Explicitly pass API key
   ```python
   import os

   chain = GeminiExtractionChain(
       model_name="gemini-2.5-flash",
       api_key=os.getenv("GOOGLE_API_KEY"),  # Explicit API key
   )
   ```

### Issue 2: API Key Not Found

**Error Message**:
```
API key not found. Please set GOOGLE_API_KEY environment variable.
```

**Solution**:

1. **Get API Key**: https://aistudio.google.com/apikey
2. **Set in `.env`**:
   ```bash
   GOOGLE_API_KEY=AIzaSy...your-key-here...
   ```
3. **Load environment**:
   ```bash
   source .env  # Linux/macOS
   # or
   export GOOGLE_API_KEY="AIzaSy...your-key..."
   ```

### Issue 3: LangChain Integration Failures

**Symptoms**:
- Direct API calls work
- LangChain's `ChatGoogleGenerativeAI` fails

**Root Cause**: LangChain credential discovery order:
1. Application Default Credentials (ADC) - **Checked first**
2. Service account JSON files
3. API key environment variable - **Checked last**

**Solution**: Explicitly pass API key to bypass discovery:
```python
chain = GeminiExtractionChain(
    model_name="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),  # Force API key usage
)
```

## Authentication Deep Dive

### Understanding Google Authentication Methods

#### 1. API Key Authentication (Recommended for JuDDGES)

**Pros**:
- Simple setup
- No OAuth2 complexity
- Works immediately
- Perfect for development

**How to get**:
```bash
# Visit Google AI Studio
https://aistudio.google.com/apikey

# Click "Create API Key in new project"
# Copy the key starting with "AIzaSy..."
```

#### 2. Application Default Credentials (ADC)

**What it is**: Google's automatic credential discovery

**Discovery order**:
1. `GOOGLE_APPLICATION_CREDENTIALS` env var
2. gcloud CLI configuration (`~/.config/gcloud`)
3. GCE/GKE metadata server
4. `GOOGLE_API_KEY` env var (last resort)

**Why it fails**: ADC requires OAuth2 scopes that aren't configured by default

#### 3. Service Account Authentication

**When to use**: Production deployments with fine-grained access control

**Setup**:
1. Create service account in GCP Console
2. Grant "Generative AI User" role
3. Download JSON key
4. Set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json`

### Why LangChain Has Issues

LangChain's authentication logic:
```python
# Simplified version of what LangChain does
if google_cloud_sdk_installed():
    try_adc_first()  # This often fails with scope errors
elif api_key_provided:
    use_api_key()    # This works
```

Our fix ensures API key is always used:
```python
# Force API key usage
api_key=os.getenv("GOOGLE_API_KEY")  # Explicit > Implicit
```

## Verification Steps

### Step 1: Test API Key Directly

```bash
# Run diagnostic script
python scripts/extraction/diagnose_api_key.py
```

Expected output:
```
Testing API key: AIzaSy...
✅ YOUR API KEY WORKS!
Response: "Hello! How can I help you today?"
```

### Step 2: Test with Helper Script

```bash
./scripts/extraction/run_extraction.sh test_langfuse_simple.py
```

Expected output:
```
Disabling Google Cloud SDK to prevent authentication issues...
Running extraction script...
✓ Extraction successful!
Extracted Data: {...}
```

### Step 3: Verify in Langfuse Dashboard

Visit: https://legal-ai-langfuse.augustyniak.ai

Check for:
- ✅ Successful traces (green status)
- ✅ Full prompts and responses
- ✅ Token usage statistics
- ✅ No authentication errors

### Step 4: Test Batch Processing

```bash
./scripts/extraction/run_extraction.sh run_10_examples.py
```

Monitor progress and verify all 10 extractions complete successfully.

## Alternative Solutions

### Option 1: Get Fresh API Key from AI Studio

If your current key has issues:

1. Visit https://aistudio.google.com/apikey
2. Create new API key
3. Update `.env` with new key
4. Test with diagnostic script

### Option 2: Use Google's SDK Directly

Skip LangChain entirely:

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content("Your prompt here")
print(response.text)
```

### Option 3: Remove Google Cloud SDK

If not needed for other projects:

```bash
# Ubuntu/Debian
sudo snap remove google-cloud-sdk

# macOS
brew uninstall google-cloud-sdk

# Verify removal
which gcloud  # Should return nothing
```

### Option 4: Configure Service Account Properly

For production use:

```bash
# 1. Create service account in GCP Console
# 2. Download JSON key
# 3. Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# 4. Grant necessary scopes
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

## Troubleshooting Checklist

- [ ] **API key is set**: `echo $GOOGLE_API_KEY | head -c 20`
- [ ] **No conflicting credentials**: `env | grep -i google`
- [ ] **Helper script works**: `./scripts/extraction/run_extraction.sh test_langfuse_simple.py`
- [ ] **Direct API works**: `python scripts/extraction/diagnose_api_key.py`
- [ ] **Cache cleared**: `rm -rf .cache/*.db`
- [ ] **Fresh terminal session**: Open new terminal and retry
- [ ] **Langfuse logging**: Check dashboard for traces
- [ ] **No gcloud interference**: `CLOUDSDK_CONFIG=/dev/null` set

## FAQ

### Q: Why does direct API work but LangChain fails?

**A**: LangChain checks for Google Cloud SDK credentials first. If found, it tries OAuth2 authentication which lacks proper scopes for Gemini API.

### Q: Is my API key broken?

**A**: If `diagnose_api_key.py` works, your key is fine. The issue is with LangChain's authentication logic.

### Q: Will getting a new API key fix this?

**A**: Only if you get it from Google AI Studio (https://aistudio.google.com/apikey). Keys from GCP Console may have different permissions.

### Q: Is Langfuse working despite the errors?

**A**: Yes! Langfuse logs all attempts, including failures. You can see the 403 errors in the dashboard.

### Q: Should I use service accounts instead?

**A**: Only for production deployments. For development and testing, API keys are simpler and work perfectly.

### Q: Can I disable the Google Cloud SDK check permanently?

**A**: Yes, by always explicitly passing `api_key=os.getenv("GOOGLE_API_KEY")` in your code.

## Environment Variables Reference

```bash
# Required
GOOGLE_API_KEY=AIzaSy...              # Your Gemini API key

# Optional (for Langfuse observability)
LANGFUSE_PUBLIC_KEY=pk-lf-...         # Langfuse public key
LANGFUSE_SECRET_KEY=sk-lf-...         # Langfuse secret key
LANGFUSE_HOST=https://cloud.langfuse.com  # Langfuse endpoint

# Troubleshooting (temporary)
CLOUDSDK_CONFIG=/dev/null             # Disable gcloud SDK

# Not recommended (complex)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json  # Service account
```

## Summary

The Gemini API authentication issue is caused by LangChain's credential discovery prioritizing Google Cloud SDK over API keys. The solution is simple:

1. **Use the helper script**: `./scripts/extraction/run_extraction.sh`
2. **Or manually disable gcloud**: `CLOUDSDK_CONFIG=/dev/null python ...`
3. **Always pass API key explicitly** in code

This ensures reliable Gemini API access for all extraction tasks.

---

## Related Documentation

- [Gemini Extraction Guide](GEMINI_EXTRACTION.md)
- [Langfuse Setup](LANGFUSE_SETUP.md)
- [Getting Started](GETTING_STARTED.md)

## Support

For additional help, check:
- GitHub Issues for similar problems
- Langfuse dashboard for error details
- Google AI Studio documentation

---

**Last Updated**: 2025-10-11 | **Version**: 1.0 | **Status**: Published