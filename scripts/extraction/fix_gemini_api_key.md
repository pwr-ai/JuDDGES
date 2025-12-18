# Fixing Gemini API Key - 403 Error

## The Problem

Your current API key `AIzaSyCQroTNnPl3xH6uMY6ZZmZDGVWhetbWiMY` returns:

```
403 Request had insufficient authentication scopes.
reason: "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
service: "generativelanguage.googleapis.com"
method: "GenerativeService.GenerateContent"
```

This means the API key doesn't have permission to call the **Gemini/Generative AI API**.

## The Solution

You need a **Google AI Studio API key** (not a general GCP key).

### Option 1: Create New API Key (Recommended - 5 minutes)

#### Step 1: Go to Google AI Studio
```
https://aistudio.google.com/apikey
```

#### Step 2: Sign in
Use your Google account (same one you use for GCP)

#### Step 3: Create API Key
Click the **"Create API Key"** button

You'll see two options:
- **Create API key in new project** - Choose this if you don't have a project
- **Create API key in existing project** - Choose this if you want to use an existing GCP project

#### Step 4: Copy the Key
The key will look like: `AIzaSy...` (different from your current one)

**Important:** Copy it immediately - you won't see it again!

#### Step 5: Update .env
Open your `.env` file and replace:
```bash
GOOGLE_API_KEY=AIzaSyCQroTNnPl3xH6uMY6ZZmZDGVWhetbWiMY
```

With:
```bash
GOOGLE_API_KEY=AIzaSy...your-new-key...
```

#### Step 6: Test It
```bash
python scripts/extraction/test_langfuse_simple.py
```

### Option 2: Enable API on Existing Key (More Complex)

If you want to keep using your current key, you need to enable the Generative Language API:

#### Step 1: Go to GCP Console
```
https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
```

#### Step 2: Select Your Project
Choose the project associated with your API key

#### Step 3: Enable the API
Click **"Enable"** button

#### Step 4: Check API Key Restrictions
Go to: https://console.cloud.google.com/apis/credentials

Find your API key and:
1. Click on it
2. Check "API restrictions"
3. If restricted, add: **"Generative Language API"**

#### Step 5: Wait and Test
Sometimes takes 5-10 minutes for changes to propagate
```bash
python scripts/extraction/test_langfuse_simple.py
```

## Quick Comparison

| Option | Time | Complexity | Recommended |
|--------|------|------------|-------------|
| **Option 1: New Key** | 2 min | Easy | ✅ Yes |
| **Option 2: Fix Existing** | 10-15 min | Medium | ❌ No |

## Why Option 1 is Better

1. **Faster** - 2 minutes vs 10-15 minutes
2. **Simpler** - No need to navigate GCP console
3. **Cleaner** - Dedicated key just for Gemini
4. **Free** - Same as your current key
5. **No quota issues** - Fresh quota allocation

## After Getting New Key

Once you have the new key and updated `.env`:

### Test Single Extraction
```bash
python scripts/extraction/test_langfuse_simple.py
```

### Run 10 Examples
```bash
python scripts/extraction/run_10_examples.py
```

### Check Langfuse Dashboard
```
https://legal-ai-langfuse.augustyniak.ai
```

You'll see:
- ✅ Successful extractions
- ✅ Full prompts and responses
- ✅ Token usage (input/output)
- ✅ Cost tracking
- ✅ Execution times
- ✅ Extracted JSON data

## Common Issues

### Issue: "API key not valid"
**Solution:** Make sure you copied the entire key from Google AI Studio

### Issue: Still getting 403
**Solution:**
1. Clear cache: `rm -rf .cache/`
2. Restart terminal (to reload .env)
3. Try again

### Issue: "Quota exceeded"
**Solution:** Gemini has daily quotas. Wait 24 hours or request increase at:
https://aistudio.google.com/quota

## Need Help?

1. **Check API key format:**
   ```bash
   echo $GOOGLE_API_KEY | grep "AIzaSy"
   ```
   Should start with `AIzaSy`

2. **Test manually:**
   ```bash
   curl -H "Content-Type: application/json" \
        -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
        "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=$GOOGLE_API_KEY"
   ```

3. **Check Langfuse logs:**
   Even failed attempts show up at:
   https://legal-ai-langfuse.augustyniak.ai/traces

## Summary

✅ **Get new API key from:** https://aistudio.google.com/apikey
✅ **Update .env file** with new key
✅ **Test:** `python scripts/extraction/test_langfuse_simple.py`
✅ **View results** in Langfuse dashboard

That's it! 🎉
