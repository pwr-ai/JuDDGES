#!/bin/bash
# Test if your Google API key works with Gemini API

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Testing Gemini API Key"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Load .env if it exists
if [ -f .env ]; then
    echo "📄 Loading .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check if API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "❌ GOOGLE_API_KEY not set"
    echo ""
    echo "Set it in .env file or export it:"
    echo "  export GOOGLE_API_KEY='your-key'"
    exit 1
fi

echo "✓ API key found: ${GOOGLE_API_KEY:0:20}..."
echo ""

# Test the API
echo "🚀 Testing Gemini API..."
echo ""

RESPONSE=$(curl -s -w "\n%{http_code}" \
    -H "Content-Type: application/json" \
    -d '{
        "contents": [{
            "parts": [{
                "text": "Hello, respond with just: Working!"
            }]
        }]
    }' \
    "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key=$GOOGLE_API_KEY")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

echo "HTTP Status: $HTTP_CODE"
echo ""

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ SUCCESS! Your API key works!"
    echo ""
    echo "Response preview:"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ You're ready to run extractions!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Try these commands:"
    echo "  python scripts/extraction/test_langfuse_simple.py"
    echo "  python scripts/extraction/run_10_examples.py"
elif [ "$HTTP_CODE" = "403" ]; then
    echo "❌ AUTHENTICATION ERROR (403)"
    echo ""
    echo "Your API key doesn't have permission to use Gemini API."
    echo ""
    echo "🔧 FIX:"
    echo "  1. Go to: https://aistudio.google.com/apikey"
    echo "  2. Click 'Create API Key'"
    echo "  3. Copy the new key"
    echo "  4. Update .env file:"
    echo "     GOOGLE_API_KEY=your-new-key"
    echo ""
    echo "Error details:"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
    exit 1
elif [ "$HTTP_CODE" = "400" ]; then
    echo "❌ INVALID API KEY (400)"
    echo ""
    echo "The API key format is invalid or the key doesn't exist."
    echo ""
    echo "🔧 FIX:"
    echo "  1. Check your API key starts with 'AIzaSy'"
    echo "  2. Make sure you copied the entire key"
    echo "  3. Get a new key: https://aistudio.google.com/apikey"
    echo ""
    echo "Error details:"
    echo "$BODY"
    exit 1
else
    echo "❌ UNEXPECTED ERROR ($HTTP_CODE)"
    echo ""
    echo "Response:"
    echo "$BODY"
    exit 1
fi
