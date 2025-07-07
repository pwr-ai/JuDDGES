#!/usr/bin/env python3
"""
Simple Docker-based ingestion script for Polish court data into Weaviate.
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Weaviate connection
WEAVIATE_URL = "http://172.17.0.1:8084"  # Docker bridge network
API_KEY = "<REDACTED>"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def create_schema():
    """Create Weaviate schema for Polish court documents."""
    schema = {
        "class": "PolishCourtDocument",
        "description": "Polish court judgments with comprehensive metadata",
        "properties": [
            {
                "name": "document_id",
                "dataType": ["text"],
                "description": "Unique judgment identifier",
            },
            {"name": "court_name", "dataType": ["text"], "description": "Polish court name"},
            {"name": "judgment_date", "dataType": ["date"], "description": "Date of judgment"},
            {"name": "docket_number", "dataType": ["text"], "description": "Case docket number"},
            {
                "name": "document_type",
                "dataType": ["text"],
                "description": "Type (wyrok, uchwała, postanowienie)",
            },
            {"name": "judges", "dataType": ["text[]"], "description": "Panel of judges"},
            {"name": "legal_bases", "dataType": ["text[]"], "description": "Legal articles cited"},
            {"name": "keywords", "dataType": ["text[]"], "description": "Legal keywords/topics"},
            {"name": "full_text", "dataType": ["text"], "description": "Complete judgment text"},
            {"name": "court_type", "dataType": ["text"], "description": "Court level"},
            {
                "name": "judgment_type",
                "dataType": ["text"],
                "description": "Judgment classification",
            },
            {"name": "country", "dataType": ["text"], "description": "Poland"},
            {"name": "language", "dataType": ["text"], "description": "pl"},
        ],
    }

    print("🏗️ Creating schema...")
    response = requests.post(f"{WEAVIATE_URL}/v1/schema", json=schema, headers=HEADERS)
    if response.status_code in [200, 422]:  # 422 means already exists
        print("✅ Schema ready")
        return True
    else:
        print(f"❌ Schema creation failed: {response.text}")
        return False


def load_parquet_sample(data_path: str, max_documents: int = 100):
    """Load sample data from parquet files."""
    print(f"📂 Loading data from {data_path}")

    parquet_files = list(Path(data_path).glob("*.parquet"))
    if not parquet_files:
        print(f"❌ No parquet files found in {data_path}")
        return []

    print(f"📊 Found {len(parquet_files)} parquet files")

    documents = []
    for file_path in parquet_files[:3]:  # Use first 3 files
        try:
            df = pd.read_parquet(file_path)
            print(f"📄 Loaded {len(df)} records from {file_path.name}")

            # Convert to documents
            sample_size = min(len(df), max_documents // len(parquet_files[:3]))
            for _, row in df.head(sample_size).iterrows():
                doc = {
                    "document_id": str(row.get("judgment_id", f"DOC_{uuid.uuid4().hex[:8]}")),
                    "court_name": str(row.get("court_name", "Unknown Court")),
                    "judgment_date": format_date(row.get("judgment_date")),
                    "docket_number": str(row.get("docket_number", "")),
                    "document_type": str(row.get("document_type", "wyrok")),
                    "judges": parse_list_field(row.get("judges", [])),
                    "legal_bases": parse_list_field(row.get("legal_bases", [])),
                    "keywords": parse_list_field(row.get("keywords", [])),
                    "full_text": str(row.get("full_text", ""))[:50000],  # Limit text length
                    "court_type": str(row.get("court_type", "district")),
                    "judgment_type": str(row.get("judgment_type", "wyrok")),
                    "country": "Poland",
                    "language": "pl",
                }
                documents.append(doc)

                if len(documents) >= max_documents:
                    break

        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            continue

        if len(documents) >= max_documents:
            break

    return documents


def format_date(date_val):
    """Format date for Weaviate."""
    if pd.isna(date_val):
        return "2023-01-01T00:00:00Z"

    try:
        if isinstance(date_val, str):
            # Try to parse common formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y"]:
                try:
                    dt = datetime.strptime(date_val, fmt)
                    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    continue
            return "2023-01-01T00:00:00Z"
        elif hasattr(date_val, "strftime"):
            return date_val.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            return "2023-01-01T00:00:00Z"
    except (ValueError, TypeError):
        return "2023-01-01T00:00:00Z"


def parse_list_field(field_val):
    """Parse list fields safely."""
    try:
        if pd.isna(field_val):
            return []
    except (ValueError, TypeError):
        # Handle arrays or complex types
        if field_val is None:
            return []
    if isinstance(field_val, str):
        # Try to split by common separators
        return [x.strip() for x in field_val.replace(";", ",").split(",") if x.strip()]
    elif isinstance(field_val, list):
        return [str(x) for x in field_val if x]
    else:
        return [str(field_val)] if field_val else []


def ingest_documents(documents):
    """Ingest documents into Weaviate."""
    print(f"\n📥 Ingesting {len(documents)} documents...")

    success_count = 0
    for i, doc in enumerate(documents):
        try:
            response = requests.post(
                f"{WEAVIATE_URL}/v1/objects",
                json={"class": "PolishCourtDocument", "properties": doc},
                headers=HEADERS,
            )

            if response.status_code in [200, 201]:
                success_count += 1
                if i % 10 == 0:
                    print(f"  ✅ Progress: {i + 1}/{len(documents)}")
            else:
                print(f"  ❌ Failed doc {i}: {response.text[:100]}")

        except Exception as e:
            print(f"  ❌ Error ingesting doc {i}: {e}")

    print(f"\n📊 Successfully ingested: {success_count}/{len(documents)} documents")
    return success_count


def test_weaviate_connection():
    """Test connection to Weaviate."""
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/meta", headers=HEADERS, timeout=5)
        if response.status_code == 200:
            meta = response.json()
            print(f"✅ Connected to Weaviate v{meta.get('version', 'unknown')}")
            return True
        else:
            print(f"❌ Weaviate connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Weaviate: {e}")
        return False


def main():
    """Main ingestion function."""
    print("🏛️ POLISH COURT DATA INGESTION TO WEAVIATE")
    print("=" * 60)

    # Get parameters
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/app/data/datasets/pl/raw"
    max_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    print(f"📂 Data path: {data_path}")
    print(f"📊 Max documents: {max_docs}")

    # Test connection
    if not test_weaviate_connection():
        sys.exit(1)

    # Create schema
    if not create_schema():
        sys.exit(1)

    # Load data
    documents = load_parquet_sample(data_path, max_docs)
    if not documents:
        print("❌ No documents to ingest")
        sys.exit(1)

    # Ingest
    success_count = ingest_documents(documents)

    print("\n🎉 INGESTION COMPLETE!")
    print(f"✅ {success_count} documents successfully ingested into Weaviate")


if __name__ == "__main__":
    main()
