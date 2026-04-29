# Weaviate Schema Update Guide

Based on research from Weaviate documentation and community forums (2024).

## Summary of Capabilities

| Operation | Possible? | Method |
|-----------|-----------|--------|
| **Add new properties** | ✅ Yes | `collection.config.add_property()` |
| **Modify existing property types** | ❌ No | Must migrate to new collection |
| **Delete properties** | ❌ No | Must migrate to new collection |
| **Change vectorizer** | ❌ No | Must migrate to new collection |

## 1. Adding New Properties to Existing Collections

### ✅ This IS Supported

You can add new properties to existing collections after creation.

### Python Code Example (v4 Client)

```python
import weaviate
import weaviate.classes.config as wvcc

# Connect to Weaviate
client = weaviate.connect_to_local()

# Get the collection
collection = client.collections.get("LegalDocuments")

# Add new property
collection.config.add_property(
    wvcc.Property(
        name="extracted_factual_state",
        data_type=wvcc.DataType.TEXT,
        description="Extracted factual circumstances and background of the case",
        skip_vectorization=False,  # Enable for semantic search
        vectorize_property_name=False,
        index_searchable=True
    )
)

# Add another property
collection.config.add_property(
    wvcc.Property(
        name="extracted_legal_state",
        data_type=wvcc.DataType.TEXT,
        description="Extracted legal basis and applicable law",
        skip_vectorization=False,
        vectorize_property_name=False,
        index_searchable=True
    )
)

client.close()
```

### ⚠️ Important Limitations

**Inverted Index Issue**: When you add a property to a collection that already contains data, there are limitations in inverted-index related behavior:

- Filtering by the new property's **length** may not work correctly
- Filtering by the new property's **null status** may not work correctly

**Why?** The inverted index is built at import time. If you add a property after importing objects, the inverted index metadata (length, null status) won't be retroactively updated for existing objects.

### Recommended Approaches

**Option 1: Add Properties Before Data Import** (Best)

```python
# 1. Add all properties to schema first
collection.config.add_property(...)

# 2. Then import data
collection.data.insert_many(objects)
```

**Option 2: Accept Limitations** (Quick)

```python
# Add property to existing data
# Accept that inverted index filtering may be limited
collection.config.add_property(...)
```

**Option 3: Re-import Data** (Most Complete)

```python
# 1. Export all existing data
objects = collection.query.fetch_objects(limit=10000)

# 2. Delete collection
client.collections.delete("LegalDocuments")

# 3. Recreate with new properties
# (use your create_collections() method)

# 4. Re-import all data
collection.data.insert_many(objects)
```

**Option 4: Wait for Re-indexing API** (Future)

- Weaviate is working on a re-indexing API
- Will allow retroactive index updates
- Not available yet (as of 2024)

## 2. Modifying Existing Property Types

### ❌ This is NOT Supported

**You CANNOT**:

- Change a property's data type (e.g., `TEXT` → `TEXT_ARRAY`)
- Delete properties from a collection
- Modify property configurations after creation

**Official Statement**: "You cannot modify existing properties after you create the collection."

### Workaround: Migration to New Collection

The only way to change property types is to create a new collection and migrate data.

#### Migration Process

```python
import weaviate
import weaviate.classes.config as wvcc
from loguru import logger
from rich.progress import Progress

# Connect to Weaviate
client = weaviate.connect_to_local()

# Step 1: Export data from old collection
logger.info("Exporting data from LegalDocuments...")
old_collection = client.collections.get("LegalDocuments")

all_objects = []
cursor = None

# Paginate through all objects
while True:
    response = old_collection.query.fetch_objects(
        limit=1000,
        after=cursor
    )

    all_objects.extend(response.objects)

    if len(response.objects) < 1000:
        break

    cursor = response.objects[-1].uuid

logger.info(f"Exported {len(all_objects)} objects")

# Step 2: Delete old collection
logger.info("Deleting old LegalDocuments collection...")
client.collections.delete("LegalDocuments")

# Step 3: Create new collection with updated schema
logger.info("Creating new LegalDocuments collection with updated schema...")

client.collections.create(
    name="LegalDocuments",
    description="Collection of legal documents with updated schema",
    properties=[
        # Updated property with new type
        wvcc.Property(
            name="legal_references",
            data_type=wvcc.DataType.TEXT_ARRAY,  # Changed from TEXT to TEXT_ARRAY
            description="References to legal acts and regulations",
            skip_vectorization=True
        ),
        # ... add all other properties
    ],
    vectorizer_config=[
        # ... vectorizer config
    ]
)

# Step 4: Transform and re-import data
new_collection = client.collections.get("LegalDocuments")

logger.info("Migrating data to new collection...")

with Progress() as progress:
    task = progress.add_task("Migrating...", total=len(all_objects))

    batch_size = 100
    for i in range(0, len(all_objects), batch_size):
        batch = all_objects[i:i+batch_size]

        # Transform objects to match new schema
        transformed = []
        for obj in batch:
            props = obj.properties.copy()

            # Transform property types as needed
            # Example: Convert JSON string to array
            if "legal_references" in props and isinstance(props["legal_references"], str):
                import json
                try:
                    props["legal_references"] = json.loads(props["legal_references"])
                except:
                    props["legal_references"] = []

            transformed.append(props)

        # Insert batch
        new_collection.data.insert_many(transformed)
        progress.update(task, advance=len(batch))

logger.info("Migration complete!")
client.close()
```

## 3. Our Specific Use Case

### What We Need to Do

Based on our schema comparison, we have two options:

#### Option A: Add Missing Properties Only (Recommended)

Add these 2 new properties:

- `extracted_factual_state` (TEXT)
- `extracted_legal_state` (TEXT)

**Pros**:

- Simple, fast
- No data migration needed
- No downtime

**Cons**:

- Type mismatches remain (JSON strings for lists)
- Less optimal for querying

**Code**:

```python
from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
import weaviate.classes.config as wvcc

with WeaviateLegalDocumentsDatabase() as db:
    collection = db.legal_documents_collection

    # Add factual_state property
    collection.config.add_property(
        wvcc.Property(
            name="extracted_factual_state",
            data_type=wvcc.DataType.TEXT,
            description="Extracted factual circumstances and background of the case",
            skip_vectorization=False,
            vectorize_property_name=False,
            index_searchable=True
        )
    )

    # Add legal_state property
    collection.config.add_property(
        wvcc.Property(
            name="extracted_legal_state",
            data_type=wvcc.DataType.TEXT,
            description="Extracted legal basis and applicable law",
            skip_vectorization=False,
            vectorize_property_name=False,
            index_searchable=True
        )
    )
```

#### Option B: Full Schema Migration (Production-Ready)

Fix all type mismatches:

- `legal_references`: TEXT (JSON) → TEXT_ARRAY
- `legal_concepts`: TEXT (JSON) → TEXT_ARRAY
- `parties`: TEXT (JSON) → TEXT_ARRAY
- `outcome`: TEXT (JSON) → TEXT
- `legal_analysis`: TEXT (JSON) → TEXT
- Add `extracted_factual_state` and `extracted_legal_state`

**Pros**:

- Proper types for optimal querying
- Native array support for list fields
- Better semantic search on text fields
- Production-ready schema

**Cons**:

- Requires data migration
- More complex
- Temporary downtime or dual-collection period

**Required**: Full migration script (see Section 2 above)

## 4. Recommendations

### For Our JuDDGES Project

**Phase 1: Quick Implementation (This Week)**

1. ✅ Add 2 missing properties (`extracted_factual_state`, `extracted_legal_state`)
2. ✅ Implement ingestion with JSON serialization for existing properties
3. ✅ Test with IP Box data (43 documents)
4. ✅ Validate extraction pipeline works end-to-end

**Phase 2: Production Migration (When Ready)**

1. ⏸️ Plan data migration window
2. ⏸️ Create migration script with proper type conversions
3. ⏸️ Test migration on staging/copy of data
4. ⏸️ Execute production migration
5. ⏸️ Validate all data migrated correctly
6. ⏸️ Update ingestion code to use new types

## 5. Script: Add Missing Properties

I'll create this as a standalone script:

```python
#!/usr/bin/env python3
"""
Add extracted_factual_state and extracted_legal_state properties to LegalDocuments collection.
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from loguru import logger

from juddges.data.documents_weaviate_db import WeaviateLegalDocumentsDatabase
from juddges.settings import ROOT_PATH
import weaviate.classes.config as wvcc

load_dotenv(ROOT_PATH / ".env", override=True)


def add_extraction_properties():
    """Add new properties for extraction fields."""
    console = Console()

    console.print("[bold blue]Adding Extraction Properties to Weaviate Schema[/bold blue]\n")

    try:
        with WeaviateLegalDocumentsDatabase() as db:
            console.print("✅ Connected to Weaviate")

            collection = db.legal_documents_collection

            # Check existing properties
            existing_props = db.legal_documents_properties
            console.print(f"\n[cyan]Current properties ({len(existing_props)}):[/cyan]")
            console.print(", ".join(existing_props[:10]) + "...")

            # Add extracted_factual_state
            if "extracted_factual_state" not in existing_props:
                console.print("\n[yellow]Adding property: extracted_factual_state[/yellow]")
                collection.config.add_property(
                    wvcc.Property(
                        name="extracted_factual_state",
                        data_type=wvcc.DataType.TEXT,
                        description="Extracted factual circumstances and background of the case",
                        skip_vectorization=False,
                        vectorize_property_name=False,
                        index_searchable=True
                    )
                )
                console.print("[green]✓ Added extracted_factual_state[/green]")
            else:
                console.print("[dim]• extracted_factual_state already exists[/dim]")

            # Add extracted_legal_state
            if "extracted_legal_state" not in existing_props:
                console.print("\n[yellow]Adding property: extracted_legal_state[/yellow]")
                collection.config.add_property(
                    wvcc.Property(
                        name="extracted_legal_state",
                        data_type=wvcc.DataType.TEXT,
                        description="Extracted legal basis and applicable law",
                        skip_vectorization=False,
                        vectorize_property_name=False,
                        index_searchable=True
                    )
                )
                console.print("[green]✓ Added extracted_legal_state[/green]")
            else:
                console.print("[dim]• extracted_legal_state already exists[/dim]")

            # Verify
            console.print("\n[cyan]Verifying changes...[/cyan]")
            updated_props = db.legal_documents_properties
            console.print(f"[green]✓ Collection now has {len(updated_props)} properties[/green]")

            if "extracted_factual_state" in updated_props:
                console.print("[green]✓ extracted_factual_state confirmed[/green]")
            if "extracted_legal_state" in updated_props:
                console.print("[green]✓ extracted_legal_state confirmed[/green]")

            console.print("\n[bold green]✅ Schema update complete![/bold green]")

    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        logger.exception("Schema update failed")
        sys.exit(1)


if __name__ == "__main__":
    add_extraction_properties()
```

## 6. Reference: Property Configuration Options

When adding properties, you can configure:

```python
wvcc.Property(
    name="property_name",                    # Required: Property name
    data_type=wvcc.DataType.TEXT,           # Required: TEXT, TEXT_ARRAY, NUMBER, BOOL, etc.
    description="Description",               # Optional: Human-readable description
    skip_vectorization=False,                # Skip this property in vectorization?
    vectorize_property_name=False,           # Include property name in vector?
    index_searchable=True,                   # Enable inverted index for filtering?
    index_filterable=True,                   # (Alias for index_searchable)
    tokenization=wvcc.Tokenization.WORD,    # Word-level or field-level tokens
)
```

### Common Data Types

```python
wvcc.DataType.TEXT          # Single text string
wvcc.DataType.TEXT_ARRAY    # Array of text strings
wvcc.DataType.NUMBER        # Integer or float
wvcc.DataType.INT           # Integer only
wvcc.DataType.BOOL          # Boolean
wvcc.DataType.DATE          # ISO 8601 datetime
wvcc.DataType.GEO_COORDINATES  # Geographic coordinates
wvcc.DataType.PHONE_NUMBER  # Phone number
wvcc.DataType.BLOB          # Binary data
wvcc.DataType.UUID          # UUID reference to another object
```

## 7. Conclusion

For our use case:

- **Immediate action**: Add 2 missing properties using `collection.config.add_property()`
- **Future consideration**: Full schema migration to fix type mismatches
- **Current limitation**: Cannot change existing property types without migration

The good news: We can start ingesting extracted data immediately by adding just 2 properties!
