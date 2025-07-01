#!/usr/bin/env python3
"""
Demo script to show Weaviate ingestion with sample legal documents.
This demonstrates the key concepts without requiring full dependencies.
"""

from typing import Any, Dict, List

import requests

# Weaviate connection details
WEAVIATE_URL = "http://localhost:8084"
API_KEY = "PQA2.12-**lafqf"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def create_legal_documents_schema():
    """Create schema for legal documents in Weaviate."""

    schema = {
        "class": "LegalDocument",
        "description": "Legal documents from courts",
        "properties": [
            {
                "name": "document_id",
                "dataType": ["text"],
                "description": "Unique document identifier",
            },
            {"name": "court_name", "dataType": ["text"], "description": "Name of the court"},
            {
                "name": "judgment_date",
                "dataType": ["date"],
                "description": "Date when judgment was issued",
            },
            {
                "name": "document_type",
                "dataType": ["text"],
                "description": "Type of legal document",
            },
            {
                "name": "full_text",
                "dataType": ["text"],
                "description": "Full text content of the document",
            },
            {"name": "judges", "dataType": ["text[]"], "description": "List of judges involved"},
            {
                "name": "legal_bases",
                "dataType": ["text[]"],
                "description": "Legal bases referenced",
            },
            {"name": "keywords", "dataType": ["text[]"], "description": "Document keywords/tags"},
            {"name": "country", "dataType": ["text"], "description": "Country/jurisdiction"},
            {"name": "language", "dataType": ["text"], "description": "Document language"},
        ],
    }

    response = requests.post(f"{WEAVIATE_URL}/v1/schema", json=schema, headers=HEADERS)
    if response.status_code == 200:
        print("✅ Legal documents schema created successfully")
    else:
        print(f"❌ Schema creation failed: {response.status_code} - {response.text}")

    return response.status_code == 200


def create_document_chunks_schema():
    """Create schema for document chunks in Weaviate."""

    schema = {
        "class": "DocumentChunk",
        "description": "Text chunks from legal documents for vector search",
        "properties": [
            {
                "name": "document_id",
                "dataType": ["text"],
                "description": "Parent document identifier",
            },
            {"name": "chunk_id", "dataType": ["text"], "description": "Unique chunk identifier"},
            {
                "name": "chunk_text",
                "dataType": ["text"],
                "description": "Text content of the chunk",
            },
            {
                "name": "position",
                "dataType": ["int"],
                "description": "Position of chunk in document",
            },
            {
                "name": "chunk_length",
                "dataType": ["int"],
                "description": "Length of chunk in characters",
            },
            {
                "name": "court_name",
                "dataType": ["text"],
                "description": "Court name from parent document",
            },
            {
                "name": "judgment_date",
                "dataType": ["date"],
                "description": "Judgment date from parent document",
            },
        ],
    }

    response = requests.post(f"{WEAVIATE_URL}/v1/schema", json=schema, headers=HEADERS)
    if response.status_code == 200:
        print("✅ Document chunks schema created successfully")
    else:
        print(f"❌ Schema creation failed: {response.status_code} - {response.text}")

    return response.status_code == 200


def create_sample_legal_documents() -> List[Dict[str, Any]]:
    """Create sample legal documents similar to JuDDGES/pl-court-raw format."""

    documents = [
        {
            "document_id": "PL_SCO_2023_001",
            "court_name": "Sąd Najwyższy",
            "judgment_date": "2023-03-15T00:00:00Z",
            "document_type": "judgment",
            "full_text": """WYROK z dnia 15 marca 2023 r.

SĄDU NAJWYŻSZEGO
Izba Cywilna

W sprawie z powództwa obywatela o zapłatę odszkodowania za szkody wyrządzone przez nieprawidłowe działanie urzędu.

UZASADNIENIE:
Sąd uznał za zasadne żądanie powoda dotyczące odszkodowania. Przepisy kodeksu cywilnego w art. 417 jasno określają odpowiedzialność Skarbu Państwa za szkody wyrządzone przez funkcjonariuszy publicznych przy wykonywaniu ich obowiązków.

W przedmiotowej sprawie udowodniono, że działania urzędników były sprzeczne z prawem i spowodowały konkretne straty materialne po stronie powoda.""",
            "judges": ["Jan Kowalski", "Anna Nowak", "Piotr Wiśniewski"],
            "legal_bases": [
                "art. 417 k.c.",
                "art. 471 k.c.",
                "ustawa o postępowaniu administracyjnym",
            ],
            "keywords": ["odszkodowanie", "odpowiedzialność", "Skarb Państwa", "szkoda"],
            "country": "Poland",
            "language": "pl",
        },
        {
            "document_id": "PL_ACO_2023_456",
            "court_name": "Wojewódzki Sąd Administracyjny w Warszawie",
            "judgment_date": "2023-04-22T00:00:00Z",
            "document_type": "judgment",
            "full_text": """WYROK z dnia 22 kwietnia 2023 r.

WOJEWÓDZKIEGO SĄDU ADMINISTRACYJNEGO W WARSZAWIE

W sprawie odwołania od decyzji administracyjnej dotyczącej odmowy wydania pozwolenia na budowę.

UZASADNIENIE:
Sąd po rozpoznaniu sprawy uznał odwołanie za uzasadnione. Organ administracji publicznej błędnie zinterpretował przepisy prawa budowlanego, w szczególności art. 35 ust. 1 ustawy Prawo budowlane.

Decyzja organu pierwszej instancji została uchylona z uwagi na naruszenie prawa materialnego. Sprawa zostaje przekazana do ponownego rozpoznania.""",
            "judges": ["Maria Kowalczyk"],
            "legal_bases": [
                "art. 35 ust. 1 Prawa budowlanego",
                "ustawa o postępowaniu administracyjnym",
            ],
            "keywords": [
                "pozwolenie na budowę",
                "prawo budowlane",
                "odwołanie",
                "uchylenie decyzji",
            ],
            "country": "Poland",
            "language": "pl",
        },
        {
            "document_id": "PL_DCO_2023_789",
            "court_name": "Sąd Rejonowy w Krakowie",
            "judgment_date": "2023-05-10T00:00:00Z",
            "document_type": "judgment",
            "full_text": """WYROK z dnia 10 maja 2023 r.

SĄDU REJONOWEGO W KRAKOWIE
Wydział Cywilny

W sprawie o zapłatę wynagrodzenia za wykonane usługi prawnicze.

UZASADNIENIE:
Sąd przychylił się do żądań powoda w pełnej wysokości. Umowa o świadczenie usług prawniczych została zawarta w sposób ważny i strony były związane jej postanowieniami.

Pozwany nie wykazał żadnych okoliczności, które mogłyby uzasadniać odmowę zapłaty wynagrodzenia. Powód udowodnił wykonanie wszystkich usług zgodnie z umową.""",
            "judges": ["Tomasz Nowacki", "Agnieszka Kowalska"],
            "legal_bases": ["art. 750 k.c.", "art. 627 k.c."],
            "keywords": ["wynagrodzenie", "usługi prawnicze", "umowa", "zapłata"],
            "country": "Poland",
            "language": "pl",
        },
    ]

    return documents


def create_document_chunks(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create text chunks from documents for vector search."""

    chunks = []

    for doc in documents:
        full_text = doc["full_text"]

        # Simple chunking by paragraphs
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]

        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > 50:  # Only meaningful chunks
                chunk = {
                    "document_id": doc["document_id"],
                    "chunk_id": f"{doc['document_id']}_chunk_{i:03d}",
                    "chunk_text": paragraph,
                    "position": i,
                    "chunk_length": len(paragraph),
                    "court_name": doc["court_name"],
                    "judgment_date": doc["judgment_date"],
                }
                chunks.append(chunk)

    return chunks


def ingest_documents(documents: List[Dict[str, Any]]) -> bool:
    """Ingest documents into Weaviate."""

    print(f"\n📥 Ingesting {len(documents)} legal documents...")

    success_count = 0
    for doc in documents:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/objects", json={"class": "LegalDocument", "properties": doc}, headers=HEADERS
        )

        if response.status_code in [200, 201]:
            success_count += 1
            print(f"  ✅ Document {doc['document_id']} ingested successfully")
        else:
            print(f"  ❌ Failed to ingest {doc['document_id']}: {response.text}")

    print(f"\n📊 Documents ingested: {success_count}/{len(documents)}")
    return success_count == len(documents)


def ingest_chunks(chunks: List[Dict[str, Any]]) -> bool:
    """Ingest document chunks into Weaviate."""

    print(f"\n📥 Ingesting {len(chunks)} document chunks...")

    success_count = 0
    for chunk in chunks:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/objects", json={"class": "DocumentChunk", "properties": chunk}, headers=HEADERS
        )

        if response.status_code in [200, 201]:
            success_count += 1
        else:
            print(f"  ❌ Failed to ingest chunk {chunk['chunk_id']}: {response.text}")

    print(f"📊 Chunks ingested: {success_count}/{len(chunks)}")
    return success_count == len(chunks)


def query_documents(query: str, limit: int = 3):
    """Query documents using keyword search."""

    print(f"\n🔍 Searching for: '{query}'")

    graphql_query = f"""
    {{
      Get {{
        LegalDocument(
          where: {{
            operator: Or
            operands: [
              {{
                path: ["full_text"]
                operator: Like
                valueText: "*{query}*"
              }}
              {{
                path: ["keywords"]
                operator: Like  
                valueText: "*{query}*"
              }}
            ]
          }}
          limit: {limit}
        ) {{
          document_id
          court_name
          judgment_date
          keywords
          full_text
        }}
      }}
    }}
    """

    response = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": graphql_query}, headers=HEADERS)

    if response.status_code == 200:
        data = response.json()
        documents = data.get("data", {}).get("Get", {}).get("LegalDocument", [])

        if documents:
            print(f"📋 Found {len(documents)} matching documents:")
            for i, doc in enumerate(documents, 1):
                print(f"\n{i}. Document: {doc['document_id']}")
                print(f"   Court: {doc['court_name']}")
                print(f"   Date: {doc['judgment_date'][:10]}")
                print(f"   Keywords: {', '.join(doc.get('keywords', []))}")
                print(f"   Text preview: {doc['full_text'][:200]}...")
        else:
            print("❌ No documents found matching the query")
    else:
        print(f"❌ Query failed: {response.status_code} - {response.text}")


def show_statistics():
    """Show collection statistics."""

    print("\n📊 WEAVIATE COLLECTION STATISTICS")
    print("=" * 50)

    # Count documents
    doc_query = """
    {
      Aggregate {
        LegalDocument {
          meta {
            count
          }
        }
      }
    }
    """

    response = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": doc_query}, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        doc_count = (
            data.get("data", {})
            .get("Aggregate", {})
            .get("LegalDocument", [{}])[0]
            .get("meta", {})
            .get("count", 0)
        )
        print(f"📄 Legal Documents: {doc_count}")

    # Count chunks
    chunk_query = """
    {
      Aggregate {
        DocumentChunk {
          meta {
            count
          }
        }
      }
    }
    """

    response = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": chunk_query}, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        chunk_count = (
            data.get("data", {})
            .get("Aggregate", {})
            .get("DocumentChunk", [{}])[0]
            .get("meta", {})
            .get("count", 0)
        )
        print(f"📝 Document Chunks: {chunk_count}")


def main():
    """Main demo function."""

    print("🏛️  WEAVIATE LEGAL DOCUMENTS INGESTION DEMO")
    print("=" * 60)
    print()
    print("This demo shows how the Universal Ingestion System")
    print("would work with Polish court documents (JuDDGES/pl-court-raw)")
    print()

    # Test connection
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/meta", headers=HEADERS)
        if response.status_code != 200:
            print("❌ Cannot connect to Weaviate")
            return
        print("✅ Connected to Weaviate successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # Create schemas
    print("\n🏗️  Creating schemas...")
    if not create_legal_documents_schema():
        return
    if not create_document_chunks_schema():
        return

    # Create sample data
    print("\n📝 Creating sample data...")
    documents = create_sample_legal_documents()
    chunks = create_document_chunks(documents)

    print(f"📄 Created {len(documents)} sample documents")
    print(f"📝 Created {len(chunks)} text chunks")

    # Ingest data
    if not ingest_documents(documents):
        print("❌ Document ingestion failed")
        return

    if not ingest_chunks(chunks):
        print("❌ Chunk ingestion failed")
        return

    # Show statistics
    show_statistics()

    # Demo queries
    print("\n🔍 DEMONSTRATION QUERIES")
    print("=" * 50)

    query_documents("odszkodowanie")
    query_documents("prawo budowlane")
    query_documents("wynagrodzenie")

    print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("\nKey features demonstrated:")
    print("✅ Automatic schema creation for Polish legal documents")
    print("✅ Document and chunk ingestion")
    print("✅ Text search across legal content")
    print("✅ Metadata handling (judges, legal bases, keywords)")
    print("✅ Multi-court document support")
    print("\nThis represents what would happen with JuDDGES/pl-court-raw:")
    print("• 1000+ documents from various Polish courts")
    print("• Automatic field mapping (judgment_id → document_id, etc.)")
    print("• Intelligent text chunking for vector search")
    print("• Rich metadata preservation")


if __name__ == "__main__":
    main()
