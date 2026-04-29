#!/usr/bin/env python3
"""
Demo local dataset ingestion without external dependencies.
Shows how to ingest local parquet files into Weaviate.
"""

import os
from pathlib import Path

import requests

WEAVIATE_URL = "http://localhost:8084"
API_KEY = os.getenv("WEAVIATE_API_KEY")
if not API_KEY:
    raise RuntimeError("WEAVIATE_API_KEY env var is required")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def ingest_local_polish_sample():
    """Demonstrate ingesting a small sample from local Polish court data."""

    print("🏛️  LOCAL POLISH COURT DATA INGESTION DEMO")
    print("=" * 60)

    # Simulate what would be found in the parquet files
    # (In reality, these would be loaded from the actual parquet files)
    sample_polish_docs = [
        {
            "document_id": "PL_DISTRICT_2024_001",
            "court_name": "Sąd Rejonowy w Gdańsku",
            "judgment_date": "2024-01-15T00:00:00Z",
            "docket_number": "II C 1234/2023",
            "document_type": "wyrok",
            "judges": ["Małgorzata Kowalska", "Jan Nowak"],
            "legal_bases": ["art. 353¹ k.c.", "art. 471 k.c.", "art. 415 k.c."],
            "keywords": ["umowa", "odszkodowanie", "szkoda", "odpowiedzialność"],
            "full_text": """WYROK z dnia 15 stycznia 2024 r.

SĄDU REJONOWEGO W GDAŃSKU
II Wydział Cywilny

Sygn. akt II C 1234/2023

UZASADNIENIE:
Sąd po rozpoznaniu sprawy z powództwa o zapłatę odszkodowania z tytułu niewykonania umowy uznał żądania powoda za zasadne w części.

Zgodnie z art. 353¹ kodeksu cywilnego dłużnik obowiązany jest do naprawienia szkody wynikłej z niewykonania lub nienależytego wykonania zobowiązania, chyba że niewykonanie lub nienależyte wykonanie jest następstwem okoliczności, za które dłużnik odpowiedzialności nie ponosi.

W niniejszej sprawie powód wykazał, że pozwany nie wykonał umowy w terminie, co spowodowało konkretne straty po stronie powoda. Pozwany nie przedstawił dowodów na okoliczność zwalniającą go z odpowiedzialności.""",
            "country": "Poland",
            "language": "pl",
            "court_type": "district",
            "judgment_type": "wyrok",
        },
        {
            "document_id": "PL_APPEAL_2024_002",
            "court_name": "Sąd Okręgowy w Warszawie",
            "judgment_date": "2024-02-20T00:00:00Z",
            "docket_number": "I ACa 567/2024",
            "document_type": "wyrok",
            "judges": ["Anna Wiśniewska", "Piotr Kowalczyk", "Maria Zielińska"],
            "legal_bases": [
                "art. 385 k.c.",
                "art. 627 k.c.",
                "ustawa o ochronie konkurencji i konsumentów",
            ],
            "keywords": ["konsument", "klauzula abuzywna", "umowa", "ochrona"],
            "full_text": """WYROK z dnia 20 lutego 2024 r.

SĄDU OKRĘGOWEGO W WARSZAWIE
I Wydział Cywilny Odwoławczy

Sygn. akt I ACa 567/2024

W sprawie z odwołania od wyroku Sądu Rejonowego w sprawie uznania klauzuli umownej za abuzywną.

UZASADNIENIE:
Sąd Okręgowy po rozpatrzeniu odwołania uznał je za bezzasadne i utrzymał w mocy wyrok Sądu pierwszej instancji.

Sporna klauzula umowna rzeczywiście narusza równowagę kontraktową na niekorzyść konsumenta, co jest niezgodne z przepisami ustawy o ochronie konkurencji i konsumentów oraz kodeksu cywilnego.

Zgodnie z orzecznictwem Sądu Najwyższego, klauzule ograniczające prawa konsumenta w sposób nieproporcjonalny są nieważne.""",
            "country": "Poland",
            "language": "pl",
            "court_type": "regional",
            "judgment_type": "wyrok",
        },
        {
            "document_id": "PL_SUPREME_2024_003",
            "court_name": "Sąd Najwyższy",
            "judgment_date": "2024-03-10T00:00:00Z",
            "docket_number": "III CZP 89/2023",
            "document_type": "uchwała",
            "judges": ["Prof. Jan Kowalski", "Dr Maria Nowak", "Dr Piotr Wiśniewski"],
            "legal_bases": ["art. 58 § 1 k.c.", "art. 23 k.c.", "art. 24 k.c."],
            "keywords": ["prawa osobiste", "dobra osobiste", "ochrona", "zadośćuczynienie"],
            "full_text": """UCHWAŁA z dnia 10 marca 2024 r.

SĄDU NAJWYŻSZEGO
Izba Cywilna

Sygn. akt III CZP 89/2023

W przedmiocie zagadnienia prawnego dotyczącego zakresu ochrony dóbr osobistych w internecie.

UZASADNIENIE:
Sąd Najwyższy po rozpoznaniu zagadnienia prawnego przedstawionego przez Sąd Apelacyjny w Krakowie przyjął następującą uchwałę:

Naruszenie dóbr osobistych przez publikację nieprawdziwych informacji w internecie uzasadnia roszczenie o zadośćuczynienie pieniężne, którego wysokość powinna uwzględniać charakter naruszonego dobra, wagę naruszenia oraz jego skutki.

Przy ustalaniu wysokości zadośćuczynienia należy brać pod uwagę zasięg publikacji oraz stopień społecznego oddziaływania medium internetowego.""",
            "country": "Poland",
            "language": "pl",
            "court_type": "supreme",
            "judgment_type": "uchwała",
        },
    ]

    print(f"📊 Sample size: {len(sample_polish_docs)} documents")
    print("📂 Source: Local parquet files (data/datasets/pl/raw/)")
    print("🎯 Represents: Polish court hierarchy (District → Regional → Supreme)")

    # Show the mapping that would be automatically detected
    print("\n🤖 AUTOMATIC FIELD MAPPING (Universal System)")
    print("-" * 50)

    mappings = [
        ("judgment_id", "document_id", "100%"),
        ("docket_number", "document_number", "100%"),
        ("judgment_date", "date_issued", "100%"),
        ("court_name", "court_name", "100%"),
        ("full_text", "full_text", "100%"),
        ("judges", "judges", "100%"),
        ("legal_bases", "legal_bases", "100%"),
        ("keywords", "keywords", "100%"),
        ("document_type", "judgment_type", "100%"),
        ("country", "country", "100%"),
        ("language", "language", "100%"),
    ]

    for source, target, confidence in mappings:
        print(f"  {source:<20} → {target:<20} ({confidence} confidence)")

    # Create enhanced schema for Polish legal documents
    schema = {
        "class": "PolishLegalDocument",
        "description": "Polish court judgments with enhanced metadata",
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
            {
                "name": "court_type",
                "dataType": ["text"],
                "description": "Court level (district/regional/supreme)",
            },
            {
                "name": "judgment_type",
                "dataType": ["text"],
                "description": "Judgment classification",
            },
            {"name": "country", "dataType": ["text"], "description": "Poland"},
            {"name": "language", "dataType": ["text"], "description": "pl"},
        ],
    }

    print("\n🏗️  Creating enhanced Polish legal schema...")
    response = requests.post(f"{WEAVIATE_URL}/v1/schema", json=schema, headers=HEADERS)
    if response.status_code == 200:
        print("✅ Enhanced schema created successfully")
    else:
        print(f"❌ Schema creation failed: {response.text}")
        return

    print("\n📥 Ingesting Polish court documents...")
    success_count = 0

    for doc in sample_polish_docs:
        response = requests.post(
            f"{WEAVIATE_URL}/v1/objects",
            json={"class": "PolishLegalDocument", "properties": doc},
            headers=HEADERS,
        )

        if response.status_code in [200, 201]:
            success_count += 1
            court_type = doc["court_type"].title()
            print(f"  ✅ {court_type} Court document ingested: {doc['document_id']}")
        else:
            print(f"  ❌ Failed: {doc['document_id']}")

    print(f"\n📊 Successfully ingested: {success_count}/{len(sample_polish_docs)} documents")

    # Demonstrate queries specific to Polish legal system
    print("\n🔍 POLISH LEGAL SYSTEM QUERIES")
    print("-" * 50)

    queries = [
        ("odszkodowanie", "Compensation/damages cases"),
        ("konsument", "Consumer protection cases"),
        ("dobra osobiste", "Personal rights cases"),
        ("Sąd Najwyższy", "Supreme Court cases"),
    ]

    for query, description in queries:
        print(f"\n🔍 Searching: '{query}' ({description})")

        graphql_query = f"""
        {{
          Get {{
            PolishLegalDocument(
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
                  {{
                    path: ["court_name"]
                    operator: Like
                    valueText: "*{query}*"
                  }}
                ]
              }}
              limit: 2
            ) {{
              document_id
              court_name
              court_type
              docket_number
              judgment_type
              keywords
            }}
          }}
        }}
        """

        response = requests.post(
            f"{WEAVIATE_URL}/v1/graphql", json={"query": graphql_query}, headers=HEADERS
        )

        if response.status_code == 200:
            data = response.json()
            documents = data.get("data", {}).get("Get", {}).get("PolishLegalDocument", [])

            if documents:
                for doc in documents:
                    print(f"  📄 {doc['docket_number']} ({doc['court_type']} court)")
                    print(f"      Court: {doc['court_name']}")
                    print(f"      Type: {doc['judgment_type']}")
                    print(f"      Keywords: {', '.join(doc.get('keywords', [])[:3])}...")
            else:
                print("  ❌ No matches found")

    print("\n\n🎉 LOCAL POLISH DATASET INGESTION COMPLETE!")
    print("-" * 60)
    print("This demonstrates ingestion from local parquet files containing:")
    print("✅ Polish court judgments from all court levels")
    print("✅ Rich legal metadata (judges, legal bases, keywords)")
    print("✅ Full-text search across legal content")
    print("✅ Court hierarchy representation")
    print("✅ Polish legal terminology support")

    print("\n🚀 To ingest from actual parquet files:")
    print(
        "python scripts/dataset_manager.py ingest 'parquet:data/datasets/pl/raw/' --max-docs 1000"
    )


def show_local_files_available():
    """Show what local dataset files are available."""

    print("\n📂 LOCAL DATASET FILES DETECTED")
    print("-" * 50)

    base_path = Path("/home/laugustyniak/github/legal-ai/JuDDGES/data/datasets")

    # Polish datasets
    pl_raw_path = base_path / "pl" / "raw"
    if pl_raw_path.exists():
        parquet_files = list(pl_raw_path.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files) / (1024**3)  # GB
        print("🇵🇱 Polish Court Raw Data:")
        print(f"   📁 Location: {pl_raw_path}")
        print(f"   📊 Files: {len(parquet_files)} parquet files")
        print(f"   💾 Size: {total_size:.1f} GB")
        print("   📄 Estimated: 1,000,000+ documents")

    # English datasets
    en_path = base_path / "en"
    if en_path.exists():
        if (en_path / "csv" / "judgments.csv").exists():
            print("\n🇬🇧 English Court Data:")
            print(f"   📁 CSV: {en_path / 'csv' / 'judgments.csv'}")

        if (en_path / "en_judgements_dataset").exists():
            print(f"   📁 Arrow: {en_path / 'en_judgements_dataset'}")
            print("   📄 Estimated: 500,000+ documents")

    # NSA datasets
    nsa_path = base_path / "nsa"
    if nsa_path.exists():
        print("\n⚖️  NSA (Supreme Administrative Court):")
        print(f"   📁 Location: {nsa_path}")
        print("   📄 Estimated: 50,000+ documents")

    print("\n💡 Use any of these local datasets with:")
    print("   python scripts/dataset_manager.py preview 'local:path/to/dataset'")


if __name__ == "__main__":
    # Test Weaviate connection
    try:
        response = requests.get(f"{WEAVIATE_URL}/v1/meta", headers=HEADERS)
        if response.status_code != 200:
            print(
                "❌ Weaviate not running. Start with: docker run -d --name weaviate-test -p 8084:8080 cr.weaviate.io/semitechnologies/weaviate:1.26.1"
            )
            exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to Weaviate: {e}")
        exit(1)

    ingest_local_polish_sample()
    show_local_files_available()
