# JuDDGES Project Overview

## What is JuDDGES?

**JuDDGES** (Judicial Decision Data Gathering, Encoding, and Sharing) is a comprehensive research platform designed to revolutionize the accessibility and analysis of judicial decisions across varied legal systems using advanced Natural Language Processing and Human-In-The-Loop technologies.

## Mission Statement

Overcome barriers related to resources, language, data, and format inhomogeneity in legal research by creating Europe's most comprehensive legal research repository while advancing empirical legal research through Open Science principles.

## Key Features

### 🌍 Multilingual Support

- **Polish Legal System**: Full coverage of Polish court decisions, Supreme Administrative Court (NSA)
- **English Legal System**: England & Wales court appeals
- **Expandable**: Architecture supports addition of new jurisdictions

### 🤖 Advanced NLP Capabilities

- **Semantic Search**: Vector-based search across millions of legal documents
- **Information Extraction**: Automated extraction of structured information from legal texts
- **Fine-tuned LLMs**: Domain-specific language models trained on legal corpora
- **Multilingual Embeddings**: Legal-specialized embeddings for semantic understanding

### 📊 Comprehensive Evaluation

- **N-gram Metrics**: ROUGE, exact match, precision/recall for structured data
- **LLM-as-Judge**: AI-powered qualitative assessment
- **Multi-seed Testing**: Statistical robustness through repeated evaluations

### 🔬 Open Science

- All software, tools, and datasets openly available
- Reproducible pipelines using DVC
- Public HuggingFace repositories
- Comprehensive documentation

## Target Domains

1. **Swiss Franc Loan Litigation**: Comprehensive analysis of Polish court decisions on Swiss franc-denominated loans
2. **Personal Rights Cases**: Polish legal rights extraction and analysis
3. **English Appeals**: Court of Appeal decisions from England & Wales
4. **Administrative Law**: Polish Supreme Administrative Court decisions
5. **Tax Interpretations**: Polish Ministry of Finance tax rulings

## Research Vision

JuDDGES aims to democratize access to legal knowledge by:

- Breaking down language barriers in cross-jurisdictional legal research
- Automating labor-intensive legal document analysis
- Enabling large-scale empirical legal studies
- Providing tools for comparative law research
- Supporting evidence-based policy making

## Project Consortium

A tri-national collaboration between:

- **Poland**: Legal AI research and data acquisition
- **United Kingdom**: Court system analysis and appeals data
- **France**: Research coordination and methodology

## Technology Stack

- **Vector Database**: Weaviate for semantic search
- **Embeddings**: `sdadas/mmlw-roberta-large` multilingual legal model
- **Language Models**: Llama 3.1/3.2, Mistral, Bielik, Qwen, Phi-4
- **Training**: PEFT/LoRA fine-tuning with DeepSpeed
- **Orchestration**: DVC for reproducible ML pipelines
- **Infrastructure**: Docker, Prefect, MongoDB, HuggingFace Hub

## Architecture Documentation

For comprehensive visual guides to the system architecture, see:

- **[System Architecture](./SYSTEM_ARCHITECTURE.md)** - High-level component overview with interactive diagrams
- **[Data Flow Pipeline](./DATA_FLOW_PIPELINE.md)** - Complete data transformation journey from raw documents to insights
- **[DVC Pipeline](../../reference/pipelines/DVC_PIPELINE.md)** - Reproducible ML workflows and pipeline stages
- **[Weaviate Integration](./WEAVIATE_INTEGRATION.md)** - Vector database schema and operations
- **[Model Training Flow](./MODEL_TRAINING_FLOW.md)** - Fine-tuning and inference workflows
- **[Component Relationships](./COMPONENT_RELATIONSHIPS.md)** - Module dependencies and interactions

These documents provide detailed Mermaid diagrams showing:

- System component interactions
- Data transformation stages and formats
- Training and inference pipelines
- Vector database architecture
- Configuration management
- Error handling strategies

## Getting Started

See our comprehensive guides:

- [Getting Started Tutorial](../../tutorials/GETTING_STARTED.md)
- [Data Ingestion](../../how-to/embeddings/INGESTION_GUIDE.md)
- [Embeddings Pipeline](../../how-to/embeddings/embeddings_embed_and_ingest_weaviate.md)
- [How-To Guides](../../how-to/)

## License

Open source under [appropriate license] - see LICENSE file for details.

## Contact

For questions, issues, or collaboration inquiries:

- GitHub Issues: [repository issues page]
- Project Website: [if available]
- Email: [contact email]
