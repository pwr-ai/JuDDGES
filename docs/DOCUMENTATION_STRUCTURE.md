# JuDDGES Documentation Structure

## Overview

The JuDDGES documentation has been reorganized according to the **Diátaxis framework**, which divides documentation into four distinct categories based on user needs and content purposes. This structure provides clear navigation paths and helps users find exactly what they need.

## Diátaxis Categories

### 📚 [Tutorials](tutorials/)
**Purpose**: Learning-oriented guides for getting started
**Audience**: New users, beginners
**Content Style**: Step-by-step, educational, hands-on practice

**Current Tutorials**:
- `GETTING_STARTED.md` - Complete setup and first steps
- `GEMINI_EXTRACTION.md` - Learn structured extraction with Gemini
- `LANGFUSE_SETUP.md` - Set up observability and tracing
- `GIT_LFS_SETUP.md` - Configure large file storage

---

### 🔧 [How-To Guides](how-to/)
**Purpose**: Task-oriented instructions for specific problems
**Audience**: Users with basic knowledge
**Content Style**: Problem-focused, practical solutions, direct steps

**Categories**:
- **[data-acquisition/](how-to/data-acquisition/)** - Data collection pipelines
  - `data_acquisition_pl_court.md` - Polish court data pipeline
  - `data_acquisition_nsa.md` - NSA data acquisition

- **[embeddings/](how-to/embeddings/)** - Vector embeddings and ingestion
  - `embeddings_deploy_weaviate.md` - Deploy Weaviate database
  - `embeddings_embed_and_ingest_weaviate.md` - Complete embedding workflow
  - `INGESTION_GUIDE.md` - Dataset ingestion guide
  - `UNIVERSAL_INGESTION.md` - Flexible ingestion approach
  - `STREAMING_INGESTER.md` - Production streaming pipeline

- **[extraction/](how-to/extraction/)** - Information extraction techniques
  - `EXTRACTION_SUCCESS_SUMMARY.md` - Implementation summary
  - `iterative_refinment.md` - Schema refinement process

- **[visualization/](how-to/visualization/)** - Data visualization
  - `UMAP_VISUALIZATION.md` - Visualization pipeline overview
  - `UMAP_SAMPLING.md` - Document sampling for UMAP
  - `UMAP_APPLY_COORDS.md` - Apply coordinates to database

- **[troubleshooting/](how-to/troubleshooting/)** - Problem solutions
  - `GEMINI_API_TROUBLESHOOTING.md` - Comprehensive troubleshooting
  - `GEMINI_API_AUTH_FIX.md` - Authentication solutions
  - `GEMINI_API_KEY_ISSUE.md` - API key problems

---

### 📖 [Reference](reference/)
**Purpose**: Information-oriented technical specifications
**Audience**: Users needing specific technical details
**Content Style**: Comprehensive, accurate, structured for lookup

**Categories**:
- **[api/](reference/api/)** - API documentation
  - `llm_fields_quick_reference.md` - LLM-extracted fields
  - `raw_text_quick_reference.md` - Raw content fields

- **[schemas/](reference/schemas/)** - Data schemas and mappings
  - `DOCUMENT_SCHEMA_MAPPING.md` - Complete schema reference
  - `dataset_weaviate_mapping.md` - Dataset to Weaviate mapping
  - `gemini_extraction_schema.md` - Gemini extraction schema
  - `llm_field_extraction_schema.yaml` - LLM field extraction schema

- **[configurations/](reference/configurations/)** - Configuration references
  - *(To be populated with configuration documentation)*

- **Standards**:
  - `STYLE_GUIDE.md` - Documentation and coding standards

---

### 💡 [Explanation](explanation/)
**Purpose**: Understanding-oriented conceptual discussions
**Audience**: Users seeking deeper understanding
**Content Style**: Conceptual, discusses alternatives, provides context

**Categories**:
- **[architecture/](explanation/architecture/)** - System design
  - `PROJECT_OVERVIEW.md` - Comprehensive project introduction

- **[research/](explanation/research/)** - Academic context
  - `RESEARCH_PUBLICATIONS.md` - Publications roadmap

- **[achievements/](explanation/achievements/)** - Project impact
  - `MILESTONES_AND_ACHIEVEMENTS.md` - Project accomplishments
  - `IMPACT_ASSESSMENT.md` - Impact analysis

- **Main Documents**:
  - `EXECUTIVE_SUMMARY.md` - High-level project overview

---

## Meta Documentation

These documents remain at the root level as they concern the documentation itself:

- `README.md` - Main documentation index and navigation hub
- `DOCUMENTATION_ANALYSIS.md` - Analysis of documentation state
- `DOCUMENTATION_IMPROVEMENTS.md` - Improvement roadmap
- `DOCUMENTATION_REVIEW_SUMMARY.md` - Review findings
- `DOCUMENTATION_STRUCTURE.md` - This file

## Navigation Principles

1. **Start Here**: New users should begin with [tutorials/GETTING_STARTED.md](tutorials/GETTING_STARTED.md)
2. **Find Solutions**: Users with specific tasks should browse [how-to/](how-to/)
3. **Look Up Details**: Technical specifications are in [reference/](reference/)
4. **Understand Concepts**: Background and theory are in [explanation/](explanation/)

## Benefits of This Structure

- **Clear Mental Model**: Users know where to look based on their needs
- **Reduced Cognitive Load**: Separation by purpose prevents information overload
- **Better Maintenance**: Clear categories make it easier to add and update docs
- **Improved Discovery**: Users can browse related content within categories
- **Consistent Quality**: Each category has specific writing guidelines

## Contributing to Documentation

When adding new documentation:

1. **Identify the Category**: Is it teaching (tutorial), solving (how-to), describing (reference), or explaining (explanation)?
2. **Follow Templates**: Each category has a README.md with templates
3. **Maintain Cross-References**: Update links when moving or adding files
4. **Use Relative Paths**: For internal links between documents
5. **Update Index**: Add new documents to the main README.md index

## Future Improvements

- Add more tutorials for common workflows
- Expand reference documentation with API specs
- Create explanation documents for design decisions
- Add configuration reference documentation
- Implement automated link checking
- Generate API documentation from code

---

**Last Updated**: 2025-10-11
**Framework**: Diátaxis (https://diataxis.fr/)
**Maintainer**: JuDDGES Documentation Team