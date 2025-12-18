# Architecture Documentation

This directory contains comprehensive architecture documentation for the JuDDGES system, featuring detailed Mermaid diagrams and conceptual explanations.

## Overview

JuDDGES is a complex legal AI system with multiple interconnected components. These documents provide visual and conceptual understanding of how the system works, how components interact, and how data flows through the pipeline.

## Documentation Structure

### Core Architecture Documents

#### [System Architecture](./SYSTEM_ARCHITECTURE.md)
**Purpose**: High-level overview of the entire JuDDGES system

**What You'll Learn**:
- Overall system architecture and main components
- External integrations (Weaviate, HuggingFace, DVC)
- Technology stack and component interactions
- Scalability and security considerations

**Diagrams**:
- High-level system architecture
- Component interaction patterns
- Technology stack relationships

**Best For**: Getting the big picture, understanding system boundaries, planning integrations

---

#### [Data Flow Pipeline](./DATA_FLOW_PIPELINE.md)
**Purpose**: Detailed visualization of data transformation journey

**What You'll Learn**:
- 9-stage data processing pipeline
- Data format transformations (PDF → Text → Vectors → Predictions)
- Parallel processing architecture
- Processing metrics and performance optimization

**Diagrams**:
- Complete data flow pipeline (Stage 1-9)
- Data format evolution
- Parallel execution paths
- Data volume flow (Sankey diagram)
- Error handling and recovery

**Best For**: Understanding data processing, debugging pipeline issues, optimizing performance

---

#### [Weaviate Integration](./WEAVIATE_INTEGRATION.md)
**Purpose**: Vector database architecture and operations

**What You'll Learn**:
- Weaviate infrastructure and deployment
- Schema design for legal documents and chunks
- Query patterns (semantic, hybrid, RAG)
- Performance optimization strategies

**Diagrams**:
- Integration architecture
- Collection schemas (class diagrams)
- Data ingestion pipeline
- Query architecture (sequence diagrams)
- Hybrid search components
- Docker deployment

**Best For**: Working with vector database, implementing search, schema design

---

#### [Model Training Flow](./MODEL_TRAINING_FLOW.md)
**Purpose**: Training and inference workflow visualization

**What You'll Learn**:
- Complete training pipeline from data to checkpoints
- PEFT/LoRA fine-tuning strategy
- Multi-model training matrix
- Inference pipeline with context retrieval
- Optimization techniques and hardware requirements

**Diagrams**:
- Training architecture (end-to-end)
- LoRA parameter-efficient fine-tuning
- Multi-model training matrix
- Inference pipeline
- Optimization strategies
- Deployment options
- Hardware requirements by model size

**Best For**: Training models, understanding fine-tuning, planning deployments

---

#### [Component Relationships](./COMPONENT_RELATIONSHIPS.md)
**Purpose**: Module dependencies and interaction patterns

**What You'll Learn**:
- JuDDGES module structure and dependencies
- Class relationships and inheritance
- Configuration hierarchy
- Error propagation paths
- Testing structure

**Diagrams**:
- High-level component architecture
- Detailed module dependencies
- Class relationship diagrams
- Data flow dependencies
- Configuration hierarchy
- Error handling flows
- Testing dependencies
- Package dependencies

**Best For**: Development, debugging, understanding codebase structure, refactoring

---

#### [Project Overview](./PROJECT_OVERVIEW.md)
**Purpose**: Introduction to JuDDGES mission and capabilities

**What You'll Learn**:
- Mission statement and research vision
- Key features and target domains
- Project consortium and collaborators
- Technology stack
- Getting started resources

**Best For**: New users, stakeholders, understanding project goals

---

## How to Use This Documentation

### For Developers
1. Start with [System Architecture](./SYSTEM_ARCHITECTURE.md) for the big picture
2. Review [Component Relationships](./COMPONENT_RELATIONSHIPS.md) to understand code structure
3. Dive into [Data Flow Pipeline](./DATA_FLOW_PIPELINE.md) when working with data processing
4. Consult [Model Training Flow](./MODEL_TRAINING_FLOW.md) for training and inference tasks
5. Reference [Weaviate Integration](./WEAVIATE_INTEGRATION.md) for database operations

### For Researchers
1. Begin with [Project Overview](./PROJECT_OVERVIEW.md) to understand the project
2. Study [System Architecture](./SYSTEM_ARCHITECTURE.md) for technical understanding
3. Examine [Data Flow Pipeline](./DATA_FLOW_PIPELINE.md) for data processing methodology
4. Review [Model Training Flow](./MODEL_TRAINING_FLOW.md) for reproducibility details

### For System Architects
1. Review [System Architecture](./SYSTEM_ARCHITECTURE.md) for overall design
2. Analyze [Component Relationships](./COMPONENT_RELATIONSHIPS.md) for dependencies
3. Study [Weaviate Integration](./WEAVIATE_INTEGRATION.md) for scalability patterns
4. Examine [Data Flow Pipeline](./DATA_FLOW_PIPELINE.md) for bottleneck identification

### For Data Engineers
1. Focus on [Data Flow Pipeline](./DATA_FLOW_PIPELINE.md) for end-to-end understanding
2. Study [Weaviate Integration](./WEAVIATE_INTEGRATION.md) for data storage patterns
3. Review [System Architecture](./SYSTEM_ARCHITECTURE.md) for data source integrations

## Diagram Legend

All diagrams in this documentation use consistent styling:

- **Blue boxes** (#e3f2fd): Input sources, raw data
- **Green boxes** (#e8f5e9): Output results, completed stages
- **Purple boxes** (#f3e5f5): Databases, persistent storage
- **Orange boxes** (#fff3e0): Configuration, orchestration, decisions
- **Amber boxes** (#ffe0b2): Model artifacts, checkpoints
- **Red boxes** (#ffebee): Predictions, evaluation results, errors

## Mermaid Diagram Types Used

- **Flowcharts**: Process flows and decision trees
- **Sequence Diagrams**: Component interactions over time
- **Class Diagrams**: Object relationships and schema
- **State Diagrams**: Error handling and state transitions
- **Graph Diagrams**: Component dependencies
- **Sankey Diagrams**: Data volume flows

## Cross-References

### Related Documentation
- **[DVC Pipeline Reference](../../reference/pipelines/DVC_PIPELINE.md)** - Pipeline stage specifications
- **[How-To Guides](../../how-to/)** - Practical task instructions
- **[Tutorials](../../tutorials/)** - Step-by-step learning guides
- **[API Reference](../../reference/)** - Technical specifications

### External Resources
- **[Mermaid Documentation](https://mermaid.js.org/)** - Diagram syntax reference
- **[Weaviate Docs](https://weaviate.io/developers/weaviate)** - Vector database documentation
- **[DVC Documentation](https://dvc.org/doc)** - Pipeline management
- **[Hugging Face](https://huggingface.co/docs)** - Models and datasets

## Contributing

When adding or updating architecture documentation:

1. **Use Mermaid diagrams**: All diagrams should be in Mermaid format
2. **Follow style guide**: Use consistent colors and diagram types
3. **Maintain cross-references**: Link related documents
4. **Update this README**: Add new documents to the structure
5. **Test diagrams**: Ensure they render correctly in GitHub and MkDocs

See [Documentation Style Guide](../../reference/STYLE_GUIDE.md) for detailed standards.

## Questions?

- **Technical Issues**: [GitHub Issues](https://github.com/[org]/JuDDGES/issues)
- **Documentation Feedback**: Create an issue with label `documentation`
- **General Questions**: See [main documentation README](../../README.md)

---

**Last Updated**: 2025-10-11
**Version**: 1.0
**Maintainer**: JuDDGES Documentation Team