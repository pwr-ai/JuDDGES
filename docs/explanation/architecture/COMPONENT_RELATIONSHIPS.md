# Component Relationships and Dependencies

## Overview

This document visualizes the relationships between JuDDGES components, their dependencies, and interaction patterns. Understanding these relationships is crucial for development, debugging, and system maintenance.

## High-Level Component Architecture

```mermaid
graph TB
    subgraph "Core Components"
        subgraph "juddges/"
            Data["📁 data/<br/>Loaders & Database"]
            Embeddings["📁 embeddings/<br/>Vector Generation"]
            Models["📁 models/<br/>LLM Management"]
            Preprocessing["📁 preprocessing/<br/>Text Processing"]
            Evaluation["📁 evaluation/<br/>Metrics & Scoring"]
            Utils["📁 utils/<br/>Helpers & Tools"]
        end

        subgraph "scripts/"
            Dataset["📁 dataset/<br/>Dataset Building"]
            Embed["📁 embed/<br/>Embedding Scripts"]
            SFT["📁 sft/<br/>Fine-tuning Scripts"]
            Predict["📁 predict/<br/>Inference Scripts"]
            Eval["📁 evaluation/<br/>Eval Scripts"]
        end

        subgraph "configs/"
            ModelConf["📁 model/<br/>Model Configs"]
            DatasetConf["📁 dataset/<br/>Dataset Configs"]
            EmbedConf["📁 embedding_model/<br/>Embed Configs"]
            PipelineConf["📁 *.yaml<br/>Pipeline Configs"]
        end

        subgraph "External"
            Weaviate[("Weaviate DB")]
            HuggingFace[("🤗 Hugging Face")]
            DVC[("DVC Pipeline")]
        end
    end

    %% Dependencies
    Data --> Weaviate
    Data --> Preprocessing
    Preprocessing --> Embeddings
    Embeddings --> HuggingFace
    Models --> HuggingFace

    Dataset --> Data
    Dataset --> Preprocessing
    Embed --> Embeddings
    SFT --> Models
    Predict --> Models
    Predict --> Weaviate
    Eval --> Evaluation

    ModelConf --> Models
    DatasetConf --> Data
    EmbedConf --> Embeddings
    PipelineConf --> DVC

    DVC --> Dataset
    DVC --> Embed
    DVC --> SFT
    DVC --> Predict
    DVC --> Eval

    style Data fill:#e3f2fd
    style Models fill:#e8f5e9
    style Weaviate fill:#f3e5f5
    style DVC fill:#fff3e0
```

## Detailed Module Dependencies

```mermaid
graph LR
    subgraph "juddges.data"
        DataInit["__init__.py"]
        Loaders["loaders/<br/>• document_loader<br/>• parquet_loader"]
        Database["database/<br/>• weaviate_client<br/>• schema_manager"]
        DataUtils["utils/<br/>• uuid_generator<br/>• validators"]
    end

    subgraph "juddges.preprocessing"
        PrepInit["__init__.py"]
        Chunking["chunking/<br/>• smart_chunker<br/>• size_limiter"]
        Parsing["parsing/<br/>• text_parser<br/>• format_detector"]
        Cleaning["cleaning/<br/>• normalizer<br/>• encoder_fix"]
    end

    subgraph "juddges.embeddings"
        EmbedInit["__init__.py"]
        Generator["generator/<br/>• embedding_model<br/>• batch_processor"]
        Aggregator["aggregator/<br/>• mean_pooling<br/>• weighted_avg"]
        Storage["storage/<br/>• vector_store<br/>• cache_manager"]
    end

    subgraph "juddges.models"
        ModelInit["__init__.py"]
        Factory["factory/<br/>• model_factory<br/>• config_loader"]
        Inference["inference/<br/>• predictor<br/>• streamer"]
        Training["training/<br/>• trainer<br/>• lora_adapter"]
    end

    subgraph "juddges.evaluation"
        EvalInit["__init__.py"]
        Metrics["metrics/<br/>• ngram_metrics<br/>• semantic_metrics"]
        Judge["llm_judge/<br/>• gpt4_judge<br/>• claude_judge"]
        Analysis["analysis/<br/>• error_analysis<br/>• report_generator"]
    end

    %% Internal Dependencies
    Loaders --> Parsing
    Parsing --> Cleaning
    Cleaning --> Chunking
    Chunking --> Generator
    Generator --> Aggregator
    Aggregator --> Storage
    Storage --> Database

    Factory --> Training
    Factory --> Inference
    Inference --> Generator
    Inference --> Database

    Metrics --> Analysis
    Judge --> Analysis

    style DataInit fill:#e3f2fd
    style PrepInit fill:#e3f2fd
    style EmbedInit fill:#e3f2fd
    style ModelInit fill:#e3f2fd
    style EvalInit fill:#e3f2fd
```

## Class Relationships

```mermaid
classDiagram
    %% Data Classes
    class DocumentLoader {
        +load_pdf()
        +load_docx()
        +load_txt()
        +validate_format()
    }

    class WeaviateClient {
        +connect()
        +create_collection()
        +insert_batch()
        +search()
        +update()
    }

    %% Preprocessing Classes
    class TextChunker {
        +chunk_by_size()
        +chunk_by_tokens()
        +preserve_context()
    }

    class TextParser {
        +extract_text()
        +detect_language()
        +parse_metadata()
    }

    %% Embedding Classes
    class EmbeddingModel {
        +load_model()
        +encode()
        +batch_encode()
        +get_dim()
    }

    class VectorStore {
        +store_vectors()
        +retrieve_similar()
        +update_vectors()
    }

    %% Model Classes
    class ModelFactory {
        +create_model()
        +load_checkpoint()
        +get_config()
    }

    class Trainer {
        +train()
        +evaluate()
        +save_checkpoint()
    }

    class Predictor {
        +predict()
        +stream_predict()
        +batch_predict()
    }

    %% Evaluation Classes
    class MetricsCalculator {
        +calculate_bleu()
        +calculate_rouge()
        +calculate_bertscore()
    }

    class LLMJudge {
        +evaluate_quality()
        +score_response()
        +generate_feedback()
    }

    %% Relationships
    DocumentLoader --> TextParser : uses
    TextParser --> TextChunker : feeds
    TextChunker --> EmbeddingModel : provides text
    EmbeddingModel --> VectorStore : generates vectors
    VectorStore --> WeaviateClient : stores in

    ModelFactory --> Trainer : creates model for
    ModelFactory --> Predictor : creates model for
    Predictor --> WeaviateClient : retrieves context
    Predictor --> MetricsCalculator : sends predictions
    MetricsCalculator --> LLMJudge : validates with
```

## Data Flow Dependencies

```mermaid
flowchart TD
    subgraph "Input Layer"
        RawDocs["Raw Documents"]
        Config["Configuration Files"]
        PretrainedModels["Pretrained Models"]
    end

    subgraph "Processing Layer"
        subgraph "Data Pipeline"
            DocLoad["Document Loading"]
            TextProc["Text Processing"]
            ChunkGen["Chunk Generation"]
        end

        subgraph "Embedding Pipeline"
            EmbedGen["Embedding Generation"]
            VecStore["Vector Storage"]
        end

        subgraph "Training Pipeline"
            DataPrep["Data Preparation"]
            ModelTrain["Model Training"]
            CheckSave["Checkpoint Saving"]
        end
    end

    subgraph "Service Layer"
        WeaviateDB["Weaviate Service"]
        ModelServe["Model Service"]
        EvalServe["Evaluation Service"]
    end

    subgraph "Output Layer"
        Predictions["Predictions"]
        Reports["Evaluation Reports"]
        Artifacts["Model Artifacts"]
    end

    %% Flow
    RawDocs --> DocLoad
    Config --> DocLoad
    DocLoad --> TextProc
    TextProc --> ChunkGen

    ChunkGen --> EmbedGen
    PretrainedModels --> EmbedGen
    EmbedGen --> VecStore
    VecStore --> WeaviateDB

    ChunkGen --> DataPrep
    Config --> DataPrep
    DataPrep --> ModelTrain
    PretrainedModels --> ModelTrain
    ModelTrain --> CheckSave
    CheckSave --> ModelServe

    WeaviateDB --> ModelServe
    ModelServe --> Predictions
    Predictions --> EvalServe
    EvalServe --> Reports
    CheckSave --> Artifacts

    style RawDocs fill:#e3f2fd
    style Predictions fill:#e8f5e9
    style WeaviateDB fill:#f3e5f5
```

## Configuration Dependencies

```mermaid
graph TB
    subgraph "Configuration Hierarchy"
        Main["main.yaml<br/>Entry point"]

        subgraph "Hydra Composition"
            Defaults["defaults:<br/>- model: llama<br/>- dataset: pl-court<br/>- embedding: mmlw"]
        end

        subgraph "Model Configs"
            Llama["llama.yaml"]
            Mistral["mistral.yaml"]
            Bielik["bielik.yaml"]
        end

        subgraph "Dataset Configs"
            PLCourt["pl-court.yaml"]
            PLFrank["pl-frankowe.yaml"]
            ENLegal["en-legal.yaml"]
        end

        subgraph "Pipeline Configs"
            SFTConf["sft_config.yaml"]
            PredictConf["predict_config.yaml"]
            EvalConf["evaluate_config.yaml"]
        end

        Runtime["Runtime Configuration<br/>Merged settings"]
    end

    Main --> Defaults
    Defaults --> Llama
    Defaults --> PLCourt
    Defaults --> SFTConf

    Llama --> Runtime
    Mistral --> Runtime
    Bielik --> Runtime
    PLCourt --> Runtime
    PLFrank --> Runtime
    ENLegal --> Runtime
    SFTConf --> Runtime
    PredictConf --> Runtime
    EvalConf --> Runtime

    style Main fill:#fff3e0
    style Runtime fill:#e8f5e9
```

## Error Propagation Paths

```mermaid
graph TD
    subgraph "Error Sources"
        DataError["Data Error<br/>• Invalid format<br/>• Missing fields"]
        ModelError["Model Error<br/>• OOM<br/>• Loading failure"]
        DBError["Database Error<br/>• Connection<br/>• Schema mismatch"]
        ConfigError["Config Error<br/>• Invalid params<br/>• Missing files"]
    end

    subgraph "Error Handlers"
        DataHandler["Data Handler<br/>• Validation<br/>• Fallback"]
        ModelHandler["Model Handler<br/>• Retry logic<br/>• Graceful degradation"]
        DBHandler["DB Handler<br/>• Reconnection<br/>• Cache fallback"]
        ConfigHandler["Config Handler<br/>• Defaults<br/>• Validation"]
    end

    subgraph "Recovery Actions"
        Skip["Skip Record"]
        Retry["Retry Operation"]
        Fallback["Use Fallback"]
        Alert["Send Alert"]
        Terminate["Terminate Pipeline"]
    end

    DataError --> DataHandler
    ModelError --> ModelHandler
    DBError --> DBHandler
    ConfigError --> ConfigHandler

    DataHandler --> Skip
    DataHandler --> Retry
    ModelHandler --> Fallback
    ModelHandler --> Retry
    DBHandler --> Retry
    DBHandler --> Alert
    ConfigHandler --> Fallback
    ConfigHandler --> Terminate

    style DataError fill:#ffebee
    style ModelError fill:#ffebee
    style DBError fill:#ffebee
    style ConfigError fill:#ffebee
    style Alert fill:#fff3e0
```

## Testing Dependencies

```mermaid
graph LR
    subgraph "Test Structure"
        subgraph "Unit Tests"
            TestData["test_data/<br/>• Loaders<br/>• Validators"]
            TestEmbed["test_embeddings/<br/>• Generation<br/>• Storage"]
            TestModels["test_models/<br/>• Factory<br/>• Inference"]
            TestEval["test_evaluation/<br/>• Metrics<br/>• Judge"]
        end

        subgraph "Integration Tests"
            TestWeaviate["test_weaviate/<br/>• Connection<br/>• CRUD ops"]
            TestPipeline["test_pipeline/<br/>• End-to-end<br/>• DVC stages"]
        end

        subgraph "Test Fixtures"
            SampleData["sample_data/<br/>• Mock documents<br/>• Test configs"]
            MockModels["mock_models/<br/>• Dummy weights<br/>• Stubs"]
        end
    end

    TestData --> SampleData
    TestEmbed --> SampleData
    TestModels --> MockModels
    TestEval --> SampleData

    TestWeaviate --> SampleData
    TestPipeline --> SampleData
    TestPipeline --> MockModels

    style TestData fill:#e3f2fd
    style TestWeaviate fill:#e8f5e9
    style SampleData fill:#fff3e0
```

## Package Dependencies

```mermaid
graph TB
    subgraph "Core Dependencies"
        Python["Python 3.10+"]
        PyTorch["PyTorch 2.0+"]
        Transformers["Transformers 4.40+"]
    end

    subgraph "ML Dependencies"
        Unsloth["Unsloth<br/>Training optimization"]
        PEFT["PEFT<br/>LoRA/QLoRA"]
        BitsBytes["BitsAndBytes<br/>Quantization"]
    end

    subgraph "Data Dependencies"
        Pandas["Pandas<br/>Data manipulation"]
        Pyarrow["PyArrow<br/>Parquet support"]
        Weaviate["Weaviate Client<br/>Vector DB"]
    end

    subgraph "Infrastructure"
        DVC_["DVC<br/>Pipeline management"]
        Hydra["Hydra<br/>Configuration"]
        Docker["Docker<br/>Containerization"]
    end

    subgraph "Utilities"
        Rich["Rich<br/>Console output"]
        Loguru["Loguru<br/>Logging"]
        Typer["Typer<br/>CLI interface"]
    end

    Python --> PyTorch
    Python --> Transformers
    PyTorch --> Unsloth
    PyTorch --> PEFT
    PyTorch --> BitsBytes

    Python --> Pandas
    Python --> Pyarrow
    Python --> Weaviate

    Python --> DVC_
    Python --> Hydra
    Python --> Docker

    Python --> Rich
    Python --> Loguru
    Python --> Typer

    style Python fill:#fff3e0
    style PyTorch fill:#e3f2fd
    style DVC_ fill:#e8f5e9
```

## Communication Patterns

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant DVC
    participant Pipeline
    participant Weaviate
    participant Model
    participant Evaluator

    User->>CLI: Execute command
    CLI->>DVC: Trigger pipeline
    DVC->>Pipeline: Run stages

    Pipeline->>Weaviate: Store embeddings
    Pipeline->>Model: Load weights
    Pipeline->>Model: Fine-tune

    Model->>Weaviate: Query context
    Weaviate-->>Model: Return results
    Model->>Model: Generate prediction

    Model->>Evaluator: Send predictions
    Evaluator->>Evaluator: Calculate metrics
    Evaluator-->>User: Return report
```

## Best Practices for Component Design

1. **Loose Coupling**: Components communicate through interfaces
2. **High Cohesion**: Related functionality grouped together
3. **Single Responsibility**: Each component has one clear purpose
4. **Dependency Injection**: Configuration passed at runtime
5. **Error Isolation**: Failures don't cascade unnecessarily
6. **Testability**: Components can be tested in isolation
7. **Documentation**: Clear interfaces and usage examples

## Related Documentation

- [System Architecture](./SYSTEM_ARCHITECTURE.md)
- [Data Flow Pipeline](./DATA_FLOW_PIPELINE.md)
- [DVC Pipeline](../../reference/pipelines/DVC_PIPELINE.md)
- [Development Guide](../../how-to/development/setup.md)
