# DVC Pipeline Architecture Reference

## Overview

This reference document details the DVC (Data Version Control) pipeline architecture used in JuDDGES for reproducible machine learning workflows. DVC orchestrates complex multi-stage pipelines with automatic dependency resolution and caching.

## Pipeline DAG (Directed Acyclic Graph)

```mermaid
graph TD
    subgraph "DVC Pipeline Stages"
        Start([Start])

        %% Data Preparation
        DataPrep["📊 Data Preparation<br/>Convert raw data to Parquet"]

        %% Embedding Stage
        Embed["🔤 embed<br/>Generate text embeddings<br/>mmlw-roberta-large"]

        %% Training Stages
        SFT["🎓 sft<br/>Supervised Fine-Tuning<br/>PEFT/LoRA"]

        %% Prediction Stage
        Predict["🔮 predict<br/>Generate predictions<br/>Batch inference"]

        %% Evaluation Stages
        Evaluate["📏 evaluate<br/>N-gram metrics<br/>BLEU/ROUGE"]
        EvaluateLLM["🤖 evaluate_llm_as_judge<br/>LLM-based evaluation<br/>Quality assessment"]

        %% Matrix Expansion
        Matrix{{"Matrix Expansion<br/>Models × Datasets"}}

        End([End])
    end

    %% Dependencies
    Start --> DataPrep
    DataPrep --> Embed
    DataPrep --> SFT
    Embed --> Predict
    SFT --> Predict
    Predict --> Evaluate
    Predict --> EvaluateLLM
    Evaluate --> End
    EvaluateLLM --> End

    %% Matrix connections
    SFT -.->|foreach| Matrix
    Predict -.->|foreach| Matrix
    Matrix -.-> SFT
    Matrix -.-> Predict

    style Start fill:#e8f5e9
    style End fill:#ffebee
    style Matrix fill:#fff3e0
    style Embed fill:#e3f2fd
    style SFT fill:#f3e5f5
    style Predict fill:#ffe0b2
    style Evaluate fill:#fce4ec
    style EvaluateLLM fill:#fce4ec
```

## Stage Specifications

### 1. Embedding Generation (`embed`)

```mermaid
flowchart LR
    subgraph "embed Stage"
        Input[("Input<br/>Parquet files")]
        Process["Process<br/>• Load documents<br/>• Tokenize text<br/>• Generate embeddings<br/>• Aggregate vectors"]
        Output[("Output<br/>Embedding vectors")]

        Config["Config<br/>• Model: mmlw-roberta-large<br/>• Batch size: 32<br/>• Max length: 512"]
    end

    Input --> Process
    Config --> Process
    Process --> Output

    style Input fill:#e3f2fd
    style Output fill:#e8f5e9
    style Config fill:#fff3e0
```

**Command**: `dvc repro embed`

**Dependencies**:

- Input: `data/datasets/{pl,en}/raw/*.parquet`
- Model: `sdadas/mmlw-roberta-large`
- Output: Embedding files in `data/embeddings/`

**Parameters**:

```yaml
embedding_model:
  name: mmlw-roberta-large
  batch_size: 32
  max_length: 512
  device: cuda
```

### 2. Supervised Fine-Tuning (`sft`)

```mermaid
flowchart TB
    subgraph "sft Stage Matrix"
        subgraph "Models"
            M1["Llama-3.2-3B"]
            M2["Mistral-7B-v0.3"]
            M3["Bielik-7B-v0.1"]
            M4["Phi-4"]
        end

        subgraph "Datasets"
            D1["pl-court-instruct"]
            D2["pl-court-frankowe"]
            D3["en-legal-instruct"]
        end

        subgraph "Training Config"
            TC["• PEFT/LoRA<br/>• Gradient Accumulation: 4<br/>• Learning Rate: 2e-4<br/>• Epochs: 3<br/>• Warmup: 0.1"]
        end

        Process["Fine-Tuning Process<br/>Unsloth Optimization"]

        Output[("Checkpoints<br/>models/{model}/{dataset}/")]
    end

    M1 --> Process
    M2 --> Process
    M3 --> Process
    M4 --> Process

    D1 --> Process
    D2 --> Process
    D3 --> Process

    TC --> Process
    Process --> Output

    style Output fill:#ffe0b2
```

**Command**: `CUDA_VISIBLE_DEVICES=0 NUM_PROC=10 dvc repro sft`

**Matrix Expansion**:

```yaml
stages:
  sft:
    foreach:
      - model: Llama-3.2-3B-Instruct
        dataset: pl-court-instruct-sft
      - model: Mistral-7B-Instruct-v0.3
        dataset: pl-court-instruct-sft
      - model: Bielik-7B-Instruct-v0.1
        dataset: pl-court-frankowe-instruct
```

### 3. Prediction (`predict`)

```mermaid
flowchart LR
    subgraph "predict Stage"
        Input1[("Fine-tuned Model")]
        Input2[("Test Dataset")]
        Input3[("Weaviate Context")]

        Process["Inference Pipeline<br/>• Load model<br/>• Retrieve context<br/>• Generate predictions<br/>• Post-process"]

        Config["Config<br/>• Batch size: 8<br/>• Max new tokens: 512<br/>• Temperature: 0.7<br/>• Top-p: 0.95"]

        Output[("Predictions<br/>JSON/Parquet")]
    end

    Input1 --> Process
    Input2 --> Process
    Input3 --> Process
    Config --> Process
    Process --> Output

    style Input1 fill:#ffe0b2
    style Input2 fill:#e3f2fd
    style Input3 fill:#e8f5e9
    style Output fill:#ffebee
```

**Command**: `CUDA_VISIBLE_DEVICES=0 dvc repro predict`

### 4. Evaluation (`evaluate` & `evaluate_llm_as_judge`)

```mermaid
flowchart TB
    subgraph "Evaluation Pipeline"
        Predictions[("Model Predictions")]

        subgraph "N-gram Metrics"
            BLEU["BLEU Score"]
            ROUGE["ROUGE Score"]
            METEOR["METEOR Score"]
            BertScore["BERTScore"]
        end

        subgraph "LLM as Judge"
            Judge["GPT-4/Claude<br/>Quality Assessment"]
            Criteria["• Accuracy<br/>• Completeness<br/>• Relevance<br/>• Coherence"]
        end

        Reports[("Evaluation Reports<br/>• Metrics JSON<br/>• Analysis HTML")]
    end

    Predictions --> BLEU
    Predictions --> ROUGE
    Predictions --> METEOR
    Predictions --> BertScore

    Predictions --> Judge
    Criteria --> Judge

    BLEU --> Reports
    ROUGE --> Reports
    METEOR --> Reports
    BertScore --> Reports
    Judge --> Reports

    style Predictions fill:#ffebee
    style Reports fill:#e8f5e9
```

## DVC Configuration Structure

```mermaid
graph TD
    subgraph "DVC Configuration Files"
        Root["dvc.yaml<br/>Pipeline definition"]

        Params["params.yaml<br/>Global parameters"]

        subgraph "Stage Configs"
            SC1["configs/sft_config.yaml"]
            SC2["configs/predict_config.yaml"]
            SC3["configs/evaluate_config.yaml"]
        end

        subgraph "Model Configs"
            MC1["configs/model/Llama-*.yaml"]
            MC2["configs/model/Mistral-*.yaml"]
            MC3["configs/model/Bielik-*.yaml"]
            MC4["configs/model/Phi-*.yaml"]
        end

        subgraph "Dataset Configs"
            DC1["configs/dataset/pl-court-*.yaml"]
            DC2["configs/dataset/en-legal-*.yaml"]
        end
    end

    Root --> Params
    Root --> SC1
    Root --> SC2
    Root --> SC3

    SC1 --> MC1
    SC1 --> MC2
    SC1 --> MC3
    SC1 --> MC4

    SC1 --> DC1
    SC1 --> DC2

    style Root fill:#fff3e0
    style Params fill:#e3f2fd
```

## Pipeline Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant DVC
    participant Git
    participant Cache
    participant GPU

    User->>DVC: dvc repro predict
    DVC->>Git: Check pipeline definition
    DVC->>Cache: Check cached outputs

    alt Output exists in cache
        Cache-->>DVC: Return cached result
        DVC-->>User: Pipeline up to date
    else Output not cached
        DVC->>GPU: Allocate resources
        DVC->>DVC: Execute pipeline stages

        loop For each stage
            DVC->>DVC: Check dependencies
            DVC->>GPU: Run computation
            GPU-->>DVC: Return output
            DVC->>Cache: Store output
        end

        DVC-->>User: Pipeline complete
    end
```

## Matrix Execution Strategy

```mermaid
graph LR
    subgraph "Matrix Definition"
        Config["dvc.yaml<br/>foreach definition"]
    end

    subgraph "Expansion"
        Expand["Cartesian Product<br/>Models × Datasets"]
    end

    subgraph "Parallel Execution"
        PE1["Llama + Dataset1"]
        PE2["Llama + Dataset2"]
        PE3["Mistral + Dataset1"]
        PE4["Mistral + Dataset2"]
        PE5["...more combinations"]
    end

    subgraph "Results"
        Results["Aggregated Results<br/>Per combination"]
    end

    Config --> Expand
    Expand --> PE1
    Expand --> PE2
    Expand --> PE3
    Expand --> PE4
    Expand --> PE5

    PE1 --> Results
    PE2 --> Results
    PE3 --> Results
    PE4 --> Results
    PE5 --> Results

    style Config fill:#fff3e0
    style Results fill:#e8f5e9
```

## Command Reference

| Command | Description | Environment Variables |
|---------|-------------|----------------------|
| `dvc repro` | Run entire pipeline | - |
| `dvc repro embed` | Generate embeddings only | `NUM_PROC` |
| `dvc repro sft` | Run fine-tuning | `CUDA_VISIBLE_DEVICES`, `NUM_PROC` |
| `dvc repro predict` | Generate predictions | `CUDA_VISIBLE_DEVICES`, `NUM_PROC` |
| `dvc repro evaluate` | Run n-gram evaluation | - |
| `dvc repro evaluate_llm_as_judge` | Run LLM evaluation | - |
| `dvc dag` | Visualize pipeline DAG | - |
| `dvc status` | Check pipeline status | - |
| `dvc stage list` | List all stages | - |

## Cache Management

```mermaid
flowchart TB
    subgraph "DVC Cache Structure"
        LocalCache[".dvc/cache/<br/>Local cache"]
        RemoteCache["Remote Storage<br/>S3/GCS/Azure"]

        subgraph "Cache Keys"
            InputHash["Input file hash"]
            ParamHash["Parameter hash"]
            CodeHash["Code hash"]
            Combined["Combined hash<br/>→ Cache key"]
        end
    end

    InputHash --> Combined
    ParamHash --> Combined
    CodeHash --> Combined
    Combined --> LocalCache
    LocalCache <--> RemoteCache

    style LocalCache fill:#e3f2fd
    style RemoteCache fill:#e8f5e9
    style Combined fill:#fff3e0
```

## Performance Optimization

1. **Stage-level Caching**: Reuse outputs when inputs haven't changed
2. **Parallel Execution**: Run independent stages concurrently
3. **Resource Allocation**: Configure GPU/CPU per stage
4. **Batch Processing**: Optimize batch sizes for memory/speed
5. **Incremental Updates**: Only rerun affected stages

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size in config |
| Stage fails | Check `dvc.log` for details |
| Cache miss | Run `dvc status` to check changes |
| Slow execution | Enable parallel processing with `NUM_PROC` |
| GPU not used | Set `CUDA_VISIBLE_DEVICES` |

## Related Documentation

- [System Architecture](../../explanation/architecture/SYSTEM_ARCHITECTURE.md)
- [Data Flow Pipeline](../../explanation/architecture/DATA_FLOW_PIPELINE.md)
- [Model Training Guide](../../how-to/training/fine_tuning.md)
