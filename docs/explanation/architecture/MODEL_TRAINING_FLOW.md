# Model Training and Inference Flow

## Overview

This document visualizes the complete model training and inference workflows in JuDDGES, including fine-tuning strategies, optimization techniques, and deployment patterns for legal document processing.

## Training Architecture

```mermaid
graph TB
    subgraph "Training Pipeline"
        subgraph "Data Preparation"
            RawData[("Raw Legal Data<br/>Parquet Files")]
            InstBuilder["Instruction Builder<br/>• Templates<br/>• Context formatting"]
            TrainSplit["Train/Val Split<br/>80/20"]
        end

        subgraph "Model Initialization"
            BaseModel["Base Model<br/>• Llama 3.1/3.2<br/>• Mistral<br/>• Bielik<br/>• Phi-4"]
            LoRAConfig["LoRA Configuration<br/>• r=16<br/>• alpha=32<br/>• dropout=0.05"]
            Quantization["Quantization<br/>• 4-bit/8-bit<br/>• BitsAndBytes"]
        end

        subgraph "Training Loop"
            DataLoader["DataLoader<br/>• Batch size: 4<br/>• Gradient accumulation: 4"]
            Forward["Forward Pass<br/>• Attention mechanism<br/>• Loss calculation"]
            Backward["Backward Pass<br/>• Gradient computation<br/>• LoRA updates"]
            Optimizer["Optimizer<br/>• AdamW<br/>• LR: 2e-4<br/>• Warmup: 0.1"]
        end

        subgraph "Checkpointing"
            SaveCheck["Save Checkpoint<br/>• Best model<br/>• Every N steps"]
            EarlySto["Early Stopping<br/>• Patience: 3<br/>• Monitor: val_loss"]
        end

        subgraph "Output"
            FinalModel[("Fine-tuned Model<br/>• LoRA weights<br/>• Config files")]
            Metrics[("Training Metrics<br/>• Loss curves<br/>• Learning rate")]
        end
    end

    RawData --> InstBuilder
    InstBuilder --> TrainSplit
    TrainSplit --> DataLoader

    BaseModel --> LoRAConfig
    LoRAConfig --> Quantization
    Quantization --> Forward

    DataLoader --> Forward
    Forward --> Backward
    Backward --> Optimizer
    Optimizer --> SaveCheck
    Optimizer --> Forward

    SaveCheck --> EarlySto
    EarlySto--> FinalModel
    SaveCheck --> Metrics

    style RawData fill:#e3f2fd
    style FinalModel fill:#e8f5e9
    style Metrics fill:#fff3e0
```

## Fine-Tuning Strategy: PEFT/LoRA

```mermaid
graph LR
    subgraph "Parameter Efficient Fine-Tuning"
        subgraph "Original Model"
            OrigWeight["Original Weights<br/>Frozen"]
            OrigSize["Size: 7B-70B params"]
        end

        subgraph "LoRA Adaptation"
            LoRAA["Matrix A<br/>r × d"]
            LoRAB["Matrix B<br/>d × r"]
            Rank["Rank r=16<br/>Compression"]
        end

        subgraph "Training"
            TrainableParams["Trainable<br/>~0.1% params"]
            MemoryUsage["Memory<br/>40GB → 16GB"]
        end

        Output["Fine-tuned Output<br/>W + BA"]
    end

    OrigWeight --> Output
    LoRAA --> LoRAB
    LoRAB --> Output
    Rank --> LoRAA
    Rank --> LoRAB
    LoRAA --> TrainableParams
    LoRAB --> TrainableParams
    TrainableParams --> MemoryUsage

    style OrigWeight fill:#f3e5f5
    style TrainableParams fill:#e8f5e9
    style Output fill:#fff3e0
```

## Multi-Model Training Matrix

```mermaid
flowchart TD
    subgraph "Model-Dataset Matrix"
        subgraph "Models"
            Llama32["Llama-3.2-3B<br/>Compact, Fast"]
            Llama31["Llama-3.1-8B<br/>Balanced"]
            Mistral["Mistral-7B-v0.3<br/>Efficient"]
            Bielik["Bielik-7B<br/>Polish-optimized"]
            Phi["Phi-4<br/>Small, Capable"]
        end

        subgraph "Datasets"
            PLCourt["pl-court-instruct<br/>Polish courts"]
            PLFrank["pl-court-frankowe<br/>Swiss franc loans"]
            ENLegal["en-legal-instruct<br/>English legal"]
        end

        subgraph "Training Configs"
            Config1["Config A<br/>Quick training<br/>3 epochs"]
            Config2["Config B<br/>Full training<br/>5 epochs"]
            Config3["Config C<br/>Specialized<br/>Domain-specific"]
        end

        Orchestrator["DVC Orchestrator<br/>Parallel execution"]

        Results[("Training Results<br/>15 model variants")]
    end

    Llama32 --> Orchestrator
    Llama31 --> Orchestrator
    Mistral --> Orchestrator
    Bielik --> Orchestrator
    Phi --> Orchestrator

    PLCourt --> Orchestrator
    PLFrank --> Orchestrator
    ENLegal --> Orchestrator

    Config1 --> Orchestrator
    Config2 --> Orchestrator
    Config3 --> Orchestrator

    Orchestrator --> Results

    style Orchestrator fill:#fff3e0
    style Results fill:#e8f5e9
```

## Inference Pipeline

```mermaid
flowchart LR
    subgraph "Inference Flow"
        Input["User Query<br/>Legal question"]

        subgraph "Context Retrieval"
            Embed["Query Embedding"]
            Weaviate["Weaviate Search<br/>Top-k chunks"]
            Context["Retrieved Context<br/>Relevant documents"]
        end

        subgraph "Prompt Engineering"
            Template["Prompt Template<br/>System + User + Context"]
            Tokenize["Tokenization<br/>Model-specific"]
            Truncate["Truncation<br/>Max context window"]
        end

        subgraph "Model Inference"
            LoadModel["Load Model<br/>+ LoRA weights"]
            Generate["Generation<br/>• Sampling<br/>• Temperature<br/>• Top-p"]
            Stream["Streaming<br/>Token by token"]
        end

        subgraph "Post-Processing"
            Parse["Parse Output<br/>Extract fields"]
            Validate["Validate<br/>Schema check"]
            Format["Format Response<br/>JSON/Text"]
        end

        Output["Final Response<br/>Structured output"]
    end

    Input --> Embed
    Embed --> Weaviate
    Weaviate --> Context
    Context --> Template
    Template --> Tokenize
    Tokenize --> Truncate
    Truncate --> LoadModel
    LoadModel --> Generate
    Generate --> Stream
    Stream --> Parse
    Parse --> Validate
    Validate --> Format
    Format --> Output

    style Input fill:#e3f2fd
    style Output fill:#e8f5e9
    style Weaviate fill:#f3e5f5
```

## Training Optimization Techniques

```mermaid
graph TB
    subgraph "Optimization Strategies"
        subgraph "Memory Optimization"
            GradAccum["Gradient Accumulation<br/>Effective batch size: 16"]
            MixedPrec["Mixed Precision<br/>FP16/BF16 training"]
            GradCheck["Gradient Checkpointing<br/>Trade compute for memory"]
            CPUOffload["CPU Offloading<br/>Optimizer states"]
        end

        subgraph "Speed Optimization"
            FlashAttn["Flash Attention 2<br/>2-3x speedup"]
            Unsloth["Unsloth<br/>2x faster training"]
            DataParallel["Data Parallelism<br/>Multi-GPU"]
            CompileMode["Torch Compile<br/>Graph optimization"]
        end

        subgraph "Quality Optimization"
            LRSchedule["Learning Rate Schedule<br/>Cosine with warmup"]
            RegTech["Regularization<br/>• Dropout<br/>• Weight decay"]
            AugData["Data Augmentation<br/>• Paraphrasing<br/>• Back-translation"]
            CurrLearn["Curriculum Learning<br/>Easy → Hard samples"]
        end
    end

    style GradAccum fill:#e3f2fd
    style FlashAttn fill:#e8f5e9
    style LRSchedule fill:#fff3e0
```

## Model Evaluation Flow

```mermaid
sequenceDiagram
    participant Dataset
    participant Model
    participant Evaluator
    participant Metrics
    participant Report

    Dataset->>Model: Test samples
    Model->>Model: Generate predictions

    loop For each prediction
        Model->>Evaluator: (prediction, reference)
        Evaluator->>Evaluator: Calculate metrics

        alt N-gram Metrics
            Evaluator->>Metrics: BLEU score
            Evaluator->>Metrics: ROUGE score
            Evaluator->>Metrics: METEOR score
        else Semantic Metrics
            Evaluator->>Metrics: BERTScore
            Evaluator->>Metrics: Embedding similarity
        else LLM Judge
            Evaluator->>Metrics: GPT-4 evaluation
            Evaluator->>Metrics: Quality score
        end
    end

    Metrics->>Report: Aggregate results
    Report->>Report: Generate visualizations
    Report->>Report: Statistical analysis
```

## Deployment Strategies

```mermaid
graph TB
    subgraph "Deployment Options"
        subgraph "Local Deployment"
            LocalGPU["Single GPU<br/>• Development<br/>• Testing"]
            MultiGPU["Multi-GPU<br/>• Production<br/>• High throughput"]
        end

        subgraph "Cloud Deployment"
            CloudGPU["Cloud GPU<br/>• AWS/GCP/Azure<br/>• Auto-scaling"]
            Serverless["Serverless<br/>• AWS Lambda<br/>• Function as Service"]
        end

        subgraph "Edge Deployment"
            Quantized["Quantized Model<br/>• 4-bit/8-bit<br/>• Mobile/Edge"]
            ONNX["ONNX Export<br/>• Cross-platform<br/>• Optimized"]
        end

        subgraph "API Serving"
            REST["REST API<br/>• FastAPI<br/>• Load balancing"]
            Streaming["Streaming API<br/>• WebSocket<br/>• Real-time"]
            Batch["Batch API<br/>• Async processing<br/>• Queue-based"]
        end
    end

    LocalGPU --> REST
    MultiGPU --> REST
    CloudGPU --> Streaming
    Serverless --> Batch
    Quantized --> ONNX
    ONNX --> REST

    style CloudGPU fill:#e3f2fd
    style REST fill:#e8f5e9
    style Quantized fill:#fff3e0
```

## Training Monitoring Dashboard

```mermaid
graph LR
    subgraph "Training Metrics"
        subgraph "Loss Tracking"
            TrainLoss["Training Loss<br/>Per batch/epoch"]
            ValLoss["Validation Loss<br/>Early stopping"]
        end

        subgraph "Performance"
            LearningRate["Learning Rate<br/>Schedule tracking"]
            GradNorm["Gradient Norm<br/>Stability check"]
            Memory["Memory Usage<br/>GPU utilization"]
        end

        subgraph "Quality"
            BLEU["BLEU Score<br/>Translation quality"]
            Perplexity["Perplexity<br/>Model confidence"]
            Accuracy["Task Accuracy<br/>Domain-specific"]
        end

        TensorBoard["TensorBoard<br/>Visualization"]
        WandB["Weights & Biases<br/>Experiment tracking"]
    end

    TrainLoss --> TensorBoard
    ValLoss --> TensorBoard
    LearningRate --> WandB
    GradNorm --> WandB
    Memory --> TensorBoard
    BLEU --> WandB
    Perplexity --> WandB
    Accuracy --> WandB

    style TensorBoard fill:#e8f5e9
    style WandB fill:#fff3e0
```

## Hardware Requirements

```mermaid
graph TB
    subgraph "GPU Requirements by Model Size"
        subgraph "Development"
            Dev3B["3B Models<br/>• RTX 3090 (24GB)<br/>• Batch size: 4"]
            Dev7B["7B Models<br/>• RTX 4090 (24GB)<br/>• Batch size: 2"]
        end

        subgraph "Production"
            Prod7B["7B Models<br/>• A100 (40GB)<br/>• Batch size: 8"]
            Prod13B["13B Models<br/>• A100 (80GB)<br/>• Batch size: 4"]
            Prod70B["70B Models<br/>• 4×A100 (80GB)<br/>• Model parallel"]
        end

        subgraph "Optimization"
            Quant["Quantization<br/>• 4-bit: 75% reduction<br/>• 8-bit: 50% reduction"]
            LoRA_["LoRA<br/>• 90% param reduction<br/>• Full performance"]
        end
    end

    Dev3B --> Quant
    Dev7B --> Quant
    Prod7B --> LoRA_
    Prod13B --> LoRA_
    Prod70B --> Quant

    style Dev3B fill:#e3f2fd
    style Prod7B fill:#e8f5e9
    style Quant fill:#fff3e0
```

## Best Practices

1. **Learning Rate**: Start with 2e-4, use cosine schedule with warmup
2. **Batch Size**: Maximize GPU memory usage with gradient accumulation
3. **Evaluation**: Evaluate every 500 steps to catch overfitting early
4. **Checkpointing**: Save best model and regular checkpoints
5. **Mixed Precision**: Use FP16/BF16 for 2x memory savings
6. **Data Quality**: Clean and validate training data thoroughly
7. **Monitoring**: Track all metrics for experiment reproducibility

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce batch size, enable gradient checkpointing |
| Slow Training | Enable Flash Attention, use Unsloth |
| Overfitting | Increase dropout, add regularization |
| Poor Quality | Check data quality, adjust learning rate |
| Unstable Training | Reduce learning rate, check gradient norms |

## Related Documentation

- [System Architecture](./SYSTEM_ARCHITECTURE.md)
- [DVC Pipeline](../../reference/pipelines/DVC_PIPELINE.md)
- [Fine-Tuning Guide](../../how-to/training/fine_tuning.md)
- [Model Configuration](../../reference/configs/model_configs.md)