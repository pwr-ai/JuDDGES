# Pipeline Reference Documentation

This directory contains technical reference documentation for all pipelines in the JuDDGES system, focusing on DVC-managed workflows and their configurations.

## Overview

JuDDGES uses Data Version Control (DVC) to manage complex machine learning pipelines with automatic dependency resolution, caching, and reproducibility. This reference provides detailed specifications for each pipeline stage.

## Documentation Structure

### [DVC Pipeline Architecture](./DVC_PIPELINE.md)

**Purpose**: Complete reference for DVC pipeline stages, configurations, and execution

**Contents**:
- Pipeline DAG (Directed Acyclic Graph) visualization
- Detailed stage specifications (embed, sft, predict, evaluate)
- Matrix expansion strategy for multi-model training
- Configuration file structure and hierarchy
- Command reference and environment variables
- Cache management and performance optimization
- Troubleshooting guide

**Diagrams Include**:
- Pipeline DAG showing all stages and dependencies
- Stage-specific flowcharts (embed, sft, predict, evaluate)
- Configuration hierarchy
- Matrix execution strategy
- Cache structure
- Pipeline execution flow (sequence diagram)

**Best For**:
- Understanding pipeline stages and their dependencies
- Configuring pipeline runs
- Debugging pipeline issues
- Optimizing pipeline performance
- Setting up CI/CD integration

---

## Pipeline Stages Reference

### 1. Embedding Generation (`embed`)

**Purpose**: Generate vector embeddings for legal documents

**Input**:
- Parquet files in `data/datasets/{pl,en}/raw/`
- Pre-trained model: `sdadas/mmlw-roberta-large`

**Output**:
- Embedding files in `data/embeddings/`
- Ready for Weaviate ingestion

**Key Parameters**:
```yaml
embedding_model:
  name: mmlw-roberta-large
  batch_size: 32
  max_length: 512
  device: cuda
```

**Execution**:
```bash
dvc repro embed
```

---

### 2. Supervised Fine-Tuning (`sft`)

**Purpose**: Fine-tune language models on legal instruction datasets

**Input**:
- Instruction datasets (Q&A format)
- Pre-trained base models (Llama, Mistral, Bielik, Phi)
- Fine-tuning configuration

**Output**:
- Model checkpoints in `models/{model}/{dataset}/`
- Training metrics and logs

**Key Parameters**:
```yaml
training:
  peft_type: lora
  lora_r: 16
  lora_alpha: 32
  learning_rate: 2e-4
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
```

**Execution**:
```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROC=10 dvc repro sft
```

**Matrix Expansion**: Automatically runs for all model/dataset combinations defined in `dvc.yaml`

---

### 3. Prediction (`predict`)

**Purpose**: Generate predictions using fine-tuned models

**Input**:
- Fine-tuned model checkpoints
- Test datasets
- Weaviate context (for RAG)

**Output**:
- Predictions in JSON/Parquet format
- Stored in `outputs/predictions/`

**Key Parameters**:
```yaml
inference:
  batch_size: 8
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.95
  do_sample: true
```

**Execution**:
```bash
CUDA_VISIBLE_DEVICES=0 dvc repro predict
```

---

### 4. N-gram Evaluation (`evaluate`)

**Purpose**: Calculate traditional metrics (BLEU, ROUGE, METEOR)

**Input**:
- Model predictions
- Reference answers

**Output**:
- Metrics JSON files
- Statistical analysis reports

**Execution**:
```bash
dvc repro evaluate
```

---

### 5. LLM-as-Judge Evaluation (`evaluate_llm_as_judge`)

**Purpose**: Qualitative evaluation using large language models

**Input**:
- Model predictions
- Evaluation criteria
- Judge model (GPT-4, Claude)

**Output**:
- Quality scores
- Detailed feedback
- Evaluation reports

**Execution**:
```bash
dvc repro evaluate_llm_as_judge
```

---

## Configuration Reference

### Configuration File Hierarchy

```
configs/
├── main.yaml                    # Entry point with defaults
├── sft_config.yaml             # Fine-tuning configuration
├── predict_config.yaml         # Inference configuration
├── evaluate_config.yaml        # Evaluation configuration
├── model/                      # Model-specific configs
│   ├── Llama-3.2-3B.yaml
│   ├── Mistral-7B-v0.3.yaml
│   ├── Bielik-7B-v0.1.yaml
│   └── Phi-4.yaml
├── dataset/                    # Dataset-specific configs
│   ├── pl-court-instruct.yaml
│   ├── pl-court-frankowe.yaml
│   └── en-legal-instruct.yaml
└── embedding_model/            # Embedding configs
    └── mmlw-roberta-large.yaml
```

### Hydra Composition

JuDDGES uses Hydra for hierarchical configuration:

```yaml
defaults:
  - model: Llama-3.2-3B-Instruct
  - dataset: pl-court-instruct-sft
  - embedding_model: mmlw-roberta-large
  - _self_

# Override with command line
# python script.py model=Mistral-7B dataset=pl-court-frankowe
```

---

## Command Reference

### Basic Commands

| Command | Description | Required ENV |
|---------|-------------|--------------|
| `dvc repro` | Run entire pipeline | - |
| `dvc repro <stage>` | Run specific stage | Varies |
| `dvc dag` | Visualize pipeline | - |
| `dvc status` | Check pipeline status | - |
| `dvc stage list` | List all stages | - |
| `dvc params diff` | Compare parameters | - |
| `dvc metrics show` | Show metrics | - |

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | Select GPU(s) | `0` or `0,1` |
| `NUM_PROC` | Parallel processes | `10` |
| `BATCH_SIZE` | Override batch size | `8` |
| `MAX_LENGTH` | Max sequence length | `2048` |

### Advanced Usage

**Run specific model-dataset combination**:
```bash
dvc repro sft@Llama-3.2-3B-Instruct@pl-court-frankowe
```

**Force rerun (ignore cache)**:
```bash
dvc repro -f <stage>
```

**Run downstream dependencies**:
```bash
dvc repro --downstream <stage>
```

**Dry run (show what would run)**:
```bash
dvc repro --dry
```

---

## Matrix Execution

### Definition in `dvc.yaml`

```yaml
stages:
  sft:
    foreach:
      - model: Llama-3.2-3B-Instruct
        dataset: pl-court-instruct-sft
        seed: 42
      - model: Mistral-7B-Instruct-v0.3
        dataset: pl-court-frankowe-instruct
        seed: 42
    do:
      cmd: python scripts/sft/train.py model=${item.model} dataset=${item.dataset} seed=${item.seed}
      deps:
        - configs/model/${item.model}.yaml
        - configs/dataset/${item.dataset}.yaml
      outs:
        - models/${item.model}/${item.dataset}/${item.seed}/
```

### Execution Strategy

1. **Cartesian Product**: All combinations of specified parameters
2. **Parallel Execution**: Independent combinations run concurrently
3. **Caching**: Reuses results for unchanged combinations
4. **Tracking**: Each combination tracked separately in DVC

---

## Pipeline Optimization

### Performance Tips

1. **Batch Size**: Maximize GPU memory usage
   ```bash
   BATCH_SIZE=16 dvc repro predict
   ```

2. **Parallel Processing**: Use multiple CPU cores
   ```bash
   NUM_PROC=20 dvc repro embed
   ```

3. **GPU Selection**: Distribute across GPUs
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 dvc repro sft
   ```

4. **Caching**: Enable remote cache for team collaboration
   ```bash
   dvc remote add -d myremote s3://mybucket/dvc-cache
   dvc push
   ```

### Resource Requirements

| Stage | GPU Memory | CPU Cores | Disk Space | Time Estimate |
|-------|------------|-----------|------------|---------------|
| embed | 8GB | 10 | 50GB | 2-4 hours |
| sft (7B) | 24GB | 8 | 100GB | 6-12 hours |
| predict | 16GB | 4 | 20GB | 1-2 hours |
| evaluate | - | 4 | 10GB | 30 min |

---

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```bash
# Reduce batch size
BATCH_SIZE=2 dvc repro sft

# Enable gradient checkpointing
# Add to config: gradient_checkpointing: true
```

#### Pipeline Stage Failed
```bash
# Check logs
cat .dvc/logs/sft.log

# Force rerun
dvc repro -f sft

# Check dependencies
dvc status sft
```

#### Cache Issues
```bash
# Clear local cache
dvc cache dir

# Validate cache
dvc cache validate

# Pull from remote
dvc pull
```

#### Configuration Errors
```bash
# Validate config
python -c "from hydra import compose, initialize; initialize(config_path='configs'); cfg = compose(config_name='main'); print(cfg)"

# Check overrides
dvc params diff
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: DVC Pipeline
on: [push, pull_request]

jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: iterative/setup-dvc@v1
      - name: Pull DVC cache
        run: dvc pull
      - name: Run pipeline
        run: dvc repro
      - name: Push results
        run: dvc push
```

---

## Related Documentation

### Architecture
- **[System Architecture](../../explanation/architecture/SYSTEM_ARCHITECTURE.md)** - Overall system design
- **[Data Flow Pipeline](../../explanation/architecture/DATA_FLOW_PIPELINE.md)** - Data transformations
- **[Model Training Flow](../../explanation/architecture/MODEL_TRAINING_FLOW.md)** - Training details

### Practical Guides
- **[Fine-Tuning How-To](../../how-to/training/fine_tuning.md)** - Step-by-step training guide
- **[Evaluation How-To](../../how-to/evaluation/evaluation_guide.md)** - Running evaluations
- **[Embeddings How-To](../../how-to/embeddings/embeddings_embed_and_ingest_weaviate.md)** - Embedding generation

### Configuration Reference
- **[Model Configs](../configs/model_configs.md)** - Model specifications
- **[Dataset Configs](../configs/dataset_configs.md)** - Dataset specifications

---

## Best Practices

1. **Version Control**: Commit `dvc.yaml` and `dvc.lock` to Git
2. **Parameterization**: Use Hydra for flexible configuration
3. **Caching**: Leverage DVC cache for expensive operations
4. **Monitoring**: Track metrics with DVC metrics tracking
5. **Documentation**: Document custom stages in this directory
6. **Testing**: Test pipeline changes with `--dry` flag first
7. **Reproducibility**: Use fixed seeds for deterministic results

---

**Last Updated**: 2025-10-11
**Version**: 1.0
**Maintainer**: JuDDGES Documentation Team