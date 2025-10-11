# Tutorial: Fine-tuning Your First Legal LLM

Learn how to fine-tune large language models for legal document analysis using JuDDGES infrastructure. This tutorial covers preparing instruction datasets, configuring training with PEFT/LoRA, and evaluating model performance.

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Prerequisites](#prerequisites)
- [Step 1: Prepare Instruction Dataset](#step-1-prepare-instruction-dataset)
- [Step 2: Configure Training](#step-2-configure-training)
- [Step 3: Run Fine-tuning](#step-3-run-fine-tuning)
- [Step 4: Evaluate Model](#step-4-evaluate-model)
- [Step 5: Use Your Fine-tuned Model](#step-5-use-your-fine-tuned-model)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Learning Objectives

By the end of this tutorial, you will:

- ✅ Prepare instruction datasets for legal tasks
- ✅ Configure PEFT/LoRA for efficient fine-tuning
- ✅ Train a model using JuDDGES infrastructure
- ✅ Evaluate fine-tuned model performance
- ✅ Deploy and use your custom legal LLM

**Estimated Time**: 60 minutes (+ training time)
**GPU Required**: Yes (40GB+ VRAM recommended)

---

## Prerequisites

### Required Knowledge
- Completion of [Tutorial 1](./tutorial-01-first-legal-document-analysis.md) and [Tutorial 2](./tutorial-02-embeddings.md)
- Understanding of language models and fine-tuning concepts
- Familiarity with command line and Python

### Required Hardware
- **GPU**: NVIDIA GPU with 40GB+ VRAM (A100, A6000, or similar)
- **RAM**: 64GB+ system RAM
- **Storage**: 100GB+ free space for models and datasets

### Required Software
- JuDDGES environment with CUDA support
- DVC installed and configured

---

## Step 1: Prepare Instruction Dataset

### Understanding Instruction Format

Instruction datasets follow this format:

```json
{
  "instruction": "Wyodrębnij sygnaturę sprawy z wyroku.",
  "input": "Sąd Okręgowy w Warszawie, sygn. II C 123/2023...",
  "output": "II C 123/2023"
}
```

### Create Sample Dataset

```python
"""Create an instruction dataset for legal information extraction."""

from datasets import Dataset, DatasetDict
from rich.console import Console
import random

console = Console()

# Sample legal documents with extraction tasks
samples = [
    {
        "instruction": "Wyodrębnij datę wyroku w formacie ISO 8601.",
        "input": "Wyrok z dnia 15 marca 2023 roku. Sąd Okręgowy w Warszawie orzekł...",
        "output": "2023-03-15",
    },
    {
        "instruction": "Wyodrębnij nazwę sądu.",
        "input": "W imieniu Rzeczypospolitej Polskiej. Sąd Okręgowy w Krakowie...",
        "output": "Sąd Okręgowy w Krakowie",
    },
    {
        "instruction": "Wyodrębnij sygnaturę sprawy.",
        "input": "Sygnatura akt: II C 456/2023. Sąd rozpoznał sprawę...",
        "output": "II C 456/2023",
    },
    # Add more samples...
]

# Create dataset
dataset = Dataset.from_list(samples)

# Split into train/test
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Save
dataset.save_to_disk("./data/instruction_dataset")

console.print(f"[green]✓ Created dataset with {len(dataset['train'])} training examples[/green]")
```

### Load Existing Dataset

```python
from datasets import load_dataset

# Load pre-built Swiss franc loans dataset
dataset = load_dataset("JuDDGES/swiss_franc_loans_instruct")

print(f"Train: {len(dataset['train'])} examples")
print(f"Test: {len(dataset['test'])} examples")

# Inspect sample
sample = dataset['train'][0]
print(f"Instruction: {sample['instruction'][:100]}...")
print(f"Output: {sample['output'][:100]}...")
```

---

## Step 2: Configure Training

### Create Configuration File

Create `configs/my_finetuning.yaml`:

```yaml
# Model configuration
llm:
  model_name: "meta-llama/Llama-3.2-3B-Instruct"
  tokenizer_name: "meta-llama/Llama-3.2-3B-Instruct"
  torch_dtype: "bfloat16"
  device_map: "auto"

# Dataset configuration
dataset:
  name: "JuDDGES/swiss_franc_loans_instruct"
  instruction_field: "instruction"
  input_field: "input"
  output_field: "output"

# PEFT/LoRA configuration
peft:
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

# Training configuration
training:
  output_dir: "./outputs/llama-3.2-3b-legal"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_steps: 100
  logging_steps: 10
  save_steps: 500
  evaluation_strategy: "steps"
  eval_steps: 500
  fp16: false
  bf16: true
  max_grad_norm: 0.3
  optim: "paged_adamw_8bit"
```

### Verify Configuration

```python
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load("configs/my_finetuning.yaml")

# Verify settings
print(f"Model: {config.llm.model_name}")
print(f"Dataset: {config.dataset.name}")
print(f"LoRA rank: {config.peft.lora_r}")
print(f"Epochs: {config.training.num_train_epochs}")
print(f"Batch size: {config.training.per_device_train_batch_size}")
```

---

## Step 3: Run Fine-tuning

### Using the Fine-tuning Script

```bash
# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run fine-tuning
python scripts/sft/fine_tune_llm.py \
    --config configs/my_finetuning.yaml \
    --output_dir ./outputs/my-legal-model
```

### Using DVC Pipeline

```bash
# Configure DVC stage
dvc stage add -n finetune \
    -d data/instruction_dataset \
    -d configs/my_finetuning.yaml \
    -o outputs/my-legal-model \
    python scripts/sft/fine_tune_llm.py

# Run pipeline
dvc repro finetune
```

### Monitor Training

```python
"""Monitor training progress."""

from pathlib import Path
import pandas as pd
import plotly.express as px

# Load training logs
log_file = Path("outputs/my-legal-model/trainer_state.json")

if log_file.exists():
    import json

    with open(log_file) as f:
        state = json.load(f)

    # Extract loss history
    history = state["log_history"]
    df = pd.DataFrame(history)

    # Plot training loss
    fig = px.line(df, x="step", y="loss", title="Training Loss")
    fig.show()

    print(f"Current epoch: {state['epoch']}")
    print(f"Global step: {state['global_step']}")
    print(f"Best metric: {state.get('best_metric', 'N/A')}")
```

**Expected Training Time**:
- Llama-3.2-3B: ~2-3 hours on A100 (1000 examples)
- Llama-3.1-8B: ~6-8 hours on A100 (1000 examples)

---

## Step 4: Evaluate Model

### Run Evaluation

```bash
# Evaluate on test set
python scripts/sft/evaluate.py \
    --model_path ./outputs/my-legal-model \
    --dataset JuDDGES/swiss_franc_loans_instruct \
    --split test \
    --output_file ./outputs/evaluation_results.json
```

### Evaluation Metrics

```python
"""Analyze evaluation results."""

import json
from rich.console import Console
from rich.table import Table

console = Console()

# Load results
with open("./outputs/evaluation_results.json") as f:
    results = json.load(f)

# Display metrics
table = Table(title="Evaluation Metrics")
table.add_column("Metric", style="cyan")
table.add_column("Score", style="green")

for metric, score in results["metrics"].items():
    table.add_row(metric, f"{score:.4f}")

console.print(table)

# Show example predictions
console.print("\n[bold]Example Predictions:[/bold]")
for i, example in enumerate(results["examples"][:3], 1):
    console.print(f"\n[cyan]Example {i}:[/cyan]")
    console.print(f"Input: {example['input'][:100]}...")
    console.print(f"Expected: {example['expected'][:100]}...")
    console.print(f"Predicted: {example['predicted'][:100]}...")
    console.print(f"Match: {example['match']}")
```

### Compare with Baseline

```python
"""Compare fine-tuned model with base model."""

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load both models
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
finetuned_model = AutoModelForCausalLM.from_pretrained("./outputs/my-legal-model")

# Test on same examples
test_input = "Wyodrębnij datę wyroku: Wyrok z dnia 15 marca 2023..."

# Generate from both
base_output = generate(base_model, test_input)
finetuned_output = generate(finetuned_model, test_input)

print(f"Base model: {base_output}")
print(f"Fine-tuned: {finetuned_output}")
```

---

## Step 5: Use Your Fine-tuned Model

### Load and Inference

```python
"""Use your fine-tuned model for inference."""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
model_path = "./outputs/my-legal-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def extract_information(instruction: str, text: str) -> str:
    """Extract information from legal text."""
    # Format prompt
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Output:\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=False,
    )

    # Decode
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract output section
    if "### Output:" in result:
        result = result.split("### Output:")[1].strip()

    return result

# Test
instruction = "Wyodrębnij datę wyroku w formacie ISO 8601."
text = "Wyrok Sądu Okręgowego w Warszawie z dnia 15 marca 2023 roku..."

output = extract_information(instruction, text)
print(f"Extracted: {output}")
```

### Deploy as API

```python
"""Simple FastAPI deployment."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model once at startup
model = load_model("./outputs/my-legal-model")

class ExtractionRequest(BaseModel):
    instruction: str
    text: str

@app.post("/extract")
def extract(request: ExtractionRequest):
    result = extract_information(
        request.instruction,
        request.text
    )
    return {"result": result}

# Run: uvicorn api:app --reload
```

---

## Best Practices

### 1. Dataset Quality

```python
# ✅ Good: Clear, specific instructions
{
    "instruction": "Extract the court name from the judgment header.",
    "input": "Court of Appeal, London...",
    "output": "Court of Appeal, London"
}

# ❌ Bad: Vague, ambiguous
{
    "instruction": "Get the important info.",
    "input": "Some text...",
    "output": "Stuff"
}
```

### 2. Hyperparameter Tuning

Start with these defaults:

```python
# For 3B-8B models
lora_r = 16          # Rank (higher = more capacity)
lora_alpha = 32      # Scaling factor
learning_rate = 2e-4 # Learning rate
batch_size = 4       # Per device

# For 11B-70B models
lora_r = 32
lora_alpha = 64
learning_rate = 1e-4
batch_size = 1
```

### 3. Training Monitoring

```python
# Check for overfitting
if eval_loss starts increasing:
    # Reduce epochs or add regularization
    num_epochs = 2  # Instead of 3
    lora_dropout = 0.1  # Instead of 0.05

# Check for underfitting
if train_loss plateaus high:
    # Increase capacity or learning rate
    lora_r = 32  # Instead of 16
    learning_rate = 3e-4  # Instead of 2e-4
```

### 4. Evaluation

Always evaluate on:
- ✅ Held-out test set (not used in training)
- ✅ Real-world examples
- ✅ Edge cases and difficult examples
- ✅ Multiple metrics (exact match, F1, BLEU)

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions**:
```python
# 1. Reduce batch size
per_device_train_batch_size = 1
gradient_accumulation_steps = 16

# 2. Use 8-bit quantization
load_in_8bit = true

# 3. Enable gradient checkpointing
gradient_checkpointing = true

# 4. Use smaller model
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Instead of 8B
```

### Issue: "Model not learning"

**Solutions**:
```python
# 1. Increase learning rate
learning_rate = 5e-4  # Instead of 2e-4

# 2. Increase LoRA rank
lora_r = 32  # Instead of 16

# 3. Train longer
num_train_epochs = 5  # Instead of 3

# 4. Check data quality
# Ensure instructions are clear and outputs are correct
```

### Issue: "Model overfitting"

**Solutions**:
```python
# 1. Reduce epochs
num_train_epochs = 2

# 2. Increase dropout
lora_dropout = 0.1

# 3. Add more training data
# Collect more diverse examples

# 4. Use early stopping
early_stopping_patience = 3
```

---

## Summary

You've successfully fine-tuned a legal LLM!

### What You've Learned

✅ **Dataset Preparation**: Created instruction datasets
✅ **Configuration**: Set up PEFT/LoRA parameters
✅ **Training**: Fine-tuned models efficiently
✅ **Evaluation**: Assessed model performance
✅ **Deployment**: Used fine-tuned models in production

### Key Metrics

Your fine-tuned model should achieve:
- **Exact Match**: 70-85% on legal extraction tasks
- **F1 Score**: 80-90% on field-level evaluation
- **Training Time**: 2-8 hours depending on model size
- **Inference**: <1s per prediction

---

## Next Steps

1. **[Tutorial 4: Advanced Information Extraction](./tutorial-04-advanced-extraction.md)** - Complex schemas and pipelines
2. **[Tutorial 5: End-to-End Project](./tutorial-05-end-to-end-project.md)** - Build complete system

### Related Documentation

- [Model Training Flow](../explanation/architecture/MODEL_TRAINING_FLOW.md)
- [DVC Pipeline Reference](../reference/pipelines/DVC_PIPELINE.md)
- [Evaluation Metrics Guide](../reference/api/evals/metrics.md)

---

**Last Updated**: 2025-10-11 | **Version**: 1.0 | **Status**: Published

**Difficulty**: Advanced | **Prerequisites**: GPU access, Tutorials 1-2
