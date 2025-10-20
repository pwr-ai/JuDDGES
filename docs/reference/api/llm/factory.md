# LLM Factory

Factory for creating and configuring large language models with PEFT/LoRA adapter support.

## Overview

The `juddges.llm.factory` module provides factory functions for creating and configuring various LLM architectures with optimizations for legal text processing. It supports:

- Multiple model families (Llama, Mistral, Phi, Bielik)
- Quantization (4-bit, 8-bit with BitsAndBytes)
- PEFT/LoRA adapter loading
- Flash Attention 2 optimization
- Unsloth integration for fast fine-tuning
- Model-specific tokenizer configuration

## Supported Models

### Llama 3 Models
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`

### Phi Models
- `microsoft/Phi-4`
- `microsoft/Phi-4-mini-instruct`

### Mistral Models
- `mistralai/Mistral-Nemo-Instruct-2407`
- `CYFRAGOVPL/PLLuM-12B-instruct`

### Bielik Models (Polish)
- `speakleash/Bielik-11B-v2.3-Instruct`

## Key Features

- **Unified Interface**: Single entry point for all model types
- **Quantization Support**: 4-bit quantization for reduced memory
- **Adapter Loading**: Load fine-tuned PEFT/LoRA adapters
- **Model-Specific Config**: Automatic configuration based on model family
- **Flash Attention**: Automatic Flash Attention 2 when available
- **Unsloth Integration**: Fast training with Unsloth framework

## Usage Examples

### Basic Model Loading

```python
from juddges.config import LLMConfig
from juddges.llm.factory import get_llm

# Create configuration
config = LLMConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=4096,
    use_4bit=True,
    padding_side="left"
)

# Load model
model_pack = get_llm(config)

# Access components
model = model_pack.model
tokenizer = model_pack.tokenizer
generate_kwargs = model_pack.generate_kwargs
```

### Loading with Fine-Tuned Adapter

```python
config = LLMConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    should_load_adapter=True,
    adapter_path="outputs/lora_adapter",
    use_4bit=True
)

model_pack = get_llm(config)
# Model is loaded with adapter merged
```

### Using Unsloth for Training

```python
config = LLMConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    use_unsloth=True,
    use_4bit=True,
    max_seq_length=4096
)

model_pack = get_llm(config)
# Uses Unsloth's FastLanguageModel for efficient training
```

### GPU Configuration

```python
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load model (automatically uses available GPU)
model_pack = get_llm(config)
```

## API Reference

::: juddges.llm.factory.ModelForGeneration
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.llm.factory.get_llm
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.llm.factory.get_llama_3
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.llm.factory.get_mistral
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.llm.factory.get_llm_with_default_setup
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.llm.factory.get_llm_tokenizer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Configuration Options

### LLMConfig Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | HuggingFace model identifier |
| `max_seq_length` | int | Maximum sequence length (default: 2048) |
| `use_4bit` | bool | Enable 4-bit quantization |
| `use_unsloth` | bool | Use Unsloth framework |
| `should_load_adapter` | bool | Load PEFT adapter |
| `adapter_path` | str | Path to adapter weights |
| `padding_side` | str | "left" or "right" padding |

### Generation Parameters

Each model returns generation kwargs optimized for that model:

**Llama 3**:
```python
{
    "eos_token_id": [tokenizer.eos_token_id, eot_token_id],
    "pad_token_id": tokenizer.eos_token_id
}
```

**Mistral/Phi/Bielik**:
```python
{
    "pad_token_id": tokenizer.eos_token_id
}
```

## Memory Optimization

### 4-bit Quantization

Reduces memory usage by ~4x:

```python
config = LLMConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    use_4bit=True  # BitsAndBytes 4-bit quantization
)
```

**Memory Usage**:
- Without quantization: ~32GB (8B model)
- With 4-bit: ~8GB (8B model)

### Flash Attention 2

Automatically enabled when CUDA is available:

- Reduces memory usage during attention computation
- Increases inference speed
- No configuration needed (automatic)

## Model-Specific Behaviors

### Llama 3

- Uses special `<|eot_id|>` token as terminator
- Requires both EOS and EOT tokens for generation
- Recommended padding: left

### Mistral

- Uses single EOS token
- Supports longer contexts (up to 32k tokens)
- Recommended for multilingual tasks

### Phi-4

- Optimized for instruction following
- Smaller model size (efficient for inference)
- Good for constrained environments

### Bielik (Polish)

- Specialized for Polish language
- Pre-trained on Polish legal and general text
- Best performance on Polish legal documents

## Error Handling

```python
from juddges.llm.factory import get_llm

try:
    model_pack = get_llm(config)
except ValueError as e:
    # Model not supported
    print(f"Model error: {e}")
except RuntimeError as e:
    # GPU/CUDA errors
    print(f"Runtime error: {e}")
except Warning as e:
    # Adapter loading warnings
    print(f"Adapter warning: {e}")
```

## Related

- [Prediction](predict.md) - Generate predictions with loaded models
- [Core Config](../core/config.md) - Configuration details
- [How-To: Fine-Tuning](../../../how-to/training/fine_tuning.md) - Model training guide

## Common Patterns

### Production Inference

```python
# Load quantized model without adapter
config = LLMConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    use_4bit=True,
    max_seq_length=4096
)
model_pack = get_llm(config)
```

### Development/Fine-Tuning

```python
# Load with Unsloth for training
config = LLMConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    use_unsloth=True,
    use_4bit=True
)
model_pack = get_llm(config)
```

### Loading Fine-Tuned Model

```python
# Load with merged adapter
config = LLMConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    should_load_adapter=True,
    adapter_path="outputs/checkpoint-1000",
    use_4bit=True
)
model_pack = get_llm(config)
```
