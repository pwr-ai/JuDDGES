# Evaluation Metrics

Comprehensive evaluation metrics for information extraction tasks with support for dates, numbers, strings, enums, and lists.

## Overview

The `juddges.evals.metrics` module provides field-level evaluation metrics for comparing predicted extractions against ground truth labels. Each metric is designed for a specific data type:

- **Dates**: Flexible date parsing and exact matching
- **Numbers**: Numeric comparison with tolerance
- **Strings**: ROUGE scores (unigram, bigram, longest common subsequence)
- **Enums**: Classification accuracy with hallucination detection
- **Lists**: Precision, recall, F1 with greedy matching

## Key Features

- **Type-Specific Metrics**: Optimized evaluation for each data type
- **Flexible Parsing**: Handles various date and number formats
- **ROUGE Scores**: Industry-standard text similarity metrics
- **Hallucination Detection**: Identifies out-of-vocabulary enum predictions
- **List Matching**: Greedy matching with standard IR metrics
- **Zero-Value Handling**: Proper handling of missing or null values

## Usage Examples

### Date Evaluation

```python
from juddges.evals.metrics import evaluate_date

# Exact match
result = evaluate_date("2024-01-15", "2024-01-15")
print(result)  # {"match": 1}

# Flexible parsing
result = evaluate_date("15 stycznia 2024", "2024-01-15")
print(result)  # {"match": 1} - parsed and matched

# Mismatch
result = evaluate_date("2024-01-15", "2024-01-16")
print(result)  # {"match": 0}

# Missing value
result = evaluate_date(None, "2024-01-15")
print(result)  # {"match": 0}
```

### Number Evaluation

```python
from juddges.evals.metrics import evaluate_number

# Exact match
result = evaluate_number(42, 42)
print(result)  # {"match": 1}

# String parsing
result = evaluate_number("1000.50", "1000.5")
print(result)  # {"match": 1}

# Tolerance-based matching
result = evaluate_number(1000.0001, 1000.0, atol=0.001)
print(result)  # {"match": 1}

# Type conversion
result = evaluate_number("42", 42)
print(result)  # {"match": 1}
```

### String (ROUGE) Evaluation

```python
from juddges.evals.metrics import evaluate_string_rouge

# Exact match
result = evaluate_string_rouge(
    "Sąd Okręgowy w Warszawie",
    "Sąd Okręgowy w Warszawie"
)
print(result)
# {
#     "rouge1": 1.0,
#     "rouge2": 1.0,
#     "rougeL": 1.0
# }

# Partial match
result = evaluate_string_rouge(
    "Sąd w Warszawie",
    "Sąd Okręgowy w Warszawie"
)
print(result)
# {
#     "rouge1": 0.75,  # 3 of 4 unigrams match
#     "rouge2": 0.67,  # 2 of 3 bigrams match
#     "rougeL": 0.75   # Longest common subsequence
# }

# Missing values
result = evaluate_string_rouge(None, "some text")
print(result)
# {
#     "rouge1": 0.0,
#     "rouge2": 0.0,
#     "rougeL": 0.0
# }
```

### Enum Evaluation

```python
from juddges.evals.metrics import evaluate_enum

# Valid prediction
result = evaluate_enum(
    predicted="Wyrok",
    gold="Wyrok",
    choices=["Wyrok", "Postanowienie", "Uchwała"]
)
print(result)
# {
#     "match": 1,
#     "predicted_in_choices": 1
# }

# Hallucination (invalid choice)
result = evaluate_enum(
    predicted="InvalidType",
    gold="Wyrok",
    choices=["Wyrok", "Postanowienie", "Uchwała"]
)
print(result)
# {
#     "match": 0,
#     "predicted_in_choices": 0  # Hallucination detected
# }

# Mismatch but valid
result = evaluate_enum(
    predicted="Postanowienie",
    gold="Wyrok",
    choices=["Wyrok", "Postanowienie", "Uchwała"]
)
print(result)
# {
#     "match": 0,
#     "predicted_in_choices": 1
# }
```

### List Evaluation

```python
from juddges.evals.metrics import evaluate_list_greedy

# Perfect match
result = evaluate_list_greedy(
    predicted=["Art. 123", "Art. 456"],
    gold=["Art. 123", "Art. 456"]
)
print(result)
# {
#     "true_positives": 2,
#     "false_positives": 0,
#     "false_negatives": 0,
#     "precision": 1.0,
#     "recall": 1.0,
#     "f1": 1.0
# }

# Partial match
result = evaluate_list_greedy(
    predicted=["Art. 123", "Art. 789"],
    gold=["Art. 123", "Art. 456"]
)
print(result)
# {
#     "true_positives": 1,
#     "false_positives": 1,
#     "false_negatives": 1,
#     "precision": 0.5,
#     "recall": 0.5,
#     "f1": 0.5
# }

# Empty lists
result = evaluate_list_greedy(
    predicted=None,
    gold=["Art. 123"]
)
print(result)
# {
#     "true_positives": 0,
#     "false_positives": 0,
#     "false_negatives": 1,
#     "precision": 0.0,
#     "recall": 0.0,
#     "f1": 0.0
# }
```

## API Reference

::: juddges.evals.metrics.evaluate_date
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.evals.metrics.evaluate_number
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.evals.metrics.evaluate_string_rouge
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.evals.metrics.evaluate_enum
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

::: juddges.evals.metrics.evaluate_list_greedy
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Metric Selection Guide

### When to Use Each Metric

| Data Type | Metric | Use Case |
|-----------|--------|----------|
| Date | `evaluate_date` | Dates, timestamps |
| Number | `evaluate_number` | Amounts, counts, IDs |
| String | `evaluate_string_rouge` | Names, titles, long text |
| Enum | `evaluate_enum` | Categories, classifications |
| List | `evaluate_list_greedy` | Arrays, multi-value fields |

### Choosing ROUGE vs Exact Match

**Use ROUGE** when:
- Text can be paraphrased
- Word order doesn't matter
- Partial credit is meaningful

**Use Exact Match** when:
- Precision is critical (legal basis references)
- Format is standardized (case numbers)
- No variation expected

## Advanced Usage

### Custom Tolerance for Numbers

```python
# Financial amounts (2 decimal places)
result = evaluate_number(
    predicted=1000.01,
    gold=1000.00,
    atol=0.01
)

# Approximate counts
result = evaluate_number(
    predicted=103,
    gold=100,
    atol=5
)
```

### Aggregating Metrics

```python
from typing import Dict, List

def aggregate_metrics(results: List[Dict]) -> Dict:
    """Aggregate evaluation results across multiple examples."""
    total = len(results)

    return {
        "accuracy": sum(r["match"] for r in results) / total,
        "total_examples": total,
        "correct": sum(r["match"] for r in results)
    }

# Evaluate multiple extractions
results = [
    evaluate_date(pred, gold)
    for pred, gold in zip(predictions, ground_truth)
]

# Aggregate
summary = aggregate_metrics(results)
print(summary)
# {
#     "accuracy": 0.85,
#     "total_examples": 100,
#     "correct": 85
# }
```

### Multi-Field Evaluation

```python
def evaluate_extraction(predicted: Dict, gold: Dict, schema: Dict) -> Dict:
    """Evaluate all fields in an extraction."""
    results = {}

    for field, config in schema.items():
        pred_val = predicted.get(field)
        gold_val = gold.get(field)

        if config["type"] == "date":
            results[field] = evaluate_date(pred_val, gold_val)
        elif config["type"] == "enum":
            results[field] = evaluate_enum(
                pred_val, gold_val, config["choices"]
            )
        elif config["type"] == "string":
            results[field] = evaluate_string_rouge(pred_val, gold_val)
        # ... handle other types

    return results

# Define schema
schema = {
    "verdict_date": {"type": "date"},
    "judgment_type": {
        "type": "enum",
        "choices": ["Wyrok", "Postanowienie", "Uchwała"]
    },
    "court_name": {"type": "string"}
}

# Evaluate
results = evaluate_extraction(predicted, gold, schema)
```

## Performance Considerations

### ROUGE Computation

ROUGE metrics use TorchMetrics for efficient computation:

```python
# For large batches, ROUGE is vectorized
results = [
    evaluate_string_rouge(pred, gold)
    for pred, gold in zip(predictions, ground_truth)
]
# TorchMetrics handles batching internally
```

### Date Parsing

Date parsing uses `dateutil.parser` which is flexible but can be slow:

```python
# For performance-critical code, pre-parse dates
from dateutil import parser

parsed_preds = [parser.parse(d) for d in date_strings]
parsed_golds = [parser.parse(d) for d in gold_strings]

results = [
    evaluate_date(str(p.date()), str(g.date()))
    for p, g in zip(parsed_preds, parsed_golds)
]
```

## Related

- [Extraction Evaluation](extraction.md) - End-to-end evaluation pipeline
- [LLM as Judge](../llm_as_judge/judge.md) - LLM-based evaluation
- [Gemini Chain](../extraction/gemini_chain.md) - Extraction pipeline
- [How-To: Evaluation](../../../how-to/evaluation/evaluation_pipeline.md) - Evaluation guide

## Common Patterns

### Production Evaluation Pipeline

```python
import pandas as pd
from juddges.evals.metrics import (
    evaluate_date,
    evaluate_string_rouge,
    evaluate_enum,
    evaluate_list_greedy
)

def evaluate_dataset(predictions_df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    """Evaluate entire dataset against schema."""
    results = []

    for idx, row in predictions_df.iterrows():
        pred = row["predicted"]
        gold = row["ground_truth"]

        # Evaluate each field
        eval_result = {"id": row["id"]}

        for field, config in schema.items():
            pred_val = pred.get(field)
            gold_val = gold.get(field)

            if config["type"] == "date":
                metrics = evaluate_date(pred_val, gold_val)
                eval_result[f"{field}_match"] = metrics["match"]

            elif config["type"] == "string":
                metrics = evaluate_string_rouge(pred_val, gold_val)
                eval_result[f"{field}_rouge1"] = metrics["rouge1"]
                eval_result[f"{field}_rouge2"] = metrics["rouge2"]
                eval_result[f"{field}_rougeL"] = metrics["rougeL"]

            # ... handle other types

        results.append(eval_result)

    return pd.DataFrame(results)
```
