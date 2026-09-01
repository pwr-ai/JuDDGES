# NeurIPS 2026 JuDDGES-Bench Experiments

Scripts for the JuDDGES-Bench paper experiments. Each script is self-contained
and outputs results as JSON for paper table generation.

## Scripts

| Script | Paper Section | Description |
|--------|--------------|-------------|
| `exp1_annotation_validation.py` | §4.1 | Compare semi-auto vs fully manual annotations |
| `exp2_rule_based_baseline.py` | §4.1 | Rule-based extraction baseline |
| `exp3_type_specific_metrics.py` | §5.1 | Compute type-stratified metrics from existing results |
| `exp4_cross_model_agreement.py` | §4.2 | GPT-4.1 vs Gemini 2.5 agreement on Schema B |
| `exp5_multidoc_aggregation.py` | §4.3 | Aggregation queries over 100K+ extractions |
| `exp6_field_difficulty.py` | §5.2 | Field difficulty taxonomy and analysis |
| `exp7_error_taxonomy.py` | §5.3 | Sample and categorize extraction errors |

## Running

```bash
# From JuDDGES repo root
python scripts/neurips_experiments/exp1_annotation_validation.py --predictions-dir results/pl-swiss-franc-loans/gpt-4.1/
python scripts/neurips_experiments/exp2_rule_based_baseline.py --dataset pl-swiss-franc-loans --output-dir results/pl-swiss-franc-loans/rule-based/
python scripts/neurips_experiments/exp3_type_specific_metrics.py --results-dir results/
python scripts/neurips_experiments/exp4_cross_model_agreement.py --sample-size 1000
python scripts/neurips_experiments/exp5_multidoc_aggregation.py --enriched-dataset JuDDGES/pl-court-raw-enriched
python scripts/neurips_experiments/exp6_field_difficulty.py --results-dir results/
python scripts/neurips_experiments/exp7_error_taxonomy.py --results-dir results/ --sample-size 50
```
