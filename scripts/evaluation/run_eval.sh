#!/bin/bash

### Script for evaluating the predictions with llm-as-judge/ngram-based evaluation
### Helps to invoke series of commands outside dvc (for batch llm-as-judge is necessary to wait for the results up to 24 hours)

set -e

pred_dirs=(
    # <fill_pred_dirs_here>
)

total_dirs=${#pred_dirs[@]}
current_num=0

for pred_dir in "${pred_dirs[@]}"; do
    current_num=$((current_num + 1))
    echo "[$current_num/$total_dirs] Evaluating: $pred_dir"
    # echo "python scripts/evaluation/ngram_based_eval.py $pred_dir"
    # python scripts/evaluation/ngram_based_eval.py $pred_dir
    # python scripts/evaluation/llm_as_judge_batch.py submit $pred_dir --judge-model gpt-4.1-mini
    # python scripts/evaluation/llm_as_judge_batch.py download_and_process_results $pred_dir --judge-model gpt-4.1-mini

done
