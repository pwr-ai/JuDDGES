#!/usr/bin/env bash
# Swiss-franc loans extraction with small models on one 24 GB GPU (issue #78):
# zero-shot predict (guided JSON) -> QLoRA fine-tune -> predict fine-tuned (guided JSON)
# -> evaluate -> summarize, for every model and seed. Finished steps are skipped.
#
# Usage:  scripts/sft/run_swiss_franc_small_models.sh
# Env:    DRY_RUN=1          only print the commands
#         JUDGE_MODEL=<name> run scripts/sft/llm_as_judge.py with this judge (needs OPENAI_API_KEY),
#                            skipped when empty
#         MODELS, SEEDS      space-separated lists (configs/llm/<name>.yaml, seeds)
#         OUT_ROOT           where everything is written
set -euo pipefail

MODELS="${MODELS:-bielik_11b_v23_instruct llama_3.1_8b_instruct}"
SEEDS="${SEEDS:-42 997 7312}"
OUT_ROOT="${OUT_ROOT:-data/experiments/swiss_franc_small_models}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
DRY_RUN="${DRY_RUN:-0}"

DATASET=pl_court_swiss_franc_loans
PROMPT=info_extraction_annotated_json
IE_SCHEMA=swiss_franc_loans
SPLIT=annotated
SCHEMA_FILE=configs/ie_schema/${IE_SCHEMA}.yaml

export PYTHONPATH=.

run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "$@"
        echo
    else
        "$@"
    fi
}

predict() { # <mode> <model> <seed> <out_dir> [hydra overrides...]
    local mode=$1 model=$2 seed=$3 out_dir=$4
    shift 4
    if [[ -f "$out_dir/predictions.json" ]]; then
        echo "# skip predict ($mode, $model, seed $seed): $out_dir/predictions.json exists"
        return
    fi
    run python scripts/sft/predict_vllm.py \
        dataset=$DATASET llm="$model" prompt=$PROMPT ie_schema=$IE_SCHEMA split=$SPLIT \
        +profile="${model}_predict_24gb" guided_json=true \
        random_seed="$seed" output_dir="$out_dir" "$@"
}

evaluate() { # <out_dir>
    local out_dir=$1
    if [[ -f "$out_dir/scores_ngram.json" ]]; then
        echo "# skip ngram evaluation: $out_dir/scores_ngram.json exists"
    elif [[ "$DRY_RUN" == "1" || -f "$out_dir/predictions.json" ]]; then
        run python scripts/evaluation/ngram_based_eval.py "$out_dir"
    fi
    if [[ -z "$JUDGE_MODEL" ]]; then
        echo "# skip llm-as-judge: JUDGE_MODEL is empty"
    elif [[ -f "$out_dir/llm_as_judge_${JUDGE_MODEL}.json" ]]; then
        echo "# skip llm-as-judge: $out_dir/llm_as_judge_${JUDGE_MODEL}.json exists"
    else
        run python scripts/sft/llm_as_judge.py "$out_dir/predictions.json" "$SCHEMA_FILE" \
            --judge-model "$JUDGE_MODEL"
    fi
}

SUB=$DATASET/$SPLIT
for model in $MODELS; do
    for seed in $SEEDS; do
        tail=$model/$PROMPT/$IE_SCHEMA/seed_$seed
        zero_shot_dir=$OUT_ROOT/raw_vllm/$SUB/$tail
        adapter_dir=$OUT_ROOT/peft-fine-tune/$model/$DATASET/$PROMPT/$IE_SCHEMA/seed_$seed
        fine_tuned_dir=$OUT_ROOT/fine_tuned_vllm/$SUB/$tail

        echo "### $model, seed $seed"
        predict zero_shot "$model" "$seed" "$zero_shot_dir"

        if [[ -f "$adapter_dir/adapter_model.safetensors" ]]; then
            echo "# skip fine-tune: $adapter_dir/adapter_model.safetensors exists"
        else
            run python scripts/sft/fine_tune_deepspeed.py \
                dataset=$DATASET llm="$model" prompt=$PROMPT ie_schema=$IE_SCHEMA \
                +profile="${model}_sft_24gb" \
                training_args.seed="$seed" output_dir="$adapter_dir"
        fi

        predict fine_tuned "$model" "$seed" "$fine_tuned_dir" llm.adapter_path="$adapter_dir"
        evaluate "$zero_shot_dir"
        evaluate "$fine_tuned_dir"
    done
done

for mode in raw_vllm fine_tuned_vllm; do
    run python scripts/sft/summarize_metrics.py --root-dir "$OUT_ROOT/$mode/$SUB"
done
