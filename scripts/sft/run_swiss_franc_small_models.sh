#!/usr/bin/env bash
# Swiss-franc loans extraction with small models on one 24 GB GPU (issue #78):
# zero-shot predict (guided JSON) -> QLoRA fine-tune -> predict fine-tuned (guided JSON)
# -> evaluate -> summarize, for every model and seed. Finished steps are skipped; a finished
# prediction is skipped only when its run_record.json has the current seed, guided_json,
# max_model_len and max_docs, otherwise the old output dir is moved aside and the run repeated.
#
# Usage:  scripts/sft/run_swiss_franc_small_models.sh
# Env:    DRY_RUN=1          only print the commands
#         SMOKE=1            1 document, 1 fine-tuning step, separate OUT_ROOT (<OUT_ROOT>_smoke):
#                            quick check of guided decoding + bitsandbytes + LoRA
#         MAX_MODEL_LEN      vLLM context length, same for all runs (default 8192)
#         HW_PROFILE         24gb (default, RTX 3090, 4-bit) or gb10 (ASUS Ascent GX10, bf16; set MAX_MODEL_LEN=16384)
#         GPU_UTIL           overrides gpu_memory_utilization of the profile (shared GPU)
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
SMOKE="${SMOKE:-0}"
HW_PROFILE="${HW_PROFILE:-24gb}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_DOCS=0 # 0 = all documents
SFT_SMOKE_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
    MAX_DOCS=1
    OUT_ROOT="${OUT_ROOT}_smoke"
    SFT_SMOKE_ARGS=(+training_args.max_steps=1)
fi

DATASET=pl_court_swiss_franc_loans
PROMPT=info_extraction_annotated_json
IE_SCHEMA=swiss_franc_loans
SPLIT="${SPLIT:-annotated}"
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
        if python scripts/sft/check_run_record.py "$out_dir" --seed "$seed" --guided-json \
            --max-model-len "$MAX_MODEL_LEN" --max-docs "$MAX_DOCS" >/dev/null; then
            echo "# skip predict ($mode, $model, seed $seed): $out_dir/predictions.json matches the config"
            return
        fi
        echo "# stale predict ($mode, $model, seed $seed): run_record.json missing or differs, rerunning"
        run mv "$out_dir" "$out_dir.stale-$(date +%s)"
    fi
    local max_docs_arg=()
    if [[ "$MAX_DOCS" != "0" ]]; then
        max_docs_arg=(max_docs="$MAX_DOCS")
    fi
    local gpu_util_arg=()
    if [[ -n "${GPU_UTIL:-}" ]]; then
        gpu_util_arg=(gpu_memory_utilization="$GPU_UTIL")
    fi
    run python scripts/sft/predict_vllm.py \
        dataset=$DATASET llm="$model" prompt=$PROMPT ie_schema=$IE_SCHEMA split=$SPLIT \
        +profile="${model}_predict_${HW_PROFILE}" guided_json=true max_model_len="$MAX_MODEL_LEN" \
        random_seed="$seed" output_dir="$out_dir" "${max_docs_arg[@]}" "${gpu_util_arg[@]}" "$@"
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
                +profile="${model}_sft_${HW_PROFILE}" \
                training_args.seed="$seed" output_dir="$adapter_dir" "${SFT_SMOKE_ARGS[@]}"
        fi

        predict fine_tuned "$model" "$seed" "$fine_tuned_dir" llm.adapter_path="$adapter_dir"
        evaluate "$zero_shot_dir"
        evaluate "$fine_tuned_dir"
    done
done

for mode in raw_vllm fine_tuned_vllm; do
    run python scripts/sft/summarize_metrics.py --root-dir "$OUT_ROOT/$mode/$SUB" \
        ${JUDGE_MODEL:+--judge-model "$JUDGE_MODEL"}
done
