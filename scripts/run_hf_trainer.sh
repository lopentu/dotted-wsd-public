#!/usr/bin/env bash
#
# Reproduce the fine-tuning grid for Chen, Lian, Hsieh (2025). Each entry
# below trains one base model end-to-end on the dotted-WSD + RPD data and
# pushes the resulting checkpoint (when --push-to-hub is set). Comment out
# the lines you don't want to run before invoking this script.
#
# Note: the paper itself reports on a subset of these (ten models per the
# abstract); a few extras are kept here for sibling-model comparison.
#
# Usage:
#   uv sync                         # one-time, sets up the environment
#   bash scripts/run_hf_trainer.sh  # runs every uncommented config sequentially
#
# By default, training metrics are not reported anywhere and the trained model
# is not pushed to the Hugging Face Hub.
#
# - Pass `--report-to wandb` (and set WANDB_ENTITY / WANDB_PROJECT env vars) to
#   log to Weights & Biases.
# - Pass `--push-to-hub --hub-model-id your-namespace/your-model-name` to push
#   the resulting model. Without `--hub-model-id`, the default lands in the
#   `lopentu/` org, which only the original authors can write to.

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m'

configs=(
    "meta-llama/Llama-3.2-3B 8 --use-lora --lora-rank 8 --lora-alpha 16 --tokenizer-config-path tokenizers/configs/meta-llama-Llama-3.2-3B.yaml --suffix EvalLoss"  # bf16
    "Mxode/SmolLM-Chinese-180M 32 --no-bf16 --tokenizer-config-path tokenizers/configs/Mxode-SmolLM-Chinese-180M.yaml --suffix -EvalLoss" # fp32
    "google/gemma-2-2b 8 --use-lora --lora-rank 8 --lora-alpha 16 --tokenizer-config-path tokenizers/configs/google-gemma-2-2b.yaml --suffix -EvalLoss --torch-dtype bfloat16"  # fp32 but errors, so use bf16
    "yentinglin/bert-base-zhtw 64"  # fp32
    "ckiplab/bert-base-chinese 64"  # no dtype, keep fp32 just like original BERT
    "google-bert/bert-base-chinese 64"  # fp32
    "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese 32"  # fp16
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 32"  # fp16
    "microsoft/mdeberta-v3-base 32 --torch-dtype bfloat16"  # no dtype
    "WENGSYX/Deberta-Chinese-Large 16 --torch-dtype bfloat16"  # no dtype
    "microsoft/deberta-v3-small 64 --torch-dtype bfloat16"  # no dtype
    "microsoft/deberta-v3-base 32 --torch-dtype bfloat16"  # no dtype
    "microsoft/deberta-v3-large 16 --torch-dtype bfloat16"  # no dtype
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli 32"  # fp16
)

total_configs=${#configs[@]}

for i in "${!configs[@]}"; do
    base_cmd="uv run python -m dotted_wsd.train.hf_trainer"
    cmd="$base_cmd ${configs[$i]}"
    echo -e "$YELLOW($(($i+1))/$total_configs)$RESET Running: $base_cmd $GREEN${configs[$i]}$RESET"
    ${cmd}
done
