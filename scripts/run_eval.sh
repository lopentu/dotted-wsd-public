#!/usr/bin/env bash
#
# Evaluate every released DottedWSD model from the lopentu/* org on the
# WSD + RPD held-out test sets used for Chen, Lian, Hsieh (2025). Comment
# out the lines you don't want to run before invoking.
#
# Usage:
#   uv sync                    # one-time
#   bash scripts/run_eval.sh   # runs every uncommented config sequentially

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m'

configs=(
    "lopentu/meta-llama-Llama-3.2-3B-DottedWSD ALL"
    "lopentu/ckiplab-bert-base-chinese-DottedWSD ALL"
    "lopentu/yentinglin-bert-base-zhtw-DottedWSD ALL"
    "lopentu/google-bert-bert-base-chinese-DottedWSD ALL"
    "lopentu/IDEA-CCNL-Erlangshen-DeBERTa-v2-97M-Chinese-DottedWSD ALL"
    "lopentu/MoritzLaurer-mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-DottedWSD ALL"
    "lopentu/microsoft-mdeberta-v3-base-DottedWSD ALL"
    "lopentu/microsoft-deberta-v3-small-DottedWSD ALL"
    "lopentu/microsoft-deberta-v3-base-DottedWSD ALL"
    "lopentu/microsoft-deberta-v3-large-DottedWSD ALL"
    "lopentu/SmolLM-Chinese-180M-DottedWSD ALL"
    "lopentu/gemma-2-2b-DottedWSD ALL"
    "lopentu/MoritzLaurer-mDeBERTa-v3-base-mnli-xnli-DottedWSD ALL"
)

total_configs=${#configs[@]}

for i in "${!configs[@]}"; do
    base_cmd="uv run python -m dotted_wsd.dwsd_eval"
    cmd="$base_cmd ${configs[$i]}"
    echo -e "$YELLOW($(($i+1))/$total_configs)$RESET Running: $base_cmd $GREEN${configs[$i]}$RESET"
    ${cmd}
done
