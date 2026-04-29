#!/usr/bin/env bash

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m'

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
parent_dir=$(dirname $script_dir)

data_dir=$parent_dir/data

echo $data_dir


common_config="WSD \
    --save-dir ${data_dir}/asbc_eval_results \
    --no-evaluate-wsd-by-example \
    --wsd-by-instance-test-set-path ${data_dir}/asbc-deduplicated-instances.feather"

configs=(
    "lopentu/google-bert-bert-base-chinese-DottedWSD ${common_config} --batch-size 1024 --debug"
    # "lopentu/meta-llama-Llama-3.2-3B-DottedWSD ${common_config} --batch-size 2048 --use-data-parallel"
)

total_configs=${#configs[@]}

for i in "${!configs[@]}"; do
    base_cmd="uv run python -m dotted_wsd.dwsd_eval"
    cmd="$base_cmd ${configs[$i]}"
    echo -e "$YELLOW($(($i + 1))/$total_configs)$RESET Running: $base_cmd $GREEN${configs[$i]}$RESET"
    ${cmd}
done
