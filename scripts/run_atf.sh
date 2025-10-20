#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/workdir/scripts/

# 定义变量
MODEL_PATH="xxx/ATF-8B" 
MODEL_NAME=$(basename "$MODEL_PATH")

# calculate pass@NUM_RUNS
NUM_RUNS=16

# Maximum Revision Iterations, default = 4
MAX_ITERATIONS=4

INPUT_DIR="../data"

OUTPUT_DIR="../data/atf_results/${MODEL_NAME}/"

QWEN3_PATH="xxx/Qwen3-32B"

QWQ_PATH="xxx/QWQ-32B"

LOG_DIR="../data/atf_results/${MODEL_NAME}/logs"

python3 scripts/ATF_main.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --model $MODEL_PATH --log_dir $LOG_DIR --qwen3_path $QWEN3_PATH --qwq_path $QWQ_PATH --num_runs $NUM_RUNS

python3 scripts/cat_res.py --input_dir $OUTPUT_DIR --max_iterations $MAX_ITERATIONS --num_runs $NUM_RUNS
