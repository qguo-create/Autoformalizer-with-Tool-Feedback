#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/workdir/scripts/

pip3 install -r scripts/lean_api_requirements.txt

# 定义变量
MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/sft_save/1008/ATF-8B-mix-thinking"
MODEL_NAME=$(basename "$MODEL_PATH")

INPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/prover/atf_opensource/data"
OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/prover/atf_opensource/data/atf_results/${MODEL_NAME}/"
QWEN3_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/model/Qwen3-32B"
QWQ_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/model/QWQ-32B"
LOG_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/prover/atf_opensource/atf_results/${MODEL_NAME}/logs"

python3 scripts/ATF_main.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --model $MODEL_PATH --log_dir $LOG_DIR --qwen3_path $QWEN3_PATH --qwq_path $QWQ_PATH --num_runs 1


