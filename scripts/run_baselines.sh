#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/workdir/scripts/

pip3 install -r scripts/lean_api_requirements.txt

pip3 freeze > /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/atf_requirements.txt

# 定义变量
MODEL_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/model"
# MODEL_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/sft_save/1008"

# MODEL_PATHS=("${MODEL_ROOT}/Kimina-Autoformalizer-7B" "${MODEL_ROOT}/StepFun-Formalizer-7B" "${MODEL_ROOT}/Goedel-Formalizer-V2-8B" "${MODEL_ROOT}/Goedel-Formalizer-V2-32B" "${MODEL_ROOT}/StepFun-Formalizer-32B")
MODEL_PATHS=("${MODEL_ROOT}/Kimina-Autoformalizer-7B" "${MODEL_ROOT}/StepFun-Formalizer-7B")
# MODEL_PATHS=("${MODEL_ROOT}/ATF-32B-mix-thinking")

# DATA_NAMES=("combibench" "proverbench" "formalmath-lite")
DATA_NAMES=("combibench" "proverbench")
TIMES=(1)
DATA_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/prover/atf_opensource/data"
QWQ_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/model/QWQ-32B"
QWEN3_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/model/Qwen3-32B"

GPU_NUM=4
NUM=16
FORMALIZATION_PARTITION_NUM=4
CONSISTENCY_PARTITION_NUM=4
TEMPERATURE=0.6

for times in "${TIMES[@]}"; do
    for MODEL_PATH in "${MODEL_PATHS[@]}"; do
        MODEL_NAME=$(basename "$MODEL_PATH")
        echo "MODEL_NAME is: $MODEL_NAME"
        # 遍历每个DATA_NAME
        for DATA_NAME in "${DATA_NAMES[@]}"; do
            echo "处理 DATA_NAME: $DATA_NAME"

            # 检查并创建文件夹
            if [ ! -d "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}" ]; then
                mkdir -p "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}"
                echo "目录 $DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times} 已创建。"
            else
                echo "目录 $DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times} 已存在。"
            fi

            # 检查 temp 文件夹下是否存在 ${MODEL_NAME}.jsonl
            if [ -f "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}.jsonl" ]; then
                echo "文件 $DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}.jsonl 已存在，跳过复制。"
            else
                echo "复制 $DATA_PATH/${DATA_NAME}.jsonl 到 $DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/"
                cp "$DATA_PATH/${DATA_NAME}.jsonl" "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/"
            fi

            # 检查分割文件是否存在
            if [ -f "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}_part_0.jsonl" ]; then
                echo "文件 $DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}_part_0.jsonl 已存在，跳过分割。"
            else
                echo "Partition ${DATA_NAME}.jsonl into $FORMALIZATION_PARTITION_NUM files"
                python3 scripts/baseline/partition.py --input_path "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}.jsonl" --n $FORMALIZATION_PARTITION_NUM
            fi

            # 遍历每个分割后的jsonl文件，调用 formalization.py

            GPU_PER_PART=$((GPU_NUM / FORMALIZATION_PARTITION_NUM))

            # 生成GPU分配字符串的函数
            get_gpu_ids() {
                local idx=$1
                local num_groups=$((GPU_NUM / GPU_PER_PART))
                local group_idx=$((idx % num_groups))
                local start=$((group_idx * GPU_PER_PART))
                local end=$((start + GPU_PER_PART - 1))
                local ids=""
                for ((i=start; i<=end; i++)); do
                    if [ -z "$ids" ]; then
                        ids="$i"
                    else
                        ids="$ids,$i"
                    fi
                done
                echo "$ids"
            }

            pids=()
            part_idx=0
            output_files=()
            for part_file in "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}_part_"[0-9].jsonl; do
                if [ -f "$part_file" ]; then
                    gpu_ids=$(get_gpu_ids $part_idx)
                    log_file="${part_file}.log"
                    output_file="${part_file/.jsonl/_with_statements.jsonl}"
                    output_files+=("$output_file")
                    echo "用nohup调用 formalization.py 处理 $part_file，指定GPU $gpu_ids，输出到 $output_file，日志输出到 $log_file"
                    nohup python3 scripts/baseline/formalization.py --input_path "$part_file" --output_path "$output_file" --gpu $gpu_ids --n $NUM --temperature $TEMPERATURE --model $MODEL_PATH > "$log_file" 2>&1 &
                    pids+=($!)
                    part_idx=$((part_idx+1))
                fi
            done

            # 等待所有python进程执行完毕
            for pid in "${pids[@]}"; do
                wait $pid
            done

            echo "全部 formalization.py 执行完成。"

            # 合并所有 output_path 文件
            merged_output="$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}_with_statements.jsonl"
            if [ -f "$merged_output" ]; then
                echo "文件 $merged_output 已存在，跳过合并。"
            else
                # 构造input_path_list参数
                input_path_list=$(IFS=,; echo "${output_files[*]}")
                echo "调用 merge.py 合并文件，输出到 $merged_output"
                python3 scripts/baseline/merge.py --input_path_list "$input_path_list" --output_path "$merged_output"
            fi

            # 调用 lean4_checkin.py
            checkin_output="${merged_output/.jsonl/_checkin.jsonl}"
            echo "调用 lean4_checkin.py，输入文件为 $merged_output，输出文件为 $checkin_output"
            python3 scripts/baseline/lean4_checkin.py --input_path "$merged_output" --output_path "$checkin_output"


            # 拆分 checkin_output
            checkin_part_prefix="${checkin_output%.jsonl}_part_"
            if [ -f "${checkin_part_prefix}0.jsonl" ]; then
                echo "文件 ${checkin_part_prefix}0.jsonl 已存在，跳过拆分。"
            else
                echo "Partition $checkin_output into $CONSISTENCY_PARTITION_NUM files"
                python3 scripts/baseline/partition.py --input_path "$checkin_output" --n $CONSISTENCY_PARTITION_NUM
            fi


            GPU_PER_PART=$((GPU_NUM / CONSISTENCY_PARTITION_NUM))

            # 生成GPU分配字符串的函数
            get_gpu_ids() {
                local idx=$1
                local num_groups=$((GPU_NUM / GPU_PER_PART))
                local group_idx=$((idx % num_groups))
                local start=$((group_idx * GPU_PER_PART))
                local end=$((start + GPU_PER_PART - 1))
                local ids=""
                for ((i=start; i<=end; i++)); do
                    if [ -z "$ids" ]; then
                        ids="$i"
                    else
                        ids="$ids,$i"
                    fi
                done
                echo "$ids"
            }

            # 并行执行 consistency_judge.py（QWQ模型）
            pids=()
            part_idx=0
            for part_file in "${checkin_output%.jsonl}_part_"[0-9].jsonl; do
                if [ -f "$part_file" ]; then
                    log_file="${part_file%.jsonl}_consistency_QWQ.log"
                    output_file="${part_file%.jsonl}_consistency_QWQ.jsonl"
                    gpu_ids=$(get_gpu_ids $part_idx)
                    echo "用nohup调用 consistency_judge.py 处理 $part_file，模型参数为QWQ，GPU参数为$gpu_ids，输出为$output_file，日志输出到 $log_file"
                    nohup python3 scripts/baseline/consistency_judge.py --input_path "$part_file" --output_path "$output_file" --model "$QWQ_PATH" --gpu "$gpu_ids" > "$log_file" 2>&1 &
                    pids+=($!)
                    part_idx=$((part_idx+1))
                fi
            done

            # 等待所有 consistency_judge.py 执行完毕
            for pid in "${pids[@]}"; do
                wait $pid
            done


            # 合并所有 QWQ consistency 结果
            q_consistency_files=()
            for part_file in "${checkin_output%.jsonl}_part_"[0-9].jsonl; do
                q_consistency_file="${part_file%.jsonl}_consistency_QWQ.jsonl"
                if [ -f "$q_consistency_file" ]; then
                    q_consistency_files+=("$q_consistency_file")
                fi
            done

            merged_q_consistency="${checkin_output%.jsonl}_consistency_QWQ.jsonl"
            if [ -f "$merged_q_consistency" ]; then
                echo "文件 $merged_q_consistency 已存在，跳过合并。"
            else
                input_path_list=$(IFS=,; echo "${q_consistency_files[*]}")
                echo "调用 merge.py 合并 QWQ consistency 结果，输出到 $merged_q_consistency"
                python3 scripts/baseline/merge.py --input_path_list "$input_path_list" --output_path "$merged_q_consistency"
            fi


            # 并行执行 consistency_judge.py（QWEN3模型）
            pids=()
            part_idx=0
            for part_file in "${checkin_output%.jsonl}_part_"[0-9].jsonl; do
                if [ -f "$part_file" ]; then
                    log_file="${part_file%.jsonl}_consistency_QWEN3.log"
                    output_file="${part_file%.jsonl}_consistency_QWEN3.jsonl"
                    gpu_ids=$(get_gpu_ids $part_idx)
                    echo "用nohup调用 consistency_judge.py 处理 $part_file，模型参数为QWEN3，GPU参数为$gpu_ids，输出为$output_file，日志输出到 $log_file"
                    nohup python3 scripts/baseline/consistency_judge.py --input_path "$part_file" --output_path "$output_file" --model "$QWEN3_PATH" --gpu "$gpu_ids" > "$log_file" 2>&1 &
                    pids+=($!)
                    part_idx=$((part_idx+1))
                fi
            done

            # 等待所有 consistency_judge.py 执行完毕
            for pid in "${pids[@]}"; do
                wait $pid
            done

            # 合并所有 QWEN3 consistency 结果
            qwen3_consistency_files=()
            for part_file in "${checkin_output%.jsonl}_part_"[0-9].jsonl; do
                qwen3_consistency_file="${part_file%.jsonl}_consistency_QWEN3.jsonl"
                if [ -f "$qwen3_consistency_file" ]; then
                    qwen3_consistency_files+=("$qwen3_consistency_file")
                fi
            done

            merged_qwen3_consistency="${checkin_output%.jsonl}_consistency_QWEN3.jsonl"
            if [ -f "$merged_qwen3_consistency" ]; then
                echo "文件 $merged_qwen3_consistency 已存在，跳过合并。"
            else
                input_path_list=$(IFS=,; echo "${qwen3_consistency_files[*]}")
                echo "调用 merge.py 合并 QWEN3 consistency 结果，输出到 $merged_qwen3_consistency"
                python3 scripts/baseline/merge.py --input_path_list "$input_path_list" --output_path "$merged_qwen3_consistency"
            fi

            consistency_res_save_path="${checkin_output%.jsonl}_consistency_results.txt"


            if [ -f "$consistency_res_save_path" ]; then
                echo "文件 $consistency_res_save_path 已存在，跳过统计。"
            else
                echo "调用 final_process.py 统计 consistency 结果，输出到 $consistency_res_save_path"
                python3 scripts/baseline/final_process.py --qwq_path "$merged_q_consistency" --qwen3_path "$merged_qwen3_consistency" --res_save_path "$consistency_res_save_path"
            fi
            echo "DATA_NAME $DATA_NAME 的处理已完成。"
        done
        echo "MODEL_NAME $MODEL_NAME 的处理已完成。"
    done
    echo "TIMES $times 的处理已完成。"
done
echo "所有数据集处理完毕。"