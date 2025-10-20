#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/workdir/scripts/

# Formalizer Path
MODEL_ROOT="xxx"

MODEL_PATHS=("${MODEL_ROOT}/Kimina-Autoformalizer-7B" "${MODEL_ROOT}/StepFun-Formalizer-7B" "${MODEL_ROOT}/Goedel-Formalizer-V2-8B" "${MODEL_ROOT}/Goedel-Formalizer-V2-32B" "${MODEL_ROOT}/StepFun-Formalizer-32B")

DATA_NAMES=("combibench" "proverbench" "formalmath-lite")

TIMES=(1)
DATA_PATH="../data"
QWQ_PATH="xxx/QWQ-32B"
QWEN3_PATH="xxx/Qwen3-32B"

GPU_NUM=8
NUM=16
FORMALIZATION_PARTITION_NUM=4
CONSISTENCY_PARTITION_NUM=4
TEMPERATURE=0.6

for times in "${TIMES[@]}"; do
    for MODEL_PATH in "${MODEL_PATHS[@]}"; do
        MODEL_NAME=$(basename "$MODEL_PATH")
        echo "MODEL_NAME is: $MODEL_NAME"
        # Iterate through each DATA_NAME
        for DATA_NAME in "${DATA_NAMES[@]}"; do
            echo "Process DATA_NAME: $DATA_NAME"

            # Check and create directory
            if [ ! -d "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}" ]; then
                mkdir -p "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}"
                echo "Directory $DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times} created."
            else
                echo "Directory $DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times} already exists."
            fi

            # Check if ${MODEL_NAME}.jsonl exists in temp directory
            if [ -f "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}.jsonl" ]; then
                echo "File $DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}.jsonl exists. Skipping copy."
            else
                echo "Copy $DATA_PATH/${DATA_NAME}.jsonl 到 $DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/"
                cp "$DATA_PATH/${DATA_NAME}.jsonl" "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/"
            fi

            # Check split files
            if [ -f "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}_part_0.jsonl" ]; then
                echo "File $DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}_part_0.jsonl exists. Skipping partitioning."
            else
                echo "Partition ${DATA_NAME}.jsonl into $FORMALIZATION_PARTITION_NUM files"
                python3 scripts/baseline/partition.py --input_path "$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}.jsonl" --n $FORMALIZATION_PARTITION_NUM
            fi

            # Process each partitioned jsonl file with formalization.py

            GPU_PER_PART=$((GPU_NUM / FORMALIZATION_PARTITION_NUM))

            # Function to generate GPU allocation
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
                    echo "Processing $part_file with formalization.py using GPUs $gpu_ids. Output: $output_file, Log: $log_file"
                    nohup python3 scripts/baseline/formalization.py --input_path "$part_file" --output_path "$output_file" --gpu $gpu_ids --n $NUM --temperature $TEMPERATURE --model $MODEL_PATH > "$log_file" 2>&1 &
                    pids+=($!)
                    part_idx=$((part_idx+1))
                fi
            done

            # Wait for all processes
            for pid in "${pids[@]}"; do
                wait $pid
            done

            echo "All formalization.py processes completed."

            # Merge output files
            merged_output="$DATA_PATH/${DATA_NAME}/${MODEL_NAME}_num=${NUM}_t=${TEMPERATURE}_temp_${times}/${DATA_NAME}_with_statements.jsonl"
            if [ -f "$merged_output" ]; then
                echo "File $merged_output exists. Skipping merge."
            else
                input_path_list=$(IFS=,; echo "${output_files[*]}")
                echo "Merging files to $merged_output"
                python3 scripts/baseline/merge.py --input_path_list "$input_path_list" --output_path "$merged_output"
            fi

            # Run lean4_checkin.py
            checkin_output="${merged_output/.jsonl/_checkin.jsonl}"
            echo "Running lean4_checkin.py. Input: $merged_output, Output: $checkin_output"
            python3 scripts/baseline/lean4_checkin.py --input_path "$merged_output" --output_path "$checkin_output"


            # Split checkin output
            checkin_part_prefix="${checkin_output%.jsonl}_part_"
            if [ -f "${checkin_part_prefix}0.jsonl" ]; then
                echo "File ${checkin_part_prefix}0.jsonl exists. Skipping split."
            else
                echo "Partition $checkin_output into $CONSISTENCY_PARTITION_NUM files"
                python3 scripts/baseline/partition.py --input_path "$checkin_output" --n $CONSISTENCY_PARTITION_NUM
            fi


            GPU_PER_PART=$((GPU_NUM / CONSISTENCY_PARTITION_NUM))

            # Allocate gpus
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

            # Process consistency check with QWQ model
            pids=()
            part_idx=0
            for part_file in "${checkin_output%.jsonl}_part_"[0-9].jsonl; do
                if [ -f "$part_file" ]; then
                    log_file="${part_file%.jsonl}_consistency_QWQ.log"
                    output_file="${part_file%.jsonl}_consistency_QWQ.jsonl"
                    gpu_ids=$(get_gpu_ids $part_idx)
                    echo "Processing $part_file with QWQ model (GPUs $gpu_ids). Output: $output_file, Log: $log_file"
                    nohup python3 scripts/baseline/consistency_judge.py --input_path "$part_file" --output_path "$output_file" --model "$QWQ_PATH" --gpu "$gpu_ids" > "$log_file" 2>&1 &
                    pids+=($!)
                    part_idx=$((part_idx+1))
                fi
            done

            for pid in "${pids[@]}"; do
                wait $pid
            done


            # Merge QWQ results
            q_consistency_files=()
            for part_file in "${checkin_output%.jsonl}_part_"[0-9].jsonl; do
                q_consistency_file="${part_file%.jsonl}_consistency_QWQ.jsonl"
                if [ -f "$q_consistency_file" ]; then
                    q_consistency_files+=("$q_consistency_file")
                fi
            done

            merged_q_consistency="${checkin_output%.jsonl}_consistency_QWQ.jsonl"
            if [ -f "$merged_q_consistency" ]; then
                echo "File $merged_q_consistency exists. Skipping merge."
            else
                input_path_list=$(IFS=,; echo "${q_consistency_files[*]}")
                echo "Merging QWQ results to $merged_q_consistency"
                python3 scripts/baseline/merge.py --input_path_list "$input_path_list" --output_path "$merged_q_consistency"
            fi


            # Process consistency check with QWEN3 model
            pids=()
            part_idx=0
            for part_file in "${checkin_output%.jsonl}_part_"[0-9].jsonl; do
                if [ -f "$part_file" ]; then
                    log_file="${part_file%.jsonl}_consistency_QWEN3.log"
                    output_file="${part_file%.jsonl}_consistency_QWEN3.jsonl"
                    gpu_ids=$(get_gpu_ids $part_idx)
                    echo "Processing $part_file with QWEN3 model (GPUs $gpu_ids). Output: $output_file, Log: $log_file"
                    nohup python3 scripts/baseline/consistency_judge.py --input_path "$part_file" --output_path "$output_file" --model "$QWEN3_PATH" --gpu "$gpu_ids" > "$log_file" 2>&1 &
                    pids+=($!)
                    part_idx=$((part_idx+1))
                fi
            done

            for pid in "${pids[@]}"; do
                wait $pid
            done

            # Merge QWEN3 results
            qwen3_consistency_files=()
            for part_file in "${checkin_output%.jsonl}_part_"[0-9].jsonl; do
                qwen3_consistency_file="${part_file%.jsonl}_consistency_QWEN3.jsonl"
                if [ -f "$qwen3_consistency_file" ]; then
                    qwen3_consistency_files+=("$qwen3_consistency_file")
                fi
            done

            merged_qwen3_consistency="${checkin_output%.jsonl}_consistency_QWEN3.jsonl"
            if [ -f "$merged_qwen3_consistency" ]; then
                echo "File $merged_qwen3_consistency exists. Skipping merge."
            else
                input_path_list=$(IFS=,; echo "${qwen3_consistency_files[*]}")
                echo "Merging QWEN3 results to $merged_qwen3_consistency"
                python3 scripts/baseline/merge.py --input_path_list "$input_path_list" --output_path "$merged_qwen3_consistency"
            fi

            consistency_res_save_path="${checkin_output%.jsonl}_consistency_results.txt"

            if [ -f "$consistency_res_save_path" ]; then
                echo "File $consistency_res_save_path exists. Skipping analysis."
            else
                echo "Generating final report at $consistency_res_save_path"
                python3 scripts/baseline/final_process.py --qwq_res_path "$merged_q_consistency" --qwen3_res_path "$merged_qwen3_consistency" --res_save_path "$consistency_res_save_path" --qwq_path $QWQ_PATH --qwen3_path $QWEN3_PATH
            fi
            echo "Completed processing DATA_NAME: $DATA_NAME"
        done
        echo "Completed processing MODEL_NAME: $MODEL_NAME"
    done
    echo "Completed iteration: $times"
done
echo "All data processing completed."