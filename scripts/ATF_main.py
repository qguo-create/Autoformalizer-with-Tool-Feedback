import os
import json
import time
import argparse
from transformers import AutoTokenizer
from atf.utils import load_model_on_gpus
from atf.evaluation import run_multi_dataset_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--qwen3_path", type=str, required=True)
    parser.add_argument("--qwq_path", type=str, required=True)

    parser.add_argument("--num_runs", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_iterations", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_prompt_length", type=int, default=1024*40)
    args = parser.parse_args()

    seed = 12

    try:
        print(f"Initializing models (seed={seed})...")
        qwen3_model = load_model_on_gpus(args.qwen3_path, [0], "qwen3", seed)
        qwq_model = load_model_on_gpus(args.qwq_path, [1], "qwq", seed)
        llm1 = load_model_on_gpus(args.model, [2], "atf1", seed)
        llm2 = load_model_on_gpus(args.model, [3], "atf2", seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        models = {
            'atf1': llm1,
            'atf2': llm2,
            'qwq_model': qwq_model,
            'qwen3_model': qwen3_model
        }
        
        print("✓ All models initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")

    # Execute multi-dataset evaluation
    run_multi_dataset_evaluation(args, models, tokenizer)
