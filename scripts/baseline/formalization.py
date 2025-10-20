
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import argparse
import time
from transformers import AutoTokenizer
from baseline.utils import read_jsonl, write_jsonl, write_res
from template import ATF_system_prompt


'''
Input format
{   'id': unique identifier,
    'informal_statement': natural language theorem,
    ...
}

Output format
{   
    'id':xxx,
    'informal_statement':xxx
    'formalization_prompt': xxx,
    'formal_statements_generated':[
        xxx,
        xxx,
        ...
    ]
}
'''

SYSTEM_PROMPT = {
    'goedel': "You are an expert in mathematics and Lean 4.",
    'kimina': "You are an expert in mathematics and Lean 4.",
    'qwen25': "You are an expert in mathematics and Lean 4.",
    'qwen3': "You are an expert in mathematics and Lean 4.",
    'stepfun': "You are an expert in mathematics and Lean 4.",
    'atf': ATF_system_prompt,
}

INPUT_PREFIX = {
    'goedel': "Please autoformalize the following natural language problem statement in Lean 4. Use the following theorem name: my_favorite_theorem.\nThe natural language statement is: \n",
    'kimina': "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n",
    'qwen25': "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n",
    'qwen3': "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n",
    'stepfun': "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n", 
    'atf': "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"
}

INPUT_SUFFIX = {
    'goedel': "Think before you provide the lean statement.",
    'kimina': "",
    'qwen25': "",
    'qwen3': "",
    'stepfun': "\n\nYour code should start with:\n```Lean4\nimport Mathlib\nimport Aesop\n```\n",
    'atf': ""
}

VALID_MODELS = ['goedel', 'kimina', 'qwen25', 'qwen3', 'stepfun', 'atf']

def get_prompt(tokenizer, model_key, d):
    prompt = INPUT_PREFIX[model_key] + d['informal_statement'] + INPUT_SUFFIX[model_key]
    messages = [
    {"role": "system", "content": SYSTEM_PROMPT[model_key]},
    {"role": "user", "content": prompt}
    ]
    if model_key in ['qwen3','atf']:   
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--gpu", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    model_path = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote=True)
    
    data = read_jsonl(args.input_path)
    model_name = args.model.split('/')[-1]
    save_path= args.output_path

    model_key = ''

    for m in VALID_MODELS:
        if m in model_path.lower():
            model_key = m
            break
    print('model key = ', model_key)
    prompts = []

    for d in data:
        try:
            id = d['source_data'] + str(d['index_in_source_data'])
        except:
            id = d['id']
        prompt = get_prompt(tokenizer, model_key, d)
        prompts.append({'id':id, 'prompt':prompt})

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length, n=args.n)
    
    if os.path.exists(save_path):
        shift = len(read_jsonl(save_path))
    else:
        shift = 0

    print(f"start at shift = {shift}")

    print("load model")

    seed = int(time.time()) % (2**32)

    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9, dtype="bfloat16", swap_space=16, disable_custom_all_reduce=True,seed=seed)

    batch_inputs = []
    for i in range(len(prompts)):
        if i < shift:
            continue
        batch_inputs.append(prompts[i])
        if len(batch_inputs) >= args.batch_size:
            print(f"====== {i - args.batch_size} ～ {i} / {len(prompts)} ======")
            batch_outputs = llm.generate(batch_inputs, sampling_params)
            batch_res = []
            for idx, output in enumerate(batch_outputs):
                batch_res.append([output.outputs[_].text for _ in range(len(output.outputs))])
            write_res(save_path, batch_res=batch_res)
            batch_inputs = []

    if len(batch_inputs) > 0:
        print(f"======{len(prompts)} / {len(prompts)} ======")
        batch_outputs = llm.generate(batch_inputs, sampling_params)
        batch_res = []
        for idx, output in enumerate(batch_outputs):
                batch_res.append([output.outputs[_].text for _ in range(len(output.outputs))])
        write_res(save_path, batch_res=batch_res)

    all_responses = read_jsonl(save_path)

    for d, p, r in zip(data, prompts, all_responses):
        d['formalization_prompt'] = p['prompt']
        d['formal_statements_generated'] = [item.replace('```Lean4\n','```lean4\n') for item in r]
    
    write_jsonl(data, save_path, 'w')