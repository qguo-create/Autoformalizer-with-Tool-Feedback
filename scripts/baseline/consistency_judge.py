from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import sys
import argparse
from transformers import AutoTokenizer
from baseline.utils import read_jsonl, write_jsonl, write_res, extract_last_lean4_code_block
from template import consistency_template
import math
from typing import List, Dict

'''
Input
{
    'id':xxx,
    'informal_statement':xxx
    'formalization_prompt': xxx,
    'formal_statements_generated':[
        xxx,
        xxx,
        ...
    ],
    'pass':[
        True,
        False,
        ...
    ]
}

Output
{
    'id':xxx,
    'informal_statement':xxx
    'formalization_prompt': xxx,
    'formal_statements_generated':[
        xxx,
        xxx,
        ...
    ],
    'pass':[
        True,
        False,
        ...
    ]
    'consistency':[
        'qwq':[
            responses 1,
            '',
            ...
        ],
        'qwen3':
        [
            responses 1,
            '',
            ...
        ]
        ...
    ]
}
'''

def get_prompt(tokenizer, informal_statement, formal_statement):

    messages = [
    {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
    {"role": "user", "content": consistency_template.replace('{informal_statement}',informal_statement).replace('{formal_statement}',formal_statement)}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text

def calculate_pass_at_k_exact(data_list: List[Dict], k_values=[1, 4, 8, 16], model='qwq'):
    def combination(n, r):
        if r > n or r < 0:
            return 0
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
    
    results = {}
    
    for k in k_values:
        total_score = 0
        valid_problems = 0
        
        for item in data_list:
            pass_list = ['"correct"' in r.split('"is_assistant_correct"')[-1].lower() for r in item['consistency'][model]]
            n = len(pass_list) 
            c = sum(pass_list) 
    
            if n < k:
                continue
                
            valid_problems += 1
            
  
            if c == 0:
                problem_pass_at_k = 0
            else:
                problem_pass_at_k = 1 - combination(n - c, k) / combination(n, k)
            
            total_score += problem_pass_at_k
        
        if valid_problems > 0:
            pass_at_k = total_score / valid_problems
            results[f'pass@{k}'] = pass_at_k
        else:
            results[f'pass@{k}'] = 0.0
    
    return results

def print_pass_at_k_results(data_list: List[Dict], method='exact',model='qwq'):
    """
    print pass@k
    """
    if method == 'exact':
        results = calculate_pass_at_k_exact(data_list, model=model)
        print("Pass@K Results (Exact Formula):")

    
    print("-" * 30)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--gpu",type=str,required=True)
    parser.add_argument("--n", type=int, default=1, help="")
    parser.add_argument("--max_length", type=int, default=8192+4096, help="")
    parser.add_argument("--batch_size", type=int, default=1024*8, help="")
    parser.add_argument("--temperature", type=float, default=0, help="")
    parser.add_argument("--only_save_prompts", type=bool, default=False, help="")
    args = parser.parse_args()

    model_path = args.model
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}" 

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote=True)
    
    data = read_jsonl(args.input_path)

    model_name = args.model.split('/')[-1]
    save_path= args.output_path if len(args.output_path) > 0 else args.input_path.replace('.jsonl',f'_consistency_{model_name}.jsonl')

    prompts = []
    id2index = {}
    for index, d in enumerate(data):
        try:
            id = d['source_data'] + str(d['index_in_source_data'])
        except:
            id = d['id']
        id2index[id] = index
        for i, s, p in zip(range(len(d['formal_statements_generated'])),d['formal_statements_generated'], d['pass']):
            if p:
                s = s.replace('```Lean4','```lean4')
                if '```lean4' in s:
                    s = extract_last_lean4_code_block(s)
                prompt = get_prompt(tokenizer, informal_statement=d['informal_statement'], formal_statement=s)
                prompts.append({'id':id, 'index': i,'prompt':prompt})

    if args.only_save_prompts:
        write_jsonl(prompts, args.input_path.replace('.jsonl',f'_prompts_{args.model}.jsonl'),'w')
    else:
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_length, n=args.n)

        if os.path.exists(save_path):
            shift = len(read_jsonl(save_path))
        else:
            shift = 0

        print(f"start at shift = {shift}")
        if shift < len(prompts):
            print("load model")

            llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.95, dtype="bfloat16", swap_space=16, disable_custom_all_reduce=True,seed=12)

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

        all_responses = read_jsonl(save_path)
        for d in data:
            if not 'consistency' in d:
                d['consistency'] = {}
            if not args.model in d['consistency']:
                d['consistency'][args.model] = [''] * len(d['pass'])

        for p, r in zip(prompts, all_responses):
            data_index = id2index[p['id']]
            statement_index = p['index']
            if isinstance(r,list):
                data[data_index]['consistency'][args.model][statement_index] = r[0]
            else:
                data[data_index]['consistency'][args.model][statement_index] = r
        
        write_jsonl(data, save_path, 'w')

        results_exact = print_pass_at_k_results(data, method='exact',model=args.model)