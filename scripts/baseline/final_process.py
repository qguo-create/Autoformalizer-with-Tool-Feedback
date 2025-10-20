from tqdm import tqdm
import argparse
from baseline.utils import read_jsonl
import math
from typing import List, Dict

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
            pass_list = item['consistency']['labels']
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

def print_pass_at_k_results(data_list: List[Dict], method='exact'):
    """
    print pass@k
    """
    res_text = ''
    if method == 'exact':
        results = calculate_pass_at_k_exact(data_list)
        print("Pass@K Results (Exact Formula):")
        res_text += "Pass@K Results (Exact Formula):\n"

    print("-" * 30)
    res_text += "-" * 30 + '\n'
    for metric, value in results.items():
        print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
        res_text += f"{metric}: {value:.4f} ({value*100:.2f}%)\n"
    
    return res_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwq_path", type=str, required=True)
    parser.add_argument("--qwen3_path", type=str, required=True)
    parser.add_argument("--qwq_res_path", type=str, required=True)
    parser.add_argument("--qwen3_res_path", type=str, required=True)
    parser.add_argument("--res_save_path", type=str, required=True)
    args = parser.parse_args()

    qwq_data = read_jsonl(args.qwq_res_path)

    qwen3_data = read_jsonl(args.qwen3_res_path)

    for qwq, qwen3 in zip(qwq_data, qwen3_data):
        qwq['consistency']['qwen3'] = qwen3['consistency'][args.qwen3_path]
        qwq['consistency']['qwq'] = qwq['consistency'][args.qwq_path]

    for d in tqdm(qwq_data):
        d['consistency']['labels'] = []
        for i in range(len(d['consistency']['qwq'])):
            l1 = '"correct"' in d['consistency']['qwq'][i].split('"is_assistant_correct"')[-1].lower()
            l2 = '"correct"' in d['consistency']['qwen3'][i].split('"is_assistant_correct"')[-1].lower()
            if l1 == True and l2 == True:
                d['consistency']['labels'].append(True)
            else:
                d['consistency']['labels'].append(False)

    results_exact = print_pass_at_k_results(qwq_data, method='exact')

    with open(args.res_save_path, 'w', encoding="utf-8") as f:
        f.write(results_exact + '\n')
        f.close()