from tqdm import tqdm
import argparse
from baseline.utils import read_jsonl
import math
from typing import List, Dict

def calculate_pass_at_k_exact(data_list: List[Dict], k_values=[1, 4, 8, 16], model='qwq'):
    """
    使用精确公式计算 pass@k（更高效的方法）
    pass@k = 1 - C(n-c, k) / C(n, k)
    其中 n 是总样本数，c 是通过的样本数
    """
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
            n = len(pass_list)  # 总样本数
            c = sum(pass_list)  # 通过的样本数
            
            # 如果没有足够的样本，跳过这个问题
            if n < k:
                continue
                
            valid_problems += 1
            
            # 使用精确公式计算 pass@k
            if c == 0:
                problem_pass_at_k = 0
            else:
                problem_pass_at_k = 1 - combination(n - c, k) / combination(n, k)
            
            total_score += problem_pass_at_k
        
        # 计算总体 pass@k
        if valid_problems > 0:
            pass_at_k = total_score / valid_problems
            results[f'pass@{k}'] = pass_at_k
        else:
            results[f'pass@{k}'] = 0.0
    
    return results

def print_pass_at_k_results(data_list: List[Dict], method='exact'):
    """
    打印 pass@k 结果
    
    Args:
        data_list: 数据列表
        method: 'exact' 或 'sampling'
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
    parser.add_argument("--res_save_path", type=str, required=True)
    args = parser.parse_args()

    qwq_path, qwen3_path = args.qwq_path, args.qwen3_path

    qwq_data = read_jsonl(qwq_path)

    qwen3_data = read_jsonl(qwen3_path)

    for qwq, qwen3 in zip(qwq_data, qwen3_data):
        qwq['consistency']['qwen3'] = qwen3['consistency']['/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/model/Qwen3-32B']
        qwq['consistency']['qwq'] = qwq['consistency']['/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/model/QWQ-32B']
        del qwq['consistency']['/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/o1/guoqi/model/QWQ-32B']


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