
from typing import Dict, List
from verify_api import batch_verify_lean_codes
from baseline.utils import read_jsonl, write_jsonl, basic_check, extract_last_lean4_code_block
import argparse
import math
from typing import List, Dict
'''
输入格式
{
    'id': 唯一标识符,
    'informal_statement': 自然语言定理,
    ...
}

输出格式
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
'''

def calculate_pass_at_k_exact(data_list: List[Dict], k_values=[1, 4, 8, 16]):
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
            pass_list = item.get('pass', [])
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
    else:
        results = calculate_pass_at_k_with_sampling(data_list)
        print("Pass@K Results (Random Sampling):")
        res_text += "Pass@K Results (Random Sampling):\n"

    print("-" * 30)
    res_text += "-" * 30 + '\n'
    for metric, value in results.items():
        print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
        res_text += f"{metric}: {value:.4f} ({value*100:.2f}%)\n"
    
    return res_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=40, help="")
    parser.add_argument("--block_size", type=int, default=4000, help="")
    args = parser.parse_args()


    data = read_jsonl(args.input_path)

    all_statements = []
    id2index = {}
    for index, d in enumerate(data):
        try:
            id = d['source_data'] + str(d['index_in_source_data'])
        except:
            id = d['id']
        id2index[id] = index
        d['pass'] = [None]* len(d['formal_statements_generated'])
        for i, p in enumerate(d['formal_statements_generated']):
            p = p.replace('```Lean4','```lean4')
            if '```lean4' in p:
                p = extract_last_lean4_code_block(p)
            dic = {'id': id, 'statement':p.replace('```lean4\n','').replace('\n```',''), 'index':i ,'pass': None}
            if 'import Mathlib' not in dic['statement']:
                d['formal_statements_generated'][i] = 'import Mathlib\n' + d['formal_statements_generated'][i]
                dic['statement'] = 'import Mathlib\n'+dic['statement']
            all_statements.append(dic)
    final_results = []

    test_statements = []

    basic_fail_num = 0
    for s in all_statements:
        if not basic_check(s['statement']):
            s['pass'] = False
            basic_fail_num+= 1
            final_results.append(s)
        else:
            test_statements.append(s)
    print('basic_fail_num = ', basic_fail_num)

    test_results = []
    test_compile_results = []

    for i in range(0, len(test_statements), args.block_size):
        batch_codes = [item['statement'] for item in test_statements[i:i+args.block_size]]
        batch_results = batch_verify_lean_codes(batch_codes)
        for r in batch_results:
            test_results.append(r['pass'])
            test_compile_results.append(r)

    for s, r in zip(test_statements, test_results):
        if r is not None:
            s['pass'] = r
        else:
            s['pass'] = False
        final_results.append(s)
    
    save_path= args.output_path if len(args.output_path) > 0 else args.input_path.replace('.jsonl','_checkin.jsonl')

    for r in final_results:
        data_index = id2index[r['id']]
        statements_index = r['index']

        data[data_index]['pass'][statements_index] = r['pass']

    c = 0
    for d in data:
        for i in range(len(d['pass'])):
            if d['pass'][i] is None:
                d['pass'][i] = False

    write_jsonl(data, save_path, 'w')

    passk_save_path = save_path.replace('.jsonl','_syntax_pass@k_results.txt')

    
    results_exact = print_pass_at_k_results(data, method='exact')

    with open(passk_save_path, 'w', encoding="utf-8") as f:
        f.write(results_exact + '\n')
        f.close()