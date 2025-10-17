import os
import json
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List

# 数据集原始大小
DATASET_SIZES = {
    'formalmath-lite': 418,
    'proverbench': 230,
    'combibench': 100
}

def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def has_syntax_pass_in_iterations(result, max_iterations=4):
    """
    检查样本是否在指定迭代次数内有语法通过
    新标准：第一次语法调用成功时，前面pass为false的工具数量 < max_iterations
    """
    tools = result.get('tools', [])
    if not tools:
        return False
    
    false_count = 0  # 统计pass为false的工具数量
    
    for i, tool_result in enumerate(tools):
        # 检查是否是语法检查结果（有pass字段）
        if isinstance(tool_result, dict) and 'pass' in tool_result:
            if tool_result.get('pass', False) and 'errors' in tool_result:
                # 找到第一次pass为True的语法检查
                # 检查前面pass为false的数量是否 < max_iterations
                return false_count < max_iterations
            else:
                # pass为false，计数器加1
                false_count += 1
    
    # 没有找到pass为True的语法检查
    return False

def get_syntax_pass_iteration(result):
    """
    获取语法通过时的迭代次数（前面pass为false的工具数量）
    """
    tools = result.get('tools', [])
    if not tools:
        return None
    
    false_count = 0
    
    for i, tool_result in enumerate(tools):
        if isinstance(tool_result, dict) and 'pass' in tool_result:
            if tool_result.get('pass', False):
                # 第一次pass为True，返回前面false的数量
                return false_count
            else:
                false_count += 1
    
    return None  # 没有找到pass为True的

def extract_syntax_passed_from_results(results, max_iterations=4):
    """从results中提取语法通过的样本"""
    syntax_passed_samples = []
    
    for result in results:
        if has_syntax_pass_in_iterations(result, max_iterations):
            syntax_pass_iteration = get_syntax_pass_iteration(result)
            
            syntax_passed_sample = {
                'index': result['index'],
                'data': result['data'],
                'syntax_pass_iteration': syntax_pass_iteration,  # 前面false的数量
                'final_completed': result.get('completed', False),
                'final_failed': result.get('failed', False),
                'final_failure_reason': result.get('failure_reason', ''),
                'total_iterations': result.get('iterations', 0),
                'informal_statement': result['data'].get('informal_statement', ''),
                'formal_statement': result['data'].get('formal_statement', ''),
            }
            syntax_passed_samples.append(syntax_passed_sample)
    
    return syntax_passed_samples

def group_results_by_run(all_results, num_runs):
    """将all_results按run分组"""
    if not all_results:
        return []
    
    # 计算每个run的样本数量
    total_samples = len(all_results)
    samples_per_run = total_samples // num_runs
    
    results_by_run = []
    for run_idx in range(num_runs):
        start_idx = run_idx * samples_per_run
        if run_idx == num_runs - 1:  # 最后一个run包含剩余的所有样本
            end_idx = total_samples
        else:
            end_idx = (run_idx + 1) * samples_per_run
        
        run_results = all_results[start_idx:end_idx]
        results_by_run.append(run_results)
    
    return results_by_run

def is_valid_tool(tools):
    """
    判断tools序列是否有效
    
    Args:
        tools: 工具调用结果列表，每个元素包含工具调用的结果
    
    Returns:
        bool: 是否有效
    """
    if not tools or len(tools) < 2:
        return False
    
    # 条件1: 最后一次调用为consistency check且倒数第二次为syntax check，均通过
    last_tool = tools[-1]
    second_last_tool = tools[-2]
    
    # 检查最后一次是否为consistency check且通过
    if not ('explanations' in last_tool and last_tool.get('pass', False)):
        return False
    
    # 检查倒数第二次是否为syntax check且通过
    if not ('errors' in second_last_tool and second_last_tool.get('pass', False)):
        return False
    
    # 条件2和3: 检查整个序列的逻辑
    for i in range(len(tools)):
        current_tool = tools[i]
        
        # 条件2: 某一次没有通过时下一次调用必须为syntax调用
        if not current_tool.get('pass', False):
            # 如果当前工具调用失败，检查下一次调用
            if i + 1 < len(tools):
                next_tool = tools[i + 1]
                if not 'errors' in next_tool:
                    return False
            # 如果是最后一次调用且失败，则无效（因为条件1要求最后一次必须通过）
            else:
                return False
        
        # 条件3: syntax调用成功后的下一次调用必须为consistency
        if 'errors' in current_tool and current_tool.get('pass', False) and i + 1 < len(tools):
            next_tool = tools[i + 1]
            if not 'explanations' in next_tool:
                return False
    
    return True

def calculate_statistics_from_all_results(all_results, dataset_name, num_runs, max_iterations=4):
    """从all_results计算统计信息"""
    if not all_results:
        return {}
    
    # 获取数据集的原始大小
    original_dataset_size = DATASET_SIZES.get(dataset_name, len(all_results) // num_runs if num_runs > 0 else len(all_results))
    print(f"  使用原始数据集大小: {original_dataset_size}, 总结果数: {len(all_results)}, 推断运行次数: {num_runs}")
    
    # 按run分组结果
    results_by_run = group_results_by_run(all_results, num_runs)
    
    filtered_runs_stats = []
    all_syntax_passed = []
    
    for run_idx, run_results in enumerate(results_by_run):
        # 统计成功的样本：completed=True 且 iterations < max_iterations
        completed_count = sum(1 for r in run_results 
                            if r.get('completed', False) and 
                            r.get('iterations', 0) < max_iterations and is_valid_tool(r.get('tools',{})))
        
        # 从结果中提取语法通过的样本（使用新标准）
        syntax_passed_samples = extract_syntax_passed_from_results(run_results, max_iterations)
        all_syntax_passed.append(syntax_passed_samples)
        syntax_passed_count = len(syntax_passed_samples)
        
        # 统计失败的样本
        failed_count = original_dataset_size - completed_count
        
        # 计算平均迭代次数（只计算iterations < max_iterations且成功的样本）
        valid_iterations = [r.get('iterations', 0) for r in run_results 
                          if r.get('iterations', 0) < max_iterations and r.get('completed', False)]
        avg_iterations = np.mean(valid_iterations) if valid_iterations else 0
        
        # 计算语法通过样本的平均迭代次数
        syntax_iterations = [s.get('syntax_pass_iteration', 0) for s in syntax_passed_samples]
        avg_syntax_iterations = np.mean(syntax_iterations) if syntax_iterations else 0
        
        success_rate = (completed_count / original_dataset_size) * 100 if original_dataset_size > 0 else 0
        syntax_pass_rate = (syntax_passed_count / original_dataset_size) * 100 if original_dataset_size > 0 else 0
        
        filtered_run_stats = {
            'dataset_name': dataset_name,
            'run_id': run_idx,
            'total_samples': original_dataset_size,
            'completed_samples': completed_count,
            'failed_samples': failed_count,
            'syntax_passed_samples': syntax_passed_count,
            'success_rate': success_rate,
            'syntax_pass_rate': syntax_pass_rate,
            'avg_iterations': avg_iterations,
            'avg_syntax_iterations': avg_syntax_iterations,  # 新增
            'max_iterations_limit': max_iterations,
            'saved_results_count': len(run_results),
            'saved_syntax_count': len(syntax_passed_samples)
        }
        filtered_runs_stats.append(filtered_run_stats)
        
        print(f"  Run {run_idx}: 原始样本={original_dataset_size}, 保存结果={len(run_results)}, 成功样本={completed_count}, 语法通过={syntax_passed_count}, 成功率={success_rate:.2f}%, 语法通过率={syntax_pass_rate:.2f}%, 平均语法迭代={avg_syntax_iterations:.2f}")
    
    # 提取各项指标
    success_rates = [stats['success_rate'] for stats in filtered_runs_stats]
    syntax_pass_rates = [stats['syntax_pass_rate'] for stats in filtered_runs_stats]
    avg_iterations = [stats['avg_iterations'] for stats in filtered_runs_stats]
    avg_syntax_iterations = [stats['avg_syntax_iterations'] for stats in filtered_runs_stats]
    
    # 计算无偏pass@k
    sample_success_by_index = {}
    sample_syntax_pass_by_index = {}
    
    # 初始化所有样本为失败
    for sample_idx in range(original_dataset_size):
        sample_success_by_index[sample_idx] = []
        sample_syntax_pass_by_index[sample_idx] = []
    
    for run_idx, (run_results, syntax_passed) in enumerate(zip(results_by_run, all_syntax_passed)):
        # 先将所有样本标记为失败
        for sample_idx in range(original_dataset_size):
            sample_success_by_index[sample_idx].append(False)
            sample_syntax_pass_by_index[sample_idx].append(False)
        
        # 然后更新实际有结果的样本 - 成功样本
        for result in run_results:
            sample_idx = result['index']
            if sample_idx < original_dataset_size:  # 确保索引在范围内
                # 成功条件：completed=True 且 iterations < max_iterations
                is_success = (result.get('completed', False) and 
                             result.get('iterations', 0) < max_iterations)
                sample_success_by_index[sample_idx][-1] = is_success
        
        # 更新语法通过的样本
        for syntax_sample in syntax_passed:
            sample_idx = syntax_sample['index']
            if sample_idx < original_dataset_size:  # 确保索引在范围内
                sample_syntax_pass_by_index[sample_idx][-1] = True
    
    def calculate_unbiased_pass_at_k(success_dict, k):
        """计算无偏pass@k"""
        if not success_dict:
            return 0.0
        
        total_samples = len(success_dict)
        passed_samples = 0
        
        for sample_idx, successes in success_dict.items():
            n = len(successes)
            c = sum(successes)
            
            if n < k:
                if c > 0:
                    passed_samples += 1
            else:
                if c > 0:
                    from math import comb
                    prob_fail = comb(n - c, k) / comb(n, k) if comb(n, k) > 0 else 0
                    prob_pass = 1 - prob_fail
                    passed_samples += prob_pass
        
        return (passed_samples / total_samples) * 100 if total_samples > 0 else 0.0
    
    # 计算pass@1, pass@8, pass@16
    pass_at_1 = calculate_unbiased_pass_at_k(sample_success_by_index, 1)
    pass_at_8 = calculate_unbiased_pass_at_k(sample_success_by_index, 8)
    pass_at_16 = calculate_unbiased_pass_at_k(sample_success_by_index, 16)
    
    # 计算语法通过的pass@k
    syntax_pass_at_1 = calculate_unbiased_pass_at_k(sample_syntax_pass_by_index, 1)
    syntax_pass_at_8 = calculate_unbiased_pass_at_k(sample_syntax_pass_by_index, 8)
    syntax_pass_at_16 = calculate_unbiased_pass_at_k(sample_syntax_pass_by_index, 16)
    
    statistics = {
        'num_runs': len(filtered_runs_stats),
        'max_iterations_limit': max_iterations,
        'original_dataset_size': original_dataset_size,
        'success_rate': {
            'mean': np.mean(success_rates),
            'std': np.std(success_rates),
            'values': success_rates
        },
        'syntax_pass_rate': {
            'mean': np.mean(syntax_pass_rates),
            'std': np.std(syntax_pass_rates),
            'values': syntax_pass_rates
        },
        'avg_iterations': {
            'mean': np.mean(avg_iterations),
            'std': np.std(avg_iterations),
            'values': avg_iterations
        },
        'avg_syntax_iterations': {  # 新增
            'mean': np.mean(avg_syntax_iterations),
            'std': np.std(avg_syntax_iterations),
            'values': avg_syntax_iterations
        },
        'unbiased_pass_at_k': {
            'pass_at_1': pass_at_1,
            'pass_at_8': pass_at_8,
            'pass_at_16': pass_at_16,
            'syntax_pass_at_1': syntax_pass_at_1,
            'syntax_pass_at_8': syntax_pass_at_8,
            'syntax_pass_at_16': syntax_pass_at_16
        }
    }
    
    return statistics, filtered_runs_stats, results_by_run

def process_dataset_from_all_results(dataset_dir, dataset_name, num_runs, max_iterations=4):
    """从all_results.jsonl处理单个数据集目录"""
    print(f"处理数据集目录: {dataset_dir}")
    
    all_results_path = os.path.join(dataset_dir, "all_results.jsonl")
    
    if not os.path.exists(all_results_path):
        print(f"  未找到 all_results.jsonl 文件")
        return None
    
    # 读取所有结果
    all_results = read_jsonl(all_results_path)
    print(f"  读取到 {len(all_results)} 个结果")
    
    if not all_results:
        print(f"  all_results.jsonl 为空")
        return None
    
    # 计算统计信息
    statistics, filtered_runs_stats, filtered_results_by_run = calculate_statistics_from_all_results(
        all_results, dataset_name, num_runs, max_iterations
    )
    
    return {
        'filtered_runs_stats': filtered_runs_stats,
        'filtered_results_by_run': filtered_results_by_run,
        'statistics': statistics,
        'all_results': all_results
    }

def main():
    parser = argparse.ArgumentParser(description="从all_results.jsonl重新统计指定最大迭代次数下的评估结果")
    parser.add_argument("--input_dir", type=str, required=True, help="原始输出目录路径")
    parser.add_argument("--max_iterations", type=int, default=4, help="最大迭代次数限制")
    parser.add_argument("--num_runs", type=int, default=16, help="运行次数（用于分组all_results）")
    
    args = parser.parse_args()
    
    print(f"开始从all_results.jsonl重新统计评估结果")
    print(f"输入目录: {args.input_dir}")
    print(f"最大迭代次数限制: {args.max_iterations}")
    print(f"运行次数: {args.num_runs}")
    print(f"统计规则:")
    print(f"  - 成功样本: completed=True 且 iterations<{args.max_iterations}")
    print(f"  - 语法通过样本: 第一次语法调用成功时，前面pass为false的工具数量<{args.max_iterations}")
    print(f"  - 总样本: 原始数据集大小")
    print(f"数据集大小: {DATASET_SIZES}")
    print("=" * 80)
    
    # 查找所有数据集目录
    dataset_dirs = []
    for item in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path) and item != "overall_summary.json":
            # 检查是否包含all_results.jsonl文件
            all_results_path = os.path.join(item_path, "all_results.jsonl")
            if os.path.exists(all_results_path):
                dataset_dirs.append((item, item_path))
    
    if not dataset_dirs:
        print("未找到任何包含all_results.jsonl的数据集目录")
        return
    
    print(f"找到 {len(dataset_dirs)} 个数据集目录")
    
    # 处理每个数据集
    all_dataset_results = {}
    
    for dataset_name, dataset_path in dataset_dirs:
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        
        result = process_dataset_from_all_results(dataset_path, dataset_name, args.num_runs, args.max_iterations)
        if result is None:
            continue
            
        all_dataset_results[dataset_name] = result
        
        # 输出统计信息
        stats = result['statistics']
        pass_at_k = stats['unbiased_pass_at_k']
        
        print(f"\n数据集 {dataset_name} 统计结果 (max_iterations={args.max_iterations}):")
        print(f"  原始数据集大小: {stats['original_dataset_size']}")
        print(f"  评估次数: {stats['num_runs']}")
        print(f"  成功率: {stats['success_rate']['mean']:.2f}% ± {stats['success_rate']['std']:.2f}%")
        print(f"  语法通过率: {stats['syntax_pass_rate']['mean']:.2f}% ± {stats['syntax_pass_rate']['std']:.2f}%")
        print(f"  平均迭代次数: {stats['avg_iterations']['mean']:.2f} ± {stats['avg_iterations']['std']:.2f}")
        print(f"  平均语法迭代次数: {stats['avg_syntax_iterations']['mean']:.2f} ± {stats['avg_syntax_iterations']['std']:.2f}")
        print(f"  无偏Pass@1: {pass_at_k['pass_at_1']:.2f}%")
        print(f"  无偏Pass@8: {pass_at_k['pass_at_8']:.2f}%")
        print(f"  无偏Pass@16: {pass_at_k['pass_at_16']:.2f}%")
        print(f"  语法Pass@1: {pass_at_k['syntax_pass_at_1']:.2f}%")
        print(f"  语法Pass@8: {pass_at_k['syntax_pass_at_8']:.2f}%")
        print(f"  语法Pass@16: {pass_at_k['syntax_pass_at_16']:.2f}%")
    
    # 输出总体统计信息
    if all_dataset_results:
        print(f"\n{'='*80}")
        print(f"总体结果汇总 (max_iterations={args.max_iterations})")
        print(f"{'='*80}")
        
        for dataset_name, results in all_dataset_results.items():
            stats = results['statistics']
            pass_at_k = stats['unbiased_pass_at_k']
            print(f"数据集: {dataset_name} (原始大小: {stats['original_dataset_size']})")
            print(f"  评估次数: {len(results['filtered_runs_stats'])}")
            print(f"  成功率: {stats['success_rate']['mean']:.2f}% ± {stats['success_rate']['std']:.2f}%")
            print(f"  语法通过率: {stats['syntax_pass_rate']['mean']:.2f}% ± {stats['syntax_pass_rate']['std']:.2f}%")
            print(f"  无偏Pass@1: {pass_at_k['pass_at_1']:.2f}%")
            print(f"  无偏Pass@8: {pass_at_k['pass_at_8']:.2f}%")
            print(f"  无偏Pass@16: {pass_at_k['pass_at_16']:.2f}%")
            print(f"  语法Pass@1: {pass_at_k['syntax_pass_at_1']:.2f}%")
            print(f"  语法Pass@8: {pass_at_k['syntax_pass_at_8']:.2f}%")
            print(f"  语法Pass@16: {pass_at_k['syntax_pass_at_16']:.2f}%")
            print(f"  平均迭代次数: {stats['avg_iterations']['mean']:.2f} ± {stats['avg_iterations']['std']:.2f}")
            print(f"  平均语法迭代次数: {stats['avg_syntax_iterations']['mean']:.2f} ± {stats['avg_syntax_iterations']['std']:.2f}")
            print("")
        
        print(f"{'='*80}")
        print("统计完成")

if __name__ == "__main__":
    main()