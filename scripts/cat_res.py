import os
import json
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List

DATASET_SIZES = {
    'formalmath-lite': 418,
    'proverbench': 230,
    'combibench': 100
}

def read_jsonl(file_path):
    """Read JSONL file"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def has_syntax_pass_in_iterations(result, max_iterations=4):
    """
    Check if sample passed syntax check within specified iterations
    New criteria: Number of failed tools before first successful syntax check < max_iterations
    """
    tools = result.get('tools', [])
    if not tools:
        return False
    
    false_count = 0  # Count of tools with pass=False
    
    for i, tool_result in enumerate(tools):
        if isinstance(tool_result, dict) and 'pass' in tool_result:
            if tool_result.get('pass', False) and 'errors' in tool_result:
                # Found first successful syntax check
                return false_count < max_iterations
            else:
                false_count += 1
    return False

def get_syntax_pass_iteration(result):
    """
    Get iteration count when syntax check passed (number of failed tools before success)
    """
    tools = result.get('tools', [])
    if not tools:
        return None
    
    false_count = 0
    
    for i, tool_result in enumerate(tools):
        if isinstance(tool_result, dict) and 'pass' in tool_result:
            if tool_result.get('pass', False):
                return false_count
            else:
                false_count += 1
    
    return None  # No successful syntax check found

def extract_syntax_passed_from_results(results, max_iterations=4):
    """Extract samples that passed syntax check from results"""
    syntax_passed_samples = []
    
    for result in results:
        if has_syntax_pass_in_iterations(result, max_iterations):
            syntax_pass_iteration = get_syntax_pass_iteration(result)
            
            syntax_passed_sample = {
                'index': result['index'],
                'data': result['data'],
                'syntax_pass_iteration': syntax_pass_iteration,
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
    """Group results by run index"""
    if not all_results:
        return []
    

    total_samples = len(all_results)
    samples_per_run = total_samples // num_runs
    
    results_by_run = []
    for run_idx in range(num_runs):
        start_idx = run_idx * samples_per_run
        if run_idx == num_runs - 1: 
            end_idx = total_samples
        else:
            end_idx = (run_idx + 1) * samples_per_run
        
        run_results = all_results[start_idx:end_idx]
        results_by_run.append(run_results)
    
    return results_by_run

def is_valid_tool(tools):
    """
    Validate tool sequence
    
    Args:
        tools: List of tool invocation results
        
    Returns:
        bool: Whether the tool sequence is valid
    """
    if not tools or len(tools) < 2:
        return False
    
    # Condition 1: Last tool must be consistency check with pass=True
    last_tool = tools[-1]
    second_last_tool = tools[-2]
    

    if not ('explanations' in last_tool and last_tool.get('pass', False)):
        return False
    

    if not ('errors' in second_last_tool and second_last_tool.get('pass', False)):
        return False
    
    # Conditions 2: Validate tool sequence logic
    for i in range(len(tools)):
        current_tool = tools[i]
        
        # Condition: After failed tool, next must be syntax check
        if not current_tool.get('pass', False):
            if i + 1 < len(tools):
                next_tool = tools[i + 1]
                if not 'errors' in next_tool:
                    return False
            else:
                return False
        
        # Condition: After successful syntax check, next must be consistency check
        if 'errors' in current_tool and current_tool.get('pass', False) and i + 1 < len(tools):
            next_tool = tools[i + 1]
            if not 'explanations' in next_tool:
                return False
    
    return True

def calculate_statistics_from_all_results(all_results, dataset_name, num_runs, max_iterations=4):
    """Calculate statistics from all results"""
    if not all_results:
        return {}
    
    # 获取数据集的原始大小
    original_dataset_size = DATASET_SIZES.get(dataset_name, len(all_results) // num_runs if num_runs > 0 else len(all_results))
    print(f"  Using original dataset size: {original_dataset_size}, Total results: {len(all_results)}, Inferred runs: {num_runs}")
    

    results_by_run = group_results_by_run(all_results, num_runs)
    
    filtered_runs_stats = []
    all_syntax_passed = []
    
    for run_idx, run_results in enumerate(results_by_run):

        completed_count = sum(1 for r in run_results 
                            if r.get('completed', False) and 
                            r.get('iterations', 0) < max_iterations and is_valid_tool(r.get('tools',{})))
        

        syntax_passed_samples = extract_syntax_passed_from_results(run_results, max_iterations)
        all_syntax_passed.append(syntax_passed_samples)
        syntax_passed_count = len(syntax_passed_samples)
        
        failed_count = original_dataset_size - completed_count
        
        valid_iterations = [r.get('iterations', 0) for r in run_results 
                          if r.get('iterations', 0) < max_iterations and r.get('completed', False)]
        avg_iterations = np.mean(valid_iterations) if valid_iterations else 0
        
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
            'avg_syntax_iterations': avg_syntax_iterations, 
            'max_iterations_limit': max_iterations,
            'saved_results_count': len(run_results),
            'saved_syntax_count': len(syntax_passed_samples)
        }
        filtered_runs_stats.append(filtered_run_stats)
        
        print(f"  Run {run_idx}: 原始样本={original_dataset_size}, 保存结果={len(run_results)}, 成功样本={completed_count}, 语法通过={syntax_passed_count}, 成功率={success_rate:.2f}%, 语法通过率={syntax_pass_rate:.2f}%, 平均语法迭代={avg_syntax_iterations:.2f}")
    
    # Extract metrics
    success_rates = [stats['success_rate'] for stats in filtered_runs_stats]
    syntax_pass_rates = [stats['syntax_pass_rate'] for stats in filtered_runs_stats]
    avg_iterations = [stats['avg_iterations'] for stats in filtered_runs_stats]
    avg_syntax_iterations = [stats['avg_syntax_iterations'] for stats in filtered_runs_stats]
    

    sample_success_by_index = {}
    sample_syntax_pass_by_index = {}
    

    for sample_idx in range(original_dataset_size):
        sample_success_by_index[sample_idx] = []
        sample_syntax_pass_by_index[sample_idx] = []
    
    for run_idx, (run_results, syntax_passed) in enumerate(zip(results_by_run, all_syntax_passed)):

        for sample_idx in range(original_dataset_size):
            sample_success_by_index[sample_idx].append(False)
            sample_syntax_pass_by_index[sample_idx].append(False)
        

        for result in run_results:
            sample_idx = result['index']
            if sample_idx < original_dataset_size:  

                is_success = (result.get('completed', False) and 
                             result.get('iterations', 0) < max_iterations)
                sample_success_by_index[sample_idx][-1] = is_success
        

        for syntax_sample in syntax_passed:
            sample_idx = syntax_sample['index']
            if sample_idx < original_dataset_size: 
                sample_syntax_pass_by_index[sample_idx][-1] = True
    
    def calculate_unbiased_pass_at_k(success_dict, k):
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
    
    # Calculate pass@k metrics
    pass_at_1 = calculate_unbiased_pass_at_k(sample_success_by_index, 1)
    pass_at_8 = calculate_unbiased_pass_at_k(sample_success_by_index, 8)
    pass_at_16 = calculate_unbiased_pass_at_k(sample_success_by_index, 16)
    
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
        'avg_syntax_iterations': { 
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
    """Process dataset directory containing all_results.jsonl"""
    print(f"Processing dataset directory: {dataset_dir}")
    
    all_results_path = os.path.join(dataset_dir, "all_results.jsonl")
    
    if not os.path.exists(all_results_path):
        print(f"  all_results.jsonl not found")
        return None
    
    # 读取所有结果
    all_results = read_jsonl(all_results_path)
    print(f"  Read {len(all_results)} results")
    
    if not all_results:
        print(f"  all_results.jsonl is empty")
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
    parser = argparse.ArgumentParser(description="Re-analyze evaluation results from all_results.jsonl with specified max iterations")
    parser.add_argument("--input_dir", type=str, required=True, help="Original output directory path")
    parser.add_argument("--max_iterations", type=int, default=4, help="Maximum revision iteration limit")
    parser.add_argument("--num_runs", type=int, default=16, help="Number of runs (for grouping results)")
    
    args = parser.parse_args()
    
    print(f"Re-analyzing evaluation results from all_results.jsonl")
    print(f"Input directory: {args.input_dir}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Analysis criteria:")
    print(f"  - Successful samples: completed=True AND iterations<{args.max_iterations}")
    print(f"  - Syntax passed samples: First successful syntax check with failed tools <{args.max_iterations}")
    print(f"  - Total samples: Original dataset size")
    print(f"Dataset sizes: {DATASET_SIZES}")
    print("=" * 80)
    
    # Find dataset directories
    dataset_dirs = []
    for item in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path) and item != "overall_summary.json":
            all_results_path = os.path.join(item_path, "all_results.jsonl")
            if os.path.exists(all_results_path):
                dataset_dirs.append((item, item_path))
    
    if not dataset_dirs:
        print("No dataset directories with all_results.jsonl found")
        return
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    
    # Process datasets
    all_dataset_results = {}
    
    for dataset_name, dataset_path in dataset_dirs:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        
        result = process_dataset_from_all_results(dataset_path, dataset_name, args.num_runs, args.max_iterations)
        if result is None:
            continue
            
        all_dataset_results[dataset_name] = result
        
        stats = result['statistics']
        pass_at_k = stats['unbiased_pass_at_k']
        
        print(f"\nDataset {dataset_name} results (max_iterations={args.max_iterations}):")
        print(f"  Original size: {stats['original_dataset_size']}")
        print(f"  Evaluation runs: {stats['num_runs']}")
        print(f"  Success rate: {stats['success_rate']['mean']:.2f}% ± {stats['success_rate']['std']:.2f}%")
        print(f"  Syntax pass rate: {stats['syntax_pass_rate']['mean']:.2f}% ± {stats['syntax_pass_rate']['std']:.2f}%")
        print(f"  Avg iterations: {stats['avg_iterations']['mean']:.2f} ± {stats['avg_iterations']['std']:.2f}")
        print(f"  Avg syntax iterations: {stats['avg_syntax_iterations']['mean']:.2f} ± {stats['avg_syntax_iterations']['std']:.2f}")
        print(f"  Unbiased Pass@1: {pass_at_k['pass_at_1']:.2f}%")
        print(f"  Unbiased Pass@8: {pass_at_k['pass_at_8']:.2f}%")
        print(f"  Unbiased Pass@16: {pass_at_k['pass_at_16']:.2f}%")
        print(f"  Syntax Pass@1: {pass_at_k['syntax_pass_at_1']:.2f}%")
        print(f"  Syntax Pass@8: {pass_at_k['syntax_pass_at_8']:.2f}%")
        print(f"  Syntax Pass@16: {pass_at_k['syntax_pass_at_16']:.2f}%")
    
    # Print overall summary
    if all_dataset_results:
        print(f"\n{'='*80}")
        print(f"Overall Summary (max_iterations={args.max_iterations})")
        print(f"{'='*80}")
        
        for dataset_name, results in all_dataset_results.items():
            stats = results['statistics']
            pass_at_k = stats['unbiased_pass_at_k']
            print(f"Dataset: {dataset_name} (Original size: {stats['original_dataset_size']})")
            print(f"  Runs: {len(results['filtered_runs_stats'])}")
            print(f"  Success rate: {stats['success_rate']['mean']:.2f}% ± {stats['success_rate']['std']:.2f}%")
            print(f"  Syntax pass rate: {stats['syntax_pass_rate']['mean']:.2f}% ± {stats['syntax_pass_rate']['std']:.2f}%")
            print(f"  Unbiased Pass@1: {pass_at_k['pass_at_1']:.2f}%")
            print(f"  Unbiased Pass@8: {pass_at_k['pass_at_8']:.2f}%")
            print(f"  Unbiased Pass@16: {pass_at_k['pass_at_16']:.2f}%")
            print(f"  Syntax Pass@1: {pass_at_k['syntax_pass_at_1']:.2f}%")
            print(f"  Syntax Pass@8: {pass_at_k['syntax_pass_at_8']:.2f}%")
            print(f"  Syntax Pass@16: {pass_at_k['syntax_pass_at_16']:.2f}%")
            print(f"  Avg iterations: {stats['avg_iterations']['mean']:.2f} ± {stats['avg_iterations']['std']:.2f}")
            print(f"  Avg syntax iterations: {stats['avg_syntax_iterations']['mean']:.2f} ± {stats['avg_syntax_iterations']['std']:.2f}")
            print("")
        
        print(f"{'='*80}")
        print("Analysis completed")

if __name__ == "__main__":
    main()