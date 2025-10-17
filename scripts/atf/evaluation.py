import numpy as np
import json
import time
import os
from vllm import SamplingParams
from atf.pipeline import EventDrivenPipelineManager
from atf.utils import parse_tool_call, get_input_prompt, setup_logger, read_jsonl

def extract_syntax_passed_samples(all_results):
    """
    从所有结果中提取语法检查通过的样本
    返回每个样本最后一次语法检查通过的代码
    """
    syntax_passed_samples = []
    
    for result in all_results:
        # 查找该样本所有的语法检查结果
        syntax_results = []
        lean4_codes = []
        
        # 遍历所有工具调用结果
        for i, tool_result in enumerate(result.get('tools', [])):
            # 检查是否是语法检查结果
            if isinstance(tool_result, dict) and 'pass' in tool_result:
                # 找到对应的response中的lean4_code
                responses = result.get('responses', [])
                if i < len(responses):
                    response = responses[i]
                    # 解析response中的tool_call来获取lean4_code
                    tool_call = parse_tool_call(response)
                    if tool_call.get('name') == 'syntax_check':
                        lean4_code = tool_call.get('arguments', {}).get('lean4_code', '')
                        if lean4_code:
                            syntax_results.append({
                                'pass': tool_result.get('pass', False),
                                'lean4_code': lean4_code,
                                'error_message': tool_result.get('error', ''),
                                'iteration': i
                            })
        
        # 找到最后一次语法检查通过的代码
        last_passed_syntax = None
        for syntax_result in reversed(syntax_results):  # 从后往前找
            if syntax_result['pass']:
                last_passed_syntax = syntax_result
                break
        
        # 如果找到了语法通过的代码，添加到结果中
        if last_passed_syntax:
            syntax_passed_sample = {
                'index': result['index'],
                'data': result['data'],
                'lean4_code': last_passed_syntax['lean4_code'],
                'syntax_pass_iteration': last_passed_syntax['iteration'],
                'final_completed': result.get('completed', False),
                'final_failed': result.get('failed', False),
                'final_failure_reason': result.get('failure_reason', ''),
                'total_iterations': result.get('iterations', 0),
                'informal_statement': result['data'].get('informal_statement', ''),
                'formal_statement': result['data'].get('formal_statement', ''),  # 如果有的话
            }
            syntax_passed_samples.append(syntax_passed_sample)
    
    return syntax_passed_samples

def calculate_statistics(all_runs_stats, all_results):
    """计算多次运行的统计信息，包括无偏pass@k"""
    if not all_runs_stats:
        return {}
    
    # 提取各项指标（保持原有统计）
    success_rates = [stats['success_rate'] for stats in all_runs_stats]
    syntax_pass_rates = [stats['syntax_pass_rate'] for stats in all_runs_stats]
    avg_iterations = [stats['avg_iterations'] for stats in all_runs_stats]
    total_times = [stats['total_time'] for stats in all_runs_stats]
    avg_times_per_sample = [stats['avg_time_per_sample'] for stats in all_runs_stats]
    
    # 计算无偏pass@k
    # 按样本index分组，收集每个样本在不同run中的成功情况
    sample_success_by_index = {}
    sample_syntax_pass_by_index = {}
    
    for run_idx, results in enumerate(all_results):
        for result in results:
            sample_idx = result['index']
            if sample_idx not in sample_success_by_index:
                sample_success_by_index[sample_idx] = []
                sample_syntax_pass_by_index[sample_idx] = []
            
            sample_success_by_index[sample_idx].append(result.get('completed', False))
            # 检查是否有语法通过的代码
            has_syntax_pass = any(
                tool_result.get('pass', False) 
                for tool_result in result.get('tools', [])
                if isinstance(tool_result, dict) and 'pass' in tool_result
            )
            sample_syntax_pass_by_index[sample_idx].append(has_syntax_pass)
    
    def calculate_unbiased_pass_at_k(success_dict, k):
        """计算无偏pass@k"""
        if not success_dict:
            return 0.0
        
        total_samples = len(success_dict)
        passed_samples = 0
        
        for sample_idx, successes in success_dict.items():
            n = len(successes)  # 该样本的尝试次数
            c = sum(successes)  # 该样本的成功次数
            
            if n < k:
                # 如果尝试次数少于k，按现有成功率计算
                if c > 0:
                    passed_samples += 1
            else:
                # 计算无偏pass@k: 1 - C(n-c, k) / C(n, k)
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
        'num_runs': len(all_runs_stats),
        # 保持原有统计（用于兼容性）
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
        'total_time': {
            'mean': np.mean(total_times),
            'std': np.std(total_times),
            'values': total_times
        },
        'avg_time_per_sample': {
            'mean': np.mean(avg_times_per_sample),
            'std': np.std(avg_times_per_sample),
            'values': avg_times_per_sample
        },
        # 新增无偏pass@k指标
        'unbiased_pass_at_k': {
            'pass_at_1': pass_at_1,
            'pass_at_8': pass_at_8,
            'pass_at_16': pass_at_16,
            'syntax_pass_at_1': syntax_pass_at_1,
            'syntax_pass_at_8': syntax_pass_at_8,
            'syntax_pass_at_16': syntax_pass_at_16
        }
    }
    
    return statistics

def run_single_evaluation(dataset_path, dataset_name, run_id, args, models, tokenizer, sampling_params, consistency_params, logger):
    """运行单次评估"""
    
    # 为每次运行创建独立的输出路径
    run_output_dir = os.path.join(args.output_dir, dataset_name, f"run_{run_id}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    output_path = os.path.join(run_output_dir, "results.jsonl")
    stats_path = os.path.join(run_output_dir, "stats.json")
    syntax_passed_path = os.path.join(run_output_dir, "syntax_passed_samples.jsonl")
    
    # 检查是否已经完成（断点续跑功能）
    if os.path.exists(stats_path) and os.path.exists(output_path):
        try:
            # 读取已有的统计信息
            with open(stats_path, 'r', encoding='utf-8') as f:
                run_stats = json.load(f)
            
            # 读取已有的结果
            results_to_save = []
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results_to_save.append(json.loads(line))
            
            # 读取语法通过的样本
            syntax_passed_samples = []
            if os.path.exists(syntax_passed_path):
                with open(syntax_passed_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            syntax_passed_samples.append(json.loads(line))
            
            # 验证结果完整性
            data = read_jsonl(dataset_path)
            if len(results_to_save) == len(data):
                logger.info(f"✓ 第 {run_id} 次评估已完成，跳过 - 数据集: {dataset_name}")
                logger.info(f"  成功率: {run_stats.get('success_rate', 0):.2f}%")
                logger.info(f"  语法通过率: {run_stats.get('syntax_pass_rate', 0):.2f}%")
                return run_stats, results_to_save, syntax_passed_samples
            else:
                logger.warning(f"第 {run_id} 次评估结果不完整 ({len(results_to_save)}/{len(data)})，重新开始")
        except Exception as e:
            logger.warning(f"读取已有结果失败: {e}，重新开始评估")
    
    # 原有的处理逻辑
    logger.info(f"开始第 {run_id} 次评估 - 数据集: {dataset_name}")
    logger.info(f"输出路径: {output_path}")
    
    # 加载数据
    data = read_jsonl(dataset_path)
    logger.info(f"成功读取数据，总计 {len(data)} 条记录")

    # 初始化样本
    samples = []
    for i, d in enumerate(data):
        samples.append({
            'index': i,
            'data': d,
            'prompts': [get_input_prompt(tokenizer, d['informal_statement'])],
            'responses': [],
            'tools': [],
            'completed': False,
            'failed': False,
            'failure_reason': '',
            'iteration_count': 0,
            'chat_count': 0
        })

    # 创建事件驱动流水线管理器
    pipeline_manager = EventDrivenPipelineManager(
        models['atf1'], models['atf2'], models['qwq_model'], models['qwen3_model'], tokenizer,
        sampling_params, consistency_params, logger, 
        args.batch_size, args.max_iterations, output_path, args.max_prompt_length
    )

    # 运行流水线
    start_time = time.time()
    try:
        finished_samples = pipeline_manager.start_pipeline(samples)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止流水线...")
        pipeline_manager.stop_pipeline()
        raise
    
    end_time = time.time()

    # 保存所有结果（包括失败的样本）
    results_to_save = []
    for sample in finished_samples:
        results_to_save.append({
            'index': sample['index'],
            'data': sample['data'],
            'prompts': sample['prompts'],
            'responses': sample.get('responses', []),
            'tools': sample['tools'],
            'completed': sample.get('completed', False),
            'failed': sample.get('failed', False),
            'failure_reason': sample.get('failure_reason', ''),
            'iterations': sample['iteration_count']
        })

    # 提取语法通过的样本
    syntax_passed_samples = extract_syntax_passed_samples(results_to_save)
    
    # 保存语法通过的样本
    with open(syntax_passed_path, 'w', encoding='utf-8') as f:
        for sample in syntax_passed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"✓ 保存 {len(syntax_passed_samples)} 个语法通过样本到 {syntax_passed_path}")

    # 统计信息
    total_time = (end_time - start_time)
    completed_count = sum(1 for sample in finished_samples if sample.get('completed', False))
    failed_count = sum(1 for sample in finished_samples if sample.get('failed', False))
    length_failed_count = sum(1 for sample in finished_samples 
                             if 'token count' in sample.get('failure_reason', '').lower())
    avg_iterations = sum(sample['iteration_count'] for sample in finished_samples) / len(finished_samples) if finished_samples else 0
    success_rate = (completed_count / len(samples)) * 100 if samples else 0
    syntax_pass_rate = (len(syntax_passed_samples) / len(samples)) * 100 if samples else 0

    # 保存运行统计信息
    run_stats = {
        'dataset_name': dataset_name,
        'run_id': run_id,
        'total_samples': len(samples),
        'completed_samples': completed_count,
        'failed_samples': failed_count,
        'length_failed_samples': length_failed_count,
        'syntax_passed_samples': len(syntax_passed_samples),
        'success_rate': success_rate,
        'syntax_pass_rate': syntax_pass_rate,
        'avg_iterations': avg_iterations,
        'total_time': total_time,
        'avg_time_per_sample': total_time / len(samples) if samples else 0
    }
    
    # 保存统计信息到文件
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"第 {run_id} 次评估完成 - 数据集: {dataset_name}")
    logger.info(f"成功率: {success_rate:.2f}%, 语法通过率: {syntax_pass_rate:.2f}%, 平均迭代次数: {avg_iterations:.2f}")
    
    return run_stats, results_to_save, syntax_passed_samples

def run_multi_dataset_evaluation(args, models, tokenizer):
    """运行多数据集多次评估"""
    
    # 设置日志记录器
    logger, log_path = setup_logger(args.log_dir)
    logger.info("=" * 80)
    logger.info("开始多数据集多次评估任务（支持断点续跑）")
    logger.info(f"数据集目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"评估次数: {args.num_runs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"最大迭代次数: {args.max_iterations}")
    logger.info("=" * 80)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有数据集文件
    dataset_files = []
    if os.path.isfile(args.input_dir):
        # 单个文件
        dataset_files = [(os.path.basename(args.input_dir).replace('.jsonl', ''), args.input_dir)]
    else:
        # 目录中的所有jsonl文件
        for filename in os.listdir(args.input_dir):
            if filename.endswith('.jsonl'):
                dataset_name = filename.replace('.jsonl', '')
                dataset_path = os.path.join(args.input_dir, filename)
                dataset_files.append((dataset_name, dataset_path))
    
    if not dataset_files:
        logger.error("未找到任何数据集文件")
        return
    
    logger.info(f"找到 {len(dataset_files)} 个数据集: {[name for name, _ in dataset_files]}")
    
    # 存储所有结果
    all_dataset_results = {}
    
    # 对每个数据集进行评估
    for dataset_name, dataset_path in dataset_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"开始评估数据集: {dataset_name}")
        logger.info(f"数据集路径: {dataset_path}")
        
        # 检查已完成的评估次数
        completed_runs = 0
        existing_runs_stats = []
        existing_results_by_run = []
        existing_syntax_passed = []
        
        for run_id in range(args.num_runs):
            run_output_dir = os.path.join(args.output_dir, dataset_name, f"run_{run_id}")
            stats_path = os.path.join(run_output_dir, "stats.json")
            results_path = os.path.join(run_output_dir, "results.jsonl")
            
            if os.path.exists(stats_path) and os.path.exists(results_path):
                try:
                    # 读取已有统计信息
                    with open(stats_path, 'r', encoding='utf-8') as f:
                        run_stats = json.load(f)
                    existing_runs_stats.append(run_stats)
                    
                    # 读取已有结果
                    run_results = []
                    with open(results_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                run_results.append(json.loads(line))
                    existing_results_by_run.append(run_results)
                    
                    # 读取语法通过的样本
                    syntax_passed_path = os.path.join(run_output_dir, "syntax_passed_samples.jsonl")
                    if os.path.exists(syntax_passed_path):
                        with open(syntax_passed_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.strip():
                                    existing_syntax_passed.append(json.loads(line))
                    
                    completed_runs += 1
                except Exception as e:
                    logger.warning(f"读取第 {run_id} 次评估结果失败: {e}")
        
        if completed_runs > 0:
            logger.info(f"发现已完成的评估: {completed_runs}/{args.num_runs}")
        
        # 如果所有评估都已完成，跳过
        if completed_runs == args.num_runs:
            logger.info(f"✓ 数据集 {dataset_name} 所有 {args.num_runs} 次评估已完成，跳过")
            
            # 使用已有结果
            dataset_runs_stats = existing_runs_stats
            dataset_results_by_run = existing_results_by_run
            dataset_all_results = []
            for run_results in existing_results_by_run:
                dataset_all_results.extend(run_results)
            dataset_all_syntax_passed = existing_syntax_passed
            
        else:
            logger.info(f"继续进行剩余 {args.num_runs - completed_runs} 次评估")
            logger.info(f"{'='*60}")
            
            dataset_runs_stats = existing_runs_stats.copy()
            dataset_results_by_run = existing_results_by_run.copy()
            dataset_all_results = []
            for run_results in existing_results_by_run:
                dataset_all_results.extend(run_results)
            dataset_all_syntax_passed = existing_syntax_passed.copy()

            # 进行剩余的评估
            for run_id in range(args.num_runs):
                # 跳过已完成的评估
                run_output_dir = os.path.join(args.output_dir, dataset_name, f"run_{run_id}")
                stats_path = os.path.join(run_output_dir, "stats.json")
                results_path = os.path.join(run_output_dir, "results.jsonl")
                
                if os.path.exists(stats_path) and os.path.exists(results_path):
                    logger.info(f"跳过已完成的第 {run_id + 1} 次评估")
                    continue
                
                logger.info(f"\n{'-'*40}")
                logger.info(f"数据集 {dataset_name} - 第 {run_id + 1}/{args.num_runs} 次评估")
                logger.info(f"{'-'*40}")
                
                # 设置采样参数
                sampling_params = SamplingParams(
                    temperature=args.temperature, 
                    max_tokens=args.max_length, 
                    n=1, 
                    stop='</tool_calls>', 
                    include_stop_str_in_output=True,
                )

                consistency_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=16000, 
                    n=1, 
                )
                
                # 运行单次评估
                try:
                    run_stats, run_results, syntax_passed_samples = run_single_evaluation(
                        dataset_path, dataset_name, run_id, args, 
                        models, tokenizer, sampling_params, consistency_params, logger
                    )
                    
                    dataset_runs_stats.append(run_stats)
                    dataset_all_results.extend(run_results)
                    dataset_all_syntax_passed.extend(syntax_passed_samples)
                    dataset_results_by_run.append(run_results)
                    
                except Exception as e:
                    logger.error(f"第 {run_id + 1} 次评估失败: {e}")
                    continue
        
        # 计算该数据集的统计信息
        if dataset_runs_stats:
            dataset_statistics = calculate_statistics(dataset_runs_stats, dataset_results_by_run)
            all_dataset_results[dataset_name] = {
                'runs_stats': dataset_runs_stats,
                'statistics': dataset_statistics,
                'all_results': dataset_all_results,
                'all_syntax_passed': dataset_all_syntax_passed
            }
            
            # 保存数据集级别的统计信息
            dataset_summary_path = os.path.join(args.output_dir, dataset_name, "summary.json")
            os.makedirs(os.path.dirname(dataset_summary_path), exist_ok=True)
            
            summary_data = {
                'dataset_name': dataset_name,
                'num_runs': len(dataset_runs_stats),
                'statistics': dataset_statistics,
                'individual_runs': dataset_runs_stats
            }
            
            with open(dataset_summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            # 保存所有结果（包括错误样本）
            all_results_path = os.path.join(args.output_dir, dataset_name, "all_results.jsonl")
            with open(all_results_path, 'w', encoding='utf-8') as f:
                for result in dataset_all_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # 保存所有语法通过的样本（合并所有运行）
            all_syntax_passed_path = os.path.join(args.output_dir, dataset_name, "all_syntax_passed_samples.jsonl")
            with open(all_syntax_passed_path, 'w', encoding='utf-8') as f:
                for sample in dataset_all_syntax_passed:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # 去重保存语法通过的样本（按index去重，保留最后一次运行的结果）
            unique_syntax_passed = {}
            for sample in dataset_all_syntax_passed:
                unique_syntax_passed[sample['index']] = sample
            
            unique_syntax_passed_path = os.path.join(args.output_dir, dataset_name, "unique_syntax_passed_samples.jsonl")
            with open(unique_syntax_passed_path, 'w', encoding='utf-8') as f:
                for sample in sorted(unique_syntax_passed.values(), key=lambda x: x['index']):
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # 输出数据集统计信息
            logger.info(f"\n{'='*60}")
            logger.info(f"数据集 {dataset_name} 评估完成")
            logger.info(f"评估次数: {len(dataset_runs_stats)}")
            logger.info(f"成功率: {dataset_statistics['success_rate']['mean']:.2f}% ± {dataset_statistics['success_rate']['std']:.2f}%")
            logger.info(f"语法通过率: {dataset_statistics['syntax_pass_rate']['mean']:.2f}% ± {dataset_statistics['syntax_pass_rate']['std']:.2f}%")
            logger.info(f"平均迭代次数: {dataset_statistics['avg_iterations']['mean']:.2f} ± {dataset_statistics['avg_iterations']['std']:.2f}")

            # pass@k指标输出
            pass_at_k = dataset_statistics['unbiased_pass_at_k']
            logger.info(f"无偏Pass@1: {pass_at_k['pass_at_1']:.2f}%")
            logger.info(f"无偏Pass@8: {pass_at_k['pass_at_8']:.2f}%")
            logger.info(f"无偏Pass@16: {pass_at_k['pass_at_16']:.2f}%")
            logger.info(f"语法Pass@1: {pass_at_k['syntax_pass_at_1']:.2f}%")
            logger.info(f"语法Pass@8: {pass_at_k['syntax_pass_at_8']:.2f}%")
            logger.info(f"语法Pass@16: {pass_at_k['syntax_pass_at_16']:.2f}%")

            logger.info(f"平均处理时间: {dataset_statistics['total_time']['mean']:.2f}s ± {dataset_statistics['total_time']['std']:.2f}s")
            logger.info(f"平均每样本时间: {dataset_statistics['avg_time_per_sample']['mean']:.2f}s ± {dataset_statistics['avg_time_per_sample']['std']:.2f}s")

    # 生成总体报告
    if all_dataset_results:
        overall_summary_path = os.path.join(args.output_dir, "overall_summary.json")
        
        overall_summary = {
            'evaluation_config': {
                'num_runs': args.num_runs,
                'max_iterations': args.max_iterations,
                'batch_size': args.batch_size,
                'temperature': args.temperature,
                'max_length': args.max_length,
                'max_prompt_length': args.max_prompt_length,
                'model': args.model
            },
            'datasets': {}
        }
        
        for dataset_name, results in all_dataset_results.items():
            overall_summary['datasets'][dataset_name] = {
                'num_runs': len(results['runs_stats']),
                'statistics': results['statistics']
            }
        
        with open(overall_summary_path, 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, ensure_ascii=False, indent=2)
        
        # 输出总体统计信息
        logger.info(f"\n{'='*80}")
        logger.info("多数据集评估总体结果")
        logger.info(f"{'='*80}")
        
        for dataset_name, results in all_dataset_results.items():
            stats = results['statistics']
            pass_at_k = stats['unbiased_pass_at_k']
            logger.info(f"数据集: {dataset_name}")
            logger.info(f"  评估次数: {len(results['runs_stats'])}")
            logger.info(f"  成功率: {stats['success_rate']['mean']:.2f}% ± {stats['success_rate']['std']:.2f}%")
            logger.info(f"  语法通过率: {stats['syntax_pass_rate']['mean']:.2f}% ± {stats['syntax_pass_rate']['std']:.2f}%")
            logger.info(f"  无偏Pass@1: {pass_at_k['pass_at_1']:.2f}%")
            logger.info(f"  无偏Pass@8: {pass_at_k['pass_at_8']:.2f}%")
            logger.info(f"  无偏Pass@16: {pass_at_k['pass_at_16']:.2f}%")
            logger.info(f"  语法Pass@1: {pass_at_k['syntax_pass_at_1']:.2f}%")
            logger.info(f"  语法Pass@8: {pass_at_k['syntax_pass_at_8']:.2f}%")
            logger.info(f"  语法Pass@16: {pass_at_k['syntax_pass_at_16']:.2f}%")
            logger.info(f"  平均迭代次数: {stats['avg_iterations']['mean']:.2f} ± {stats['avg_iterations']['std']:.2f}")
            logger.info(f"  平均处理时间: {stats['total_time']['mean']:.2f}s ± {stats['total_time']['std']:.2f}s")
            logger.info("")

        logger.info(f"总体报告保存至: {overall_summary_path}")
        logger.info(f"{'='*80}")