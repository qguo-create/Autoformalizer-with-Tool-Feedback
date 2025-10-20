import numpy as np
import json
import time
import os
from vllm import SamplingParams
from atf.pipeline import EventDrivenPipelineManager
from atf.utils import parse_tool_call, get_input_prompt, setup_logger, read_jsonl

def extract_syntax_passed_samples(all_results):
    """
    Extract samples that have passed syntax checks from all results.
    Return the code of each sample that passed the syntax check the last time.
    """
    syntax_passed_samples = []
    
    for result in all_results:
        # Find all syntax check results for this sample
        syntax_results = []
        lean4_codes = []
        
        # Iterate through all tool call results
        for i, tool_result in enumerate(result.get('tools', [])):
            # Check if it's a syntax check result
            if isinstance(tool_result, dict) and 'pass' in tool_result:
                # Find corresponding lean4_code in responses
                responses = result.get('responses', [])
                if i < len(responses):
                    response = responses[i]
                    # Parse tool_call to get lean4_code
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
        
        # Find the last passed syntax check code
        last_passed_syntax = None
        for syntax_result in reversed(syntax_results):  # Search from last
            if syntax_result['pass']:
                last_passed_syntax = syntax_result
                break
        
        # If found, add to results
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
                'formal_statement': result['data'].get('formal_statement', ''),  
            }
            syntax_passed_samples.append(syntax_passed_sample)
    
    return syntax_passed_samples

def calculate_statistics(all_runs_stats, all_results):
    """Calculate statistics for multiple runs, including unbiased pass@k"""
    if not all_runs_stats:
        return {}
    
    # Extract metrics
    success_rates = [stats['success_rate'] for stats in all_runs_stats]
    syntax_pass_rates = [stats['syntax_pass_rate'] for stats in all_runs_stats]
    avg_iterations = [stats['avg_iterations'] for stats in all_runs_stats]
    total_times = [stats['total_time'] for stats in all_runs_stats]
    avg_times_per_sample = [stats['avg_time_per_sample'] for stats in all_runs_stats]
    
    # Calculate unbiased pass@k
    # Group samples by index, collect success status across runs
    sample_success_by_index = {}
    sample_syntax_pass_by_index = {}
    
    for run_idx, results in enumerate(all_results):
        for result in results:
            sample_idx = result['index']
            if sample_idx not in sample_success_by_index:
                sample_success_by_index[sample_idx] = []
                sample_syntax_pass_by_index[sample_idx] = []
            
            sample_success_by_index[sample_idx].append(result.get('completed', False))
            # Check if any syntax check passed
            has_syntax_pass = any(
                tool_result.get('pass', False) 
                for tool_result in result.get('tools', [])
                if isinstance(tool_result, dict) and 'pass' in tool_result
            )
            sample_syntax_pass_by_index[sample_idx].append(has_syntax_pass)
    
    def calculate_unbiased_pass_at_k(success_dict, k):
        """Calculate unbiased pass@k"""
        if not success_dict:
            return 0.0
        
        total_samples = len(success_dict)
        passed_samples = 0
        
        for sample_idx, successes in success_dict.items():
            n = len(successes)  # Number of attempts
            c = sum(successes)  # Number of successes
            
            if n < k:
                # If attempts < k, use existing success rate
                if c > 0:
                    passed_samples += 1
            else:
                # Calculate unbiased pass@k: 1 - C(n-c, k) / C(n, k)
                if c > 0:
                    from math import comb
                    prob_fail = comb(n - c, k) / comb(n, k) if comb(n, k) > 0 else 0
                    prob_pass = 1 - prob_fail
                    passed_samples += prob_pass
        
        return (passed_samples / total_samples) * 100 if total_samples > 0 else 0.0
    
    # Calculate pass@1, pass@8, pass@16
    pass_at_1 = calculate_unbiased_pass_at_k(sample_success_by_index, 1)
    pass_at_8 = calculate_unbiased_pass_at_k(sample_success_by_index, 8)
    pass_at_16 = calculate_unbiased_pass_at_k(sample_success_by_index, 16)
    
    # Calculate syntax pass@k
    syntax_pass_at_1 = calculate_unbiased_pass_at_k(sample_syntax_pass_by_index, 1)
    syntax_pass_at_8 = calculate_unbiased_pass_at_k(sample_syntax_pass_by_index, 8)
    syntax_pass_at_16 = calculate_unbiased_pass_at_k(sample_syntax_pass_by_index, 16)
    
    statistics = {
        'num_runs': len(all_runs_stats),
        # Preserve original statistics (for compatibility)
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
    """Run single evaluation run"""
    
    # Create independent output path for each run
    run_output_dir = os.path.join(args.output_dir, dataset_name, f"run_{run_id}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    output_path = os.path.join(run_output_dir, "results.jsonl")
    stats_path = os.path.join(run_output_dir, "stats.json")
    syntax_passed_path = os.path.join(run_output_dir, "syntax_passed_samples.jsonl")
    
    # Check if already completed (resume functionality)
    if os.path.exists(stats_path) and os.path.exists(output_path):
        try:
            # Read existing statistics
            with open(stats_path, 'r', encoding='utf-8') as f:
                run_stats = json.load(f)
            
            # Read existing results
            results_to_save = []
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results_to_save.append(json.loads(line))
            
            # Read syntax passed samples
            syntax_passed_samples = []
            if os.path.exists(syntax_passed_path):
                with open(syntax_passed_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            syntax_passed_samples.append(json.loads(line))
            
            # Validate result completeness
            data = read_jsonl(dataset_path)
            if len(results_to_save) == len(data):
                logger.info(f"✓ Evaluation {run_id} already completed, skipping - Dataset: {dataset_name}")
                logger.info(f"  Success rate: {run_stats.get('success_rate', 0):.2f}%")
                logger.info(f"  Syntax pass rate: {run_stats.get('syntax_pass_rate', 0):.2f}%")
                return run_stats, results_to_save, syntax_passed_samples
            else:
                logger.warning(f"Evaluation {run_id} incomplete ({len(results_to_save)}/{len(data)}), restarting")
        except Exception as e:
            logger.warning(f"Failed to read existing results: {e}, restarting evaluation")
    
    # Original processing logic
    logger.info(f"Starting evaluation {run_id} - Dataset: {dataset_name}")
    logger.info(f"Output path: {output_path}")
    
    # Load data
    data = read_jsonl(dataset_path)
    logger.info(f"Successfully read data, total {len(data)} records")

    # Initialize samples
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

    # Create event-driven pipeline manager
    pipeline_manager = EventDrivenPipelineManager(
        models['atf1'], models['atf2'], models['qwq_model'], models['qwen3_model'], tokenizer,
        sampling_params, consistency_params, logger, 
        args.batch_size, args.max_iterations, output_path, args.max_prompt_length
    )

    # Run pipeline

    start_time = time.time()
    try:
        finished_samples = pipeline_manager.start_pipeline(samples)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping pipeline...")
        pipeline_manager.stop_pipeline()
        raise
    
    end_time = time.time()

    # Save all results (including failed samples)
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

    # Extract syntax passed samples
    syntax_passed_samples = extract_syntax_passed_samples(results_to_save)
    
    # Save syntax passed samples
    with open(syntax_passed_path, 'w', encoding='utf-8') as f:
        for sample in syntax_passed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Saved {len(syntax_passed_samples)} syntax passed samples to {syntax_passed_path}")

    # Statistics
    total_time = (end_time - start_time)
    completed_count = sum(1 for sample in finished_samples if sample.get('completed', False))
    failed_count = sum(1 for sample in finished_samples if sample.get('failed', False))
    length_failed_count = sum(1 for sample in finished_samples 
                             if 'token count' in sample.get('failure_reason', '').lower())
    avg_iterations = sum(sample['iteration_count'] for sample in finished_samples) / len(finished_samples) if finished_samples else 0
    success_rate = (completed_count / len(samples)) * 100 if samples else 0
    syntax_pass_rate = (len(syntax_passed_samples) / len(samples)) * 100 if samples else 0

    # Save run statistics
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
    
    # Save statistics to file
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Completed evaluation {run_id} - Dataset: {dataset_name}")
    logger.info(f"Success rate: {success_rate:.2f}%, Syntax pass rate: {syntax_pass_rate:.2f}%, Avg iterations: {avg_iterations:.2f}")
    
    return run_stats, results_to_save, syntax_passed_samples

def run_multi_dataset_evaluation(args, models, tokenizer):
    """Run multi-dataset multi-run evaluation"""
    
    # Setup logger
    logger, log_path = setup_logger(args.log_dir)
    logger.info("=" * 80)
    logger.info("Starting multi-dataset multi-run evaluation (with resume support)")
    logger.info(f"Dataset directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of runs: {args.num_runs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all dataset files
    dataset_files = []
    if os.path.isfile(args.input_dir):
        # Single file
        dataset_files = [(os.path.basename(args.input_dir).replace('.jsonl', ''), args.input_dir)]
    else:
        # All jsonl files in directory
        for filename in os.listdir(args.input_dir):
            if filename.endswith('.jsonl'):
                dataset_name = filename.replace('.jsonl', '')
                dataset_path = os.path.join(args.input_dir, filename)
                dataset_files.append((dataset_name, dataset_path))
    
    if not dataset_files:
        logger.error("No dataset files found")
        return
    
    logger.info(f"Found {len(dataset_files)} datasets: {[name for name, _ in dataset_files]}")
    
    # Store all results
    all_dataset_results = {}
    
    # Evaluate each dataset
    for dataset_name, dataset_path in dataset_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating dataset: {dataset_name}")
        logger.info(f"Dataset path: {dataset_path}")
        
        # Check completed runs
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
                    # Read existing statistics
                    with open(stats_path, 'r', encoding='utf-8') as f:
                        run_stats = json.load(f)
                    existing_runs_stats.append(run_stats)
                    
                    # Read existing results
                    run_results = []
                    with open(results_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                run_results.append(json.loads(line))
                    existing_results_by_run.append(run_results)
                    
                    # Read syntax passed samples
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
        
        # Skip if all runs completed
        if completed_runs == args.num_runs:
            logger.info(f"✓ Dataset {dataset_name} all {args.num_runs} runs completed, skipping")
            
            # Use existing results
            dataset_runs_stats = existing_runs_stats
            dataset_results_by_run = existing_results_by_run
            dataset_all_results = []
            for run_results in existing_results_by_run:
                dataset_all_results.extend(run_results)
            dataset_all_syntax_passed = existing_syntax_passed
            
        else:
            logger.info(f"Proceeding with remaining {args.num_runs - completed_runs} evaluations")
            logger.info(f"{'='*60}")
            
            dataset_runs_stats = existing_runs_stats.copy()
            dataset_results_by_run = existing_results_by_run.copy()
            dataset_all_results = []
            for run_results in existing_results_by_run:
                dataset_all_results.extend(run_results)
            dataset_all_syntax_passed = existing_syntax_passed.copy()

            # Run remaining evaluations
            for run_id in range(args.num_runs):
                # Skip completed runs
                run_output_dir = os.path.join(args.output_dir, dataset_name, f"run_{run_id}")
                stats_path = os.path.join(run_output_dir, "stats.json")
                results_path = os.path.join(run_output_dir, "results.jsonl")
                
                if os.path.exists(stats_path) and os.path.exists(results_path):
                    logger.info(f"Skipping completed run {run_id + 1}")
                    continue
                
                logger.info(f"\n{'-'*40}")
                logger.info(f"Dataset {dataset_name} - Run {run_id + 1}/{args.num_runs}")
                logger.info(f"{'-'*40}")
                
                # Set sampling parameters
                sampling_params = SamplingParams(
                    temperature=args.temperature, 
                    max_tokens=args.max_length, 
                    n=1, 
                    stop='</tool_calls>', 
                    include_stop_str_in_output=True,
                )

                consistency_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=16384, 
                    n=1, 
                )
                
                # Run single evaluation
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
                    logger.error(f"Run {run_id + 1} failed: {e}")
                    continue
        
        # Calculate dataset statistics
        if dataset_runs_stats:
            dataset_statistics = calculate_statistics(dataset_runs_stats, dataset_results_by_run)
            all_dataset_results[dataset_name] = {
                'runs_stats': dataset_runs_stats,
                'statistics': dataset_statistics,
                'all_results': dataset_all_results,
                'all_syntax_passed': dataset_all_syntax_passed
            }
            
            # Save dataset-level statistics
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
            
            # Save all results (including errors)
            all_results_path = os.path.join(args.output_dir, dataset_name, "all_results.jsonl")
            with open(all_results_path, 'w', encoding='utf-8') as f:
                for result in dataset_all_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # Save all syntax passed samples (merged runs)
            all_syntax_passed_path = os.path.join(args.output_dir, dataset_name, "all_syntax_passed_samples.jsonl")
            with open(all_syntax_passed_path, 'w', encoding='utf-8') as f:
                for sample in dataset_all_syntax_passed:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # Deduplicate syntax passed samples (keep last run's result)
            unique_syntax_passed = {}
            for sample in dataset_all_syntax_passed:
                unique_syntax_passed[sample['index']] = sample
            
            unique_syntax_passed_path = os.path.join(args.output_dir, dataset_name, "unique_syntax_passed_samples.jsonl")
            with open(unique_syntax_passed_path, 'w', encoding='utf-8') as f:
                for sample in sorted(unique_syntax_passed.values(), key=lambda x: x['index']):
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # Output dataset statistics
            logger.info(f"\n{'='*60}")
            logger.info(f"Dataset {dataset_name} evaluation completed")
            logger.info(f"Number of runs: {len(dataset_runs_stats)}")
            logger.info(f"Success rate: {dataset_statistics['success_rate']['mean']:.2f}% ± {dataset_statistics['success_rate']['std']:.2f}%")
            logger.info(f"Syntax pass rate: {dataset_statistics['syntax_pass_rate']['mean']:.2f}% ± {dataset_statistics['syntax_pass_rate']['std']:.2f}%")
            logger.info(f"Average iterations: {dataset_statistics['avg_iterations']['mean']:.2f} ± {dataset_statistics['avg_iterations']['std']:.2f}")

            # pass@k output
            pass_at_k = dataset_statistics['unbiased_pass_at_k']
            logger.info(f"Unbiased Pass@1: {pass_at_k['pass_at_1']:.2f}%")
            logger.info(f"Unbiased Pass@8: {pass_at_k['pass_at_8']:.2f}%")
            logger.info(f"Unbiased Pass@16: {pass_at_k['pass_at_16']:.2f}%")
            logger.info(f"Syntax Pass@1: {pass_at_k['syntax_pass_at_1']:.2f}%")
            logger.info(f"Syntax Pass@8: {pass_at_k['syntax_pass_at_8']:.2f}%")
            logger.info(f"Syntax Pass@16: {pass_at_k['syntax_pass_at_16']:.2f}%")

            logger.info(f"Average processing time: {dataset_statistics['total_time']['mean']:.2f}s ± {dataset_statistics['total_time']['std']:.2f}s")
            logger.info(f"Average time per sample: {dataset_statistics['avg_time_per_sample']['mean']:.2f}s ± {dataset_statistics['avg_time_per_sample']['std']:.2f}s")

    # Generate overall report
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
        
        # Output overall statistics
        logger.info(f"\n{'='*80}")
        logger.info("Multi-dataset Evaluation Summary")
        logger.info(f"{'='*80}")
        
        for dataset_name, results in all_dataset_results.items():
            stats = results['statistics']
            pass_at_k = stats['unbiased_pass_at_k']
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"  Number of runs: {len(results['runs_stats'])}")
            logger.info(f"  Success rate: {stats['success_rate']['mean']:.2f}% ± {stats['success_rate']['std']:.2f}%")
            logger.info(f"  Syntax pass rate: {stats['syntax_pass_rate']['mean']:.2f}% ± {stats['syntax_pass_rate']['std']:.2f}%")
            logger.info(f"  Unbiased Pass@1: {pass_at_k['pass_at_1']:.2f}%")
            logger.info(f"  Unbiased Pass@8: {pass_at_k['pass_at_8']:.2f}%")
            logger.info(f"  Unbiased Pass@16: {pass_at_k['pass_at_16']:.2f}%")
            logger.info(f"  Syntax Pass@1: {pass_at_k['syntax_pass_at_1']:.2f}%")
            logger.info(f"  Syntax Pass@8: {pass_at_k['syntax_pass_at_8']:.2f}%")
            logger.info(f"  Syntax Pass@16: {pass_at_k['syntax_pass_at_16']:.2f}%")
            logger.info(f"  Average iterations: {stats['avg_iterations']['mean']:.2f} ± {stats['avg_iterations']['std']:.2f}")
            logger.info(f"  Average processing time: {stats['total_time']['mean']:.2f}s ± {stats['total_time']['std']:.2f}s")
            logger.info("")

        logger.info(f"Overall report saved to: {overall_summary_path}")
        logger.info(f"{'='*80}")