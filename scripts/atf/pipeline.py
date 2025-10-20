import os
import json
import time
import threading
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set
from atf.utils import get_consistency_prompt, parse_tool_call, format_tool_results_as_user_message, parse_consistency_response, batch_syntax_check, batch_generate

SEEDS=[12,34,56,78,90]

@dataclass
class SampleState:
    """Sample state enumeration"""
    NEED_INFERENCE = "need_inference"
    NEED_SYNTAX_CHECK = "need_syntax_check" 
    NEED_QWQ_CHECK = "need_qwq_check"
    NEED_QWEN3_CHECK = "need_qwen3_check"
    COMPLETED = "completed"
    FAILED = "failed"

# 添加一个简单的增量保存器
class IncrementalSaver:
    """Incremental saver for saving completed samples"""
    
    def __init__(self, output_path: str, logger):
        self.output_path = output_path
        self.logger = logger
        self.save_lock = threading.Lock()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    def save_completed_samples(self, samples: List[dict]):
        """Save completed samples"""
        if not samples:
            return
            
        with self.save_lock:
            try:
                # Prepare data to save
                results_to_save = []
                for sample in samples:
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
                

                with open(self.output_path, 'a', encoding='utf-8') as f:
                    for result in results_to_save:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                self.logger.info(f"✓ Incrementally saved {len(samples)} completed samples to {self.output_path}")
                
            except Exception as e:
                self.logger.error(f"✗ Incremental save failed: {e}")

class UnifiedDataPool:
    """Unified data pool supporting multi-state sample management"""
    
    def __init__(self, tokenizer, max_batch_size=256, max_prompt_length=40000):
        self.max_batch_size = max_batch_size
        self.max_prompt_length = max_prompt_length
        self.lock = threading.RLock()
        
        # State-grouped sample queues
        self.samples_by_state = defaultdict(deque)
        
        self.tokenizer = tokenizer

        # Sample index mapping (fast lookup)
        self.sample_index_map = {}
        
        # Processing sample tracking
        self.processing_samples = {}
        
        # Completed and failed samples
        self.completed_samples = []
        self.failed_samples = []
        
        # Statistics
        self.stats = {
            'total_samples': 0,
            'completed_samples': 0,
            'failed_samples': 0,
            'processing_counts': defaultdict(int),
            'length_failed_samples': 0
        }
        
        # Batch update lock to prevent workers from taking samples during batch updates
        self.batch_update_lock = threading.Lock()
    
    def _check_prompt_length(self, prompt: str) -> bool:
        """Check if prompt length exceeds limit (using tokenizer)"""
        try:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            token_count = len(tokens)
            return token_count <= self.max_prompt_length
        except Exception as e:
            print(f"Warning: Failed to tokenize prompt for length check: {e}")
            return False

    def add_samples(self, samples: List[dict]):
        """Add new samples to data pool"""
        with self.lock:
            for sample in samples:
                sample['state'] = SampleState.NEED_INFERENCE
                sample['processing'] = False
                
                # Check initial prompt length
                initial_prompt = sample['prompts'][-1] if sample['prompts'] else ""
                if not self._check_prompt_length(initial_prompt):
                    actual_tokens = len(self.tokenizer.encode(initial_prompt, add_special_tokens=True)) if self.tokenizer else len(initial_prompt)
                    sample['failed'] = True
                    sample['failure_reason'] = f"Initial prompt token count ({actual_tokens}) exceeds maximum ({self.max_prompt_length})"
                    self.failed_samples.append(sample)
                    self.stats['failed_samples'] += 1
                    self.stats['length_failed_samples'] += 1
                else:
                    self.samples_by_state[SampleState.NEED_INFERENCE].append(sample)
                    self.sample_index_map[sample['index']] = sample
                
                self.stats['total_samples'] += 1

    def get_samples_for_processing(self, state: str, max_count: int = None) -> List[dict]:
        """Get samples in specified state for processing - LIFO priority"""
        if max_count is None:
            max_count = self.max_batch_size
        
        with self.batch_update_lock:
            pass
            
        with self.lock:
            available_samples = []
            queue = self.samples_by_state[state]
            
            temp_samples = []
            
            while queue and len(available_samples) < max_count:
                sample = queue.pop()
                temp_samples.append(sample)
                
                if not sample.get('processing', False):
                    sample['processing'] = True
                    available_samples.append(sample)
                    self.processing_samples[sample['index']] = sample
                    self.stats['processing_counts'][state] += 1
            
            for sample in reversed(temp_samples):
                queue.append(sample)
            
            return available_samples
    
    def batch_update_sample_states(self, updates: List[tuple]):
        """Batch update sample states with length check"""
        with self.batch_update_lock:
            with self.lock:
                completed_samples = []
                failed_samples = []
                length_failed_count = 0
                
                for sample, new_state, remove_processing in updates:
                    old_state = sample.get('state')
                    
                    if remove_processing and sample.get('processing'):
                        sample['processing'] = False
                        self.processing_samples.pop(sample['index'], None)
                        
                        if old_state:
                            self.stats['processing_counts'][old_state] -= 1
                        
                        if old_state and old_state in self.samples_by_state:
                            queue = self.samples_by_state[old_state]
                            for i, s in enumerate(queue):
                                if s['index'] == sample['index']:
                                    del queue[i]
                                    break
                    
                    if new_state == SampleState.NEED_INFERENCE:
                        current_prompt = sample['prompts'][-1] if sample['prompts'] else ""
                        if not self._check_prompt_length(current_prompt):
                            actual_tokens = len(self.tokenizer.encode(current_prompt, add_special_tokens=True)) if self.tokenizer else len(current_prompt)
                            new_state = SampleState.FAILED
                            sample['failed'] = True
                            sample['failure_reason'] = f"Prompt token count ({actual_tokens}) exceeds maximum ({self.max_prompt_length})"
                            length_failed_count += 1
                    
                    if new_state == SampleState.COMPLETED:
                        if sample['index'] in self.sample_index_map:
                            del self.sample_index_map[sample['index']]
                        completed_samples.append(sample)
                        self.stats['completed_samples'] += 1
                    elif new_state == SampleState.FAILED:
                        if sample['index'] in self.sample_index_map:
                            del self.sample_index_map[sample['index']]
                        failed_samples.append(sample)
                        self.stats['failed_samples'] += 1
                    else:
                        sample['state'] = new_state
                        self.samples_by_state[new_state].append(sample)
                
                self.completed_samples.extend(completed_samples)
                self.failed_samples.extend(failed_samples)
                self.stats['length_failed_samples'] += length_failed_count
    
    def update_sample_state(self, sample: dict, new_state: str, remove_processing=True):
        """Single sample state update"""
        self.batch_update_sample_states([(sample, new_state, remove_processing)])
    
    def has_processing_samples(self) -> bool:
        """Check if any samples are being processed"""
        with self.lock:
            return len(self.processing_samples) > 0
    
    def get_processing_sample_count(self) -> int:
        """Get count of processing samples"""
        with self.lock:
            return len(self.processing_samples)
    
    def get_pool_status(self) -> Dict:
        """Get data pool status"""
        with self.lock:
            finished_samples = self.stats['completed_samples'] + self.stats['failed_samples']
            
            samples_by_state_total = {}
            samples_by_state_available = {}
            
            for state, queue in self.samples_by_state.items():
                total_count = len(queue)
                available_count = sum(1 for sample in queue if not sample.get('processing', False))
                
                samples_by_state_total[state] = total_count
                samples_by_state_available[state] = available_count
            
            status = {
                'total_samples': self.stats['total_samples'],
                'completed_samples': self.stats['completed_samples'],
                'failed_samples': self.stats['failed_samples'],
                'length_failed_samples': self.stats['length_failed_samples'],
                'finished_samples': finished_samples,
                'remaining_samples': self.stats['total_samples'] - finished_samples,
                'samples_by_state_total': samples_by_state_total,
                'samples_by_state_available': samples_by_state_available,
                'processing_counts': dict(self.stats['processing_counts']),
                'processing_samples_count': len(self.processing_samples)
            }
            return status
    
    def has_work(self) -> bool:
        """Check if there's remaining work"""
        with self.lock:
            has_queued_samples = any(len(queue) > 0 for queue in self.samples_by_state.values())
            has_processing_samples = len(self.processing_samples) > 0
            finished_samples = self.stats['completed_samples'] + self.stats['failed_samples']
            remaining_samples = self.stats['total_samples'] - finished_samples
            
            return has_queued_samples or has_processing_samples or remaining_samples > 0
    
    def get_all_finished_samples(self) -> List[dict]:
        """Get all finished samples (including success/failure)"""
        with self.lock:
            return self.completed_samples.copy() + self.failed_samples.copy()

# 保持原有的Worker类不变...
class AsyncStageWorker:
    """Asynchronous stage worker base class"""
    
    def __init__(self, stage_name: str, data_pool: UnifiedDataPool, logger, check_interval: float = 0.1):
        self.stage_name = stage_name
        self.data_pool = data_pool
        self.logger = logger
        self.check_interval = check_interval
        self.running = False
        self.worker_thread = None
        
    def start(self):
        """Start worker thread"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._work_loop, name=f"{self.stage_name}_worker")
        self.worker_thread.start()
        self.logger.info(f"[{self.stage_name}] Worker thread started")
    
    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        self.logger.info(f"[{self.stage_name}] Worker thread stopped")
    
    def _work_loop(self):
        """Work loop"""
        while self.running:
            try:
                samples = self.get_samples_to_process()
                
                if samples:
                    self.logger.info(f"[{self.stage_name}] Processing {len(samples)} samples")
                    start_time = time.time()
                    
                    self.process_samples(samples)
                    
                    end_time = time.time()
                    self.logger.info(f"[{self.stage_name}] Processing completed, time: {end_time-start_time:.2f}s")
                else:
                    time.sleep(self.check_interval)
                    
            except Exception as e:
                self.logger.error(f"[{self.stage_name}] Processing error: {e}")
                time.sleep(1)
    
    def get_samples_to_process(self) -> List[dict]:
        """Get samples to process (to be implemented by subclass)"""
        raise NotImplementedError
    
    def process_samples(self, samples: List[dict]):
        """Process samples (to be implemented by subclass)"""
        raise NotImplementedError

class LLMInferenceWorker(AsyncStageWorker):
    """LLM inference worker"""
    
    def __init__(self, llm, sampling_params, data_pool, logger, max_iterations=8, batch_size=256, name='LLM_Inference 1'):
        super().__init__("LLM_Inference", data_pool, logger, check_interval=0.05)
        self.llm = llm
        self.sampling_params = sampling_params
        self.max_iterations = max_iterations
        self.batch_size = batch_size
    
    def get_samples_to_process(self) -> List[dict]:
        return self.data_pool.get_samples_for_processing(SampleState.NEED_INFERENCE, max_count=self.batch_size)
    
    def process_samples(self, samples: List[dict]):
        if not samples:
            return
        
        prompts = [sample['prompts'][-1] for sample in samples]
        responses = batch_generate(self.llm, prompts, self.sampling_params)
        
        updates = []
        
        for sample, response in zip(samples, responses):
            tool_call = parse_tool_call(response)
            sample['chat_count'] += 1
            sample['last_response'] = response
            sample['last_tool_call'] = tool_call
            
            sample['responses'].append(response)
            
            if sample['chat_count'] > 2 * self.max_iterations:
                sample['failed'] = True
                sample['failure_reason'] = f"Exceeded maximum chat count ({2 * self.max_iterations})"
                next_state = SampleState.FAILED
            else:
                tool_name = tool_call.get('name', '')
                if tool_name == 'syntax_check':
                    next_state = SampleState.NEED_SYNTAX_CHECK
                elif tool_name == 'consistency_check':
                    next_state = SampleState.NEED_QWQ_CHECK
                else:
                    next_state = self._handle_invalid_tool_call(sample)
            
            updates.append((sample, next_state, True))
        
        self.data_pool.batch_update_sample_states(updates)

    def _handle_invalid_tool_call(self, sample) -> str:
        """Handle invalid tool calls"""
        sample['iteration_count'] += 1
        
        if sample['iteration_count'] >= self.max_iterations:
            sample['failed'] = True
            sample['failure_reason'] = f"Exceeded maximum iterations ({self.max_iterations}) with invalid tool calls"
            return SampleState.FAILED
        
        tool_feedback = format_tool_results_as_user_message('', {"error": "No valid tool call"})
        next_prompt = sample['prompts'][-1] + sample['last_response'] + '\n\n' + tool_feedback + '\n'
        sample['prompts'].append(next_prompt)
        sample['tools'].append({"error": "No valid tool call"})
        
        sample.pop('last_response', None)
        sample.pop('last_tool_call', None)
        
        return SampleState.NEED_INFERENCE

class SyntaxCheckWorker(AsyncStageWorker):
    """Syntax check worker"""
    
    def __init__(self, data_pool, logger, max_iterations=8, batch_size=256):
        super().__init__("Syntax_Check", data_pool, logger, check_interval=0.1)
        self.max_iterations = max_iterations
        self.batch_size = batch_size
    
    def get_samples_to_process(self) -> List[dict]:
        return self.data_pool.get_samples_for_processing(SampleState.NEED_SYNTAX_CHECK, max_count=self.batch_size)
    
    def process_samples(self, samples: List[dict]):
        if not samples:
            return
            
        lean4_codes = []
        for sample in samples:
            tool_call = sample.get('last_tool_call', {})
            code = tool_call.get('arguments', {}).get('lean4_code', '')
            lean4_codes.append(code)
        
        syntax_results = batch_syntax_check(lean4_codes)
        
        updates = []
        
        for sample, result in zip(samples, syntax_results):
            sample['tools'].append(result)
            next_state = self._determine_next_state(sample, result)
            updates.append((sample, next_state, True))
        
        self.data_pool.batch_update_sample_states(updates)
    
    def _determine_next_state(self, sample, result) -> str:
        """Determine next state based on syntax check result"""
        
        if not result.get('pass'):
            sample['iteration_count'] += 1

        if sample['iteration_count'] >= self.max_iterations:
            sample['failed'] = True
            sample['failure_reason'] = f"Exceeded maximum iterations ({self.max_iterations}) with syntax errors"
            return SampleState.FAILED

        response = sample.get('last_response', '')
        tool_call = sample.get('last_tool_call', {})
        tool_feedback = format_tool_results_as_user_message(tool_call.get('name', ''), result)
        next_prompt = sample['prompts'][-1] + response + '\n\n' + tool_feedback + '\n'
        sample['prompts'].append(next_prompt)
        
        sample.pop('last_response', None)
        sample.pop('last_tool_call', None)
        
        return SampleState.NEED_INFERENCE

class QWQConsistencyWorker(AsyncStageWorker):
    """QWQ consistency check worker"""
    
    def __init__(self, qwq_model, tokenizer, consistency_params, data_pool, logger, max_iterations=8, batch_size=256):
        super().__init__("QWQ_Consistency", data_pool, logger, check_interval=0.1)
        self.qwq_model = qwq_model
        self.tokenizer = tokenizer
        self.consistency_params = consistency_params
        self.max_iterations = max_iterations
        self.batch_size = batch_size
    
    def get_samples_to_process(self) -> List[dict]:
        return self.data_pool.get_samples_for_processing(SampleState.NEED_QWQ_CHECK, max_count=self.batch_size)
    
    def process_samples(self, samples: List[dict]):
        if not samples:
            return
            
        prompts = []
        for sample in samples:
            tool_call = sample.get('last_tool_call', {})
            informal_statement = sample['data']['informal_statement']
            lean4_code = tool_call.get('arguments', {}).get('lean4_code', '')
            consistency_prompt = get_consistency_prompt(self.tokenizer, informal_statement, lean4_code)
            prompts.append(consistency_prompt)
        
        qwq_responses = batch_generate(self.qwq_model, prompts, self.consistency_params)
        
        updates = []
        
        for sample, response in zip(samples, qwq_responses):
            qwq_result = parse_consistency_response(response)
            sample['qwq_result'] = qwq_result
            
            if qwq_result.get('pass'):
                next_state = SampleState.NEED_QWEN3_CHECK
            else:
                sample['tools'].append(qwq_result)
                next_state = self._determine_next_state(sample, qwq_result)
            
            updates.append((sample, next_state, True))
        
        self.data_pool.batch_update_sample_states(updates)
    
    def _determine_next_state(self, sample, result) -> str:
        """Determine next state based on QWQ result"""
        sample['iteration_count'] += 1
        
        if sample['iteration_count'] >= self.max_iterations:
            sample['failed'] = True
            sample['failure_reason'] = f"Exceeded maximum iterations ({self.max_iterations}) with QWQ consistency check failures"
            return SampleState.FAILED
        
        response = sample.get('last_response', '')
        tool_call = sample.get('last_tool_call', {})
        tool_feedback = format_tool_results_as_user_message(tool_call.get('name', ''), result)
        next_prompt = sample['prompts'][-1] + response + '\n\n' + tool_feedback + '\n'
        sample['prompts'].append(next_prompt)
        
        sample.pop('last_response', None)
        sample.pop('last_tool_call', None)
        sample.pop('qwq_result', None)
        
        return SampleState.NEED_INFERENCE

class QWen3ConsistencyWorker(AsyncStageWorker):
    """QWen3 consistency check worker"""
    
    def __init__(self, qwen3_model, tokenizer, consistency_params, data_pool, logger, max_iterations=8, batch_size=256, saver=None):
        super().__init__("QWen3_Consistency", data_pool, logger, check_interval=0.1)
        self.qwen3_model = qwen3_model
        self.tokenizer = tokenizer
        self.consistency_params = consistency_params
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.saver = saver
    
    def get_samples_to_process(self) -> List[dict]:
        return self.data_pool.get_samples_for_processing(SampleState.NEED_QWEN3_CHECK, max_count=self.batch_size)
    
    def process_samples(self, samples: List[dict]):
        if not samples:
            return
            
        prompts = []
        for sample in samples:
            tool_call = sample.get('last_tool_call', {})
            informal_statement = sample['data']['informal_statement']
            lean4_code = tool_call.get('arguments', {}).get('lean4_code', '')
            consistency_prompt = get_consistency_prompt(self.tokenizer, informal_statement, lean4_code)
            prompts.append(consistency_prompt)
        
        qwen3_responses = batch_generate(self.qwen3_model, prompts, self.consistency_params)
        
        updates = []
        completed_samples = []
        
        for sample, response in zip(samples, qwen3_responses):
            qwen3_result = parse_consistency_response(response)
            sample['tools'].append(qwen3_result)
            
            next_state = self._determine_next_state(sample, qwen3_result)
            
            if next_state == SampleState.COMPLETED:
                completed_samples.append(sample)
            
            updates.append((sample, next_state, True))
        
        self.data_pool.batch_update_sample_states(updates)
        
        if completed_samples and self.saver:
            self.saver.save_completed_samples(completed_samples)
    
    def _determine_next_state(self, sample, result) -> str:
        """Determine next state based on QWen3 result"""
        
        if result.get('pass'):
            sample['completed'] = True
            return SampleState.COMPLETED
        else:
            sample['iteration_count'] += 1
            
            if sample['iteration_count'] >= self.max_iterations:
                sample['failed'] = True
                sample['failure_reason'] = f"Exceeded maximum iterations ({self.max_iterations}) with QWen3 consistency check failures"
                return SampleState.FAILED
            
            response = sample.get('last_response', '')
            tool_call = sample.get('last_tool_call', {})
            tool_feedback = format_tool_results_as_user_message(tool_call.get('name', ''), result)
            next_prompt = sample['prompts'][-1] + response + '\n\n' + tool_feedback + '\n'
            sample['prompts'].append(next_prompt)
            
            sample.pop('last_response', None)
            sample.pop('last_tool_call', None)
            sample.pop('qwq_result', None)
            
            return SampleState.NEED_INFERENCE

class EventDrivenPipelineManager:
    """Event-driven pipeline manager"""

    def __init__(self, llm1, llm2, qwq_model, qwen3_model, tokenizer, 
                 sampling_params, consistency_params, logger, 
                 max_batch_size=256, max_iterations=8, output_path=None, max_prompt_length=40000):
        
        self.data_pool = UnifiedDataPool(tokenizer, max_batch_size, max_prompt_length)
        self.logger = logger
        self.max_iterations = max_iterations
        
        self.saver = IncrementalSaver(output_path, logger) if output_path else None
        
        self.workers = [
            LLMInferenceWorker(llm1, sampling_params, self.data_pool, logger, max_iterations, max_batch_size, name='LLM_Inference 1'),
            LLMInferenceWorker(llm2, sampling_params, self.data_pool, logger, max_iterations, max_batch_size, name='LLM_Inference 2'),
            SyntaxCheckWorker(self.data_pool, logger, max_iterations, max_batch_size),
            QWQConsistencyWorker(qwq_model, tokenizer, consistency_params, self.data_pool, logger, max_iterations, max_batch_size),
            QWen3ConsistencyWorker(qwen3_model, tokenizer, consistency_params, self.data_pool, logger, max_iterations, max_batch_size, self.saver)
        ]
        
        self.monitor_thread = None
        self.running = False

    def start_pipeline(self, samples):
        """Start pipeline processing"""
        
        self.logger.info(f"Starting event-driven pipeline, sample count: {len(samples)}")
        self.data_pool.add_samples(samples)
        
        for worker in self.workers:
            worker.start()
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_progress, 
                                             name="progress_monitor")
        self.monitor_thread.start()
        
        self._wait_for_completion()
        
        self.stop_pipeline()
        
        return self.data_pool.get_all_finished_samples()
    
    def _monitor_progress(self):
        """Monitor processing progress"""
        while self.running:
            status = self.data_pool.get_pool_status()
            self.logger.info(f"Progress monitor - Total samples: {status['total_samples']}, "
                        f"Completed: {status['completed_samples']}, "
                        f"Failed: {status['failed_samples']}, "
                        f"Length failed: {status['length_failed_samples']}, "
                        f"Remaining: {status['remaining_samples']}")
            self.logger.info(f"Total samples by state: {status['samples_by_state_total']}")
            self.logger.info(f"Available samples by state: {status['samples_by_state_available']}")
            self.logger.info(f"Processing counts: {status['processing_counts']}")
            
            time.sleep(60)
    
    def _wait_for_completion(self):
        """Wait for all samples to complete processing"""
        while self.data_pool.has_work():
            time.sleep(5)
        
        time.sleep(10)
    
    def stop_pipeline(self):
        """Stop pipeline"""
        self.running = False
        
        for worker in self.workers:
            worker.stop()
        
        if self.monitor_thread:
            self.monitor_thread.join()
        
        self.logger.info("Event-driven pipeline stopped")
