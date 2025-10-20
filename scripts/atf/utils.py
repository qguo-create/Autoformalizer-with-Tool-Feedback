import re
import json
import os
import logging
from vllm import LLM
import time
from tqdm import tqdm
from typing import Optional, List, Union
from verify_api import batch_verify_lean_codes
from template import consistency_template, ATF_system_prompt

def extract_json_blocks(text: str, return_first_only: bool = False) -> Union[Optional[str], List[str]]:
    """
    Extract JSON content between ```json and ``` blocks
    
    Args:
        text: Text containing JSON code blocks
        return_first_only: If True, returns first match; otherwise returns all matches
        
    Returns:
        If return_first_only=True: First matched JSON string or None
        If return_first_only=False: List of all matched JSON strings
    """
    # Regular expression to match content between ```json and ```
    pattern = r"```json\s*(.*?)\s*```"

    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        return None if return_first_only else []
    
    return matches[0] if return_first_only else matches

def parse_tool_call(text: str) -> dict:
    """Simplified tool call parser"""
    match = re.search(r'<tool_calls>(.*?)</tool_calls>', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return {}
    return {}

def format_tool_results_as_user_message(tool_name, tool_result):
    """Format tool results into human-readable message"""
    message = "<tool_results>\n"
    message += f"Function: {tool_name}\n"
    message += f"Output: {json.dumps(tool_result, indent=2, ensure_ascii=False)}\n"
    return message+"</tool_results>\n"

def parse_consistency_response(response):
    try:
        answer_json = json.loads(extract_json_blocks(response.split('</think>')[-1])[-1])
        return {
            "pass": not 'incorrect' in answer_json['is_assistant_correct'].lower(),
            "explanations": answer_json["reasons"]
        }
    except:
        return {
            "pass": False,
            "explanations": 'invalid response from judge'
        }

def get_input_prompt(tokenizer, informal_statement):
    msg = [{'role':'system', 'content':ATF_system_prompt},
           {'role':'user', 'content':'Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n' + informal_statement}]
    return tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True
    )

def get_consistency_prompt(tokenizer, informal_statement, formal_statement):
    msg = [{'role':'user', 'content':consistency_template.replace('{informal_statement}', informal_statement).replace('{formal_statement}', formal_statement)}]
    return tokenizer.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True
    )

def setup_logger(log_dir):
    """Configure logging system"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S",time.localtime(time.time()))
    log_filename = f"inference_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger('inference_pipeline')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.propagate = False
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path

def batch_syntax_check(lean4_code_list):
    syntax_results = []
    lean4_code_list = [c.replace('```lean4\n','').replace('\n```','') for c in lean4_code_list]
    temp_results = batch_verify_lean_codes(lean4_code_list)
    for code, temp_result in zip(lean4_code_list, temp_results):

        try:
            syntax_results.append({"pass": temp_result['pass'], "errors": temp_result['info']['errors']})
        except:
            max_count = 1
            retry_result = {'pass':False, 'errors':['Unsuccessful Lean4 Excution']}
            while max_count > 0:
                try:
                    retry_result = batch_verify_lean_codes([code.replace('```lean4\n','').replace('\n```','')])[0]
                    retry_result = {"pass": retry_result['pass'], "errors": retry_result['info']['errors']}
                    break
                except:
                    max_count -= 1
            syntax_results.append(retry_result)
    return syntax_results

def load_model_on_gpus(model_path, gpu_ids, model_name, seed):
    """Load model on specified GPUs"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    print(f"Loading {model_name} on GPUs {gpu_ids} with seed {seed}...")
    model = LLM(
        model=model_path,
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=0.9,
        dtype="bfloat16", 
        swap_space=16,
        disable_custom_all_reduce=True,
        seed=seed 
    )
    return model

def read_jsonl(path):
    res = []
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            res.append(json.loads(line))
    return res

def write_jsonl(data_to_write, path, mode):
    with open(path, mode) as f:
        for x in tqdm(data_to_write):
            line = json.dumps(x, ensure_ascii=False)
            f.write(line + '\n')

def batch_generate(llm, prompts, sampling_params, batch_size=4096):
    """Batch generate responses with configurable batch size"""
    all_outputs = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_outputs = llm.generate(batch, sampling_params, use_tqdm=False)
        all_outputs.extend([output.outputs[0].text for output in batch_outputs])
    
    return all_outputs
