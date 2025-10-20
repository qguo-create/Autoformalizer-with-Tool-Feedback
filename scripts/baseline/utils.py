
import json
from tqdm import tqdm
import re
import time
from typing import Dict, List, Any, Tuple,Set

def read_jsonl(path):
    res = []
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            res.append(json.loads(line))
        return res

def write_jsonl(data_to_write, path, mode):
    with open(path, mode, encoding="utf-8") as f:
        for x in tqdm(data_to_write):
            line = json.dumps(x, ensure_ascii=False)
            f.write(line + '\n')

def write_res(path, batch_res):
    with open(path,"a+", encoding='utf-8') as f:
        for r in batch_res:
            try:
                f.write(json.dumps(r, ensure_ascii=False)+"\n")
            except:
                try:
                    f.write(json.dumps(r, ensure_ascii=True)+"\n")
                except:
                    raise ValueError
                
def check_brackets_balance(text: str) -> bool:
    """check brackets balance"""
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}
    temp = text.replace('/-','(').replace('-/',')')
    for char in temp:
        if char in pairs:
            stack.append(char)
        elif char in pairs.values():
            if not stack or pairs[stack.pop()] != char:
                return False
    return len(stack) == 0

def extract_imports(code: str) -> Set[str]:
    """Extract import statements from code"""
    import_pattern = r'^\s*import\s+([A-Za-z0-9_\.]+)'
    imports = set()
    
    for line in code.split('\n'):
        match = re.match(import_pattern, line)
        if match:
            if not 'markers' in match.group(0).strip():
                imports.add(match.group(0).strip())
            
    return imports

def group_by_imports(statements) -> Dict[frozenset, List[str]]:
    """Group statements by their import dependencies"""
    groups = {}
    for statement in statements:
        s = statement['statement']
        imports = extract_imports(s)
        key = frozenset(imports)
        if key not in groups:
            groups[key] = []
        groups[key].append(statement)
    return groups

def combine_lean4_statements_simple(statements: List[str]) -> str:
    """
    Combine multiple Lean 4 theorem statements into a single file
    
    Args:
        statements: List of Lean 4 statements
        
    Returns:
        Combined Lean 4 code as a string
    """
    # Extract all import statements
    all_imports = set()
    for stmt in statements:
        all_imports.update(extract_imports(stmt))
    
    # Remove import statements from individual statements
    cleaned_statements = []
    for stmt in statements:
        cleaned_stmt = stmt
        for import_stmt in all_imports:
            cleaned_stmt = re.sub(re.escape(import_stmt) + r'\s*\n', '', cleaned_stmt)
        cleaned_statements.append(cleaned_stmt.strip())
    
    # Build combined code
    current_date = time.strftime("%Y-%m-%d")
    
    # Header and imports
    result = [
        f"-- Lean 4 语法验证文件",
        f"-- 自动生成于 {current_date}",
        "",
        "-- 共享导入区域"
    ]
    
    # Add all imports
    if all_imports:
        result.extend(sorted(all_imports))
    else:
        result.append("-- 无导入语句")
    
    result.append("")
    
    # Add statements with namespace isolation
    for i, stmt in enumerate(cleaned_statements, 1):
        result.extend([
            f"-- Statement {i}",
            f"namespace Statement{i}",
            stmt,
            f"  #eval \"Statement {i} syntax check passed\"",
            f"end Statement{i}",
            ""
        ])
    
    # Add verification summary
    result.append("-- Verification Summary")
    result.append("#eval \"All statements syntax check completed\"")
    
    return "\n".join(result)

def analyze_lean4_results(result_json: Dict[str, Any]) -> List[bool]:
    """
    Analyze Lean 4 execution results and return syntax check results for each code block
    
    Args:
        result_json: JSON object containing Lean 4 execution results
    
    Returns:
        List of boolean values indicating syntax check results (True = passed, False = failed)
    """
    # Get error information
    errors = result_json.get('info', {}).get('errors', [])
   # errors = result_json.get('errors', [])
   # print(errors)
    # Get code block line ranges
    block_ranges = get_block_ranges(result_json)
    
    # Initialize results list (default all passed)
    results = [True] * len(block_ranges)
    try:
        sys_errors = result_json['info']['system_errors']
    except:
        return [None] * len(block_ranges)

    if sys_errors:
        return [None] * len(block_ranges)
    # Process error messages
    for error in errors:
        error_line = error.get('pos', {}).get('line', 0)
        
        # Locate error in code blocks
        for i, (start, end) in enumerate(block_ranges):
            if start <= error_line <= end:
                results[i] = False
                break
    return results

def get_block_ranges(result_json: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Get line number ranges for each code block
    
    Args:
        result_json: JSON object containing Lean 4 execution results
    
    Returns:
        List of tuples representing line ranges (start, end) for each code block
    """
    verified_code = result_json.get('verified_code', '')
    lines = verified_code.split('\n')

    block_ranges = []
    start_line = 0
    block_number = 0
    
    for i, line in enumerate(lines, 1):
        if line.startswith('namespace Statement'):
            # 开始新的代码块
            start_line = i
            block_number = int(line.split('namespace Statement', 1)[1].strip())
        elif line.startswith('end Statement'):
            # 结束当前代码块
            end_line = i
            block_ranges.append((start_line, end_line))
    
    return block_ranges

def basic_check(s):
    return check_brackets_balance(s) and bool(re.search(r'by\s+sorry', s))

def extract_last_lean4_code_block(text):
    """
    Extract the last Lean4 code block from a string
    
    Args:
        text (str): Input string containing code blocks
        
    Returns:
        str: Content of the last Lean4 code block wrapped in code markers.
             Returns empty code block if none found.
    """

    pattern = r'```lean4\s*\n(.*?)\n\s*```'
    

    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return '```lean4\n' + matches[-1].strip() + '\n```'
    else:
        return '```lean4\n\n```'