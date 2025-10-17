
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
      #      print(r[0])
            try:
                f.write(json.dumps(r, ensure_ascii=False)+"\n")
            except:
                try:
                    f.write(json.dumps(r, ensure_ascii=True)+"\n")
                except:
                    raise ValueError
                
def check_brackets_balance(text: str) -> bool:
        """检查括号平衡"""
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
    """提取代码中的导入语句"""
    import_pattern = r'^\s*import\s+([A-Za-z0-9_\.]+)'
    imports = set()
    
    for line in code.split('\n'):
        match = re.match(import_pattern, line)
        if match:
            if not 'markers' in match.group(0).strip():
                imports.add(match.group(0).strip())
            
    return imports

def group_by_imports(statements) -> Dict[frozenset, List[str]]:
    """根据import库分组"""
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
    """将多个 Lean 4 定理陈述合并到一个文件中"""
    # 提取所有导入语句
    all_imports = set()
    for stmt in statements:
        all_imports.update(extract_imports(stmt))
    
    # 移除陈述中的导入语句
    cleaned_statements = []
    for stmt in statements:
        cleaned_stmt = stmt
        for import_stmt in all_imports:
            cleaned_stmt = re.sub(re.escape(import_stmt) + r'\s*\n', '', cleaned_stmt)
        cleaned_statements.append(cleaned_stmt.strip())
    
    # 构建合并后的代码
    current_date = time.strftime("%Y-%m-%d")
    
    # 头部和导入
    result = [
        f"-- Lean 4 语法验证文件",
        f"-- 自动生成于 {current_date}",
        "",
        "-- 共享导入区域"
    ]
    
    # 添加所有导入
    if all_imports:
        result.extend(sorted(all_imports))
    else:
        result.append("-- 无导入语句")
    
    result.append("")
    
    # 添加每个陈述，用命名空间隔离
    for i, stmt in enumerate(cleaned_statements, 1):
        result.extend([
            f"-- 陈述 {i}",
            f"namespace Statement{i}",
            stmt,
            f"  #eval \"陈述 {i} 语法检查通过\"",
            f"end Statement{i}",
            ""
        ])
    
    # 添加验证汇总
    result.append("-- 验证汇总")
    result.append("#eval \"所有陈述语法检查完成\"")
    
    return "\n".join(result)

def analyze_lean4_results(result_json: Dict[str, Any]) -> List[bool]:
    """
    分析 Lean 4 执行结果，返回每个代码块的语法检查结果。
    
    Args:
        result_json: Lean 4 执行结果的 JSON 对象
    
    Returns:
        List[bool]: 每个代码块的语法检查结果，True 表示通过，False 表示失败
    """
    # 获取错误信息
    errors = result_json.get('info', {}).get('errors', [])
   # errors = result_json.get('errors', [])
   # print(errors)
    # 获取代码块的行号范围
    block_ranges = get_block_ranges(result_json)
    
    # 初始化结果列表（默认所有代码块都通过）
    results = [True] * len(block_ranges)
    try:
        sys_errors = result_json['info']['system_errors']
    except:
        return [None] * len(block_ranges)



    if sys_errors:
        return [None] * len(block_ranges)
    # 处理错误信息
    for error in errors:
        # 获取错误位置
        error_line = error.get('pos', {}).get('line', 0)
        
        # 确定错误所在的代码块
        for i, (start, end) in enumerate(block_ranges):
            if start <= error_line <= end:
                results[i] = False
                break
    return results

def get_block_ranges(result_json: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    获取代码块的行号范围。
    
    Args:
        result_json: Lean 4 执行结果的 JSON 对象
    
    Returns:
        List[Tuple[int, int]]: 每个代码块的行号范围 (开始行, 结束行)
    """
    # 获取验证后的代码
    # verified_code = result_json['info'].get('verified_code', '')
    verified_code = result_json.get('verified_code', '')
    lines = verified_code.split('\n')
    # print(lines)
    # 查找每个命名空间的开始和结束行
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
    提取字符串中最后一个Lean4代码块
    
    Args:
        text (str): 输入的字符串
        
    Returns:
        str: 最后一个Lean4代码块的内容，如果没有找到则返回空的lean4代码块
    """
    # 修改正则表达式，允许结束标记前有空白字符
    pattern = r'```lean4\s*\n(.*?)\n\s*```'
    
    # 使用 re.DOTALL 标志让 . 匹配换行符
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        # 返回最后一个匹配的代码块内容，包装在lean4代码块中
        return '```lean4\n' + matches[-1].strip() + '\n```'
    else:
        # 如果没有找到，返回空的lean4代码块
        return '```lean4\n\n```'