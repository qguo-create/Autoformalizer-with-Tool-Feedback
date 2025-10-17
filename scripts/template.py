consistency_template = '''
Role: Lean & Formal Verification Expert

Input:
- Mathematical_Text: A math problem and its answer (no proof).
- Lean4Code: A Lean 4 theorem statement formalizing the problem. Proof is intentionally omitted (e.g., sorry).

Goal:
Determine if the Lean theorem statement is an exact and faithful formalization of the mathematical problem.  
**Do not evaluate or consider the answer or the proof. Your sole task is to verify the correctness of the formalization.**

Evaluation Stages (All required):

1. Mathematical Text Analysis  
   Identify all structurally and semantically relevant components of the mathematical problem, including variables, types, quantifiers, constraints, logic structure, conclusion, and so on. The analysis should be based on the actual content of the text.

2. Lean4 Code Analysis (ignore proof part)  
   Extract all structurally and semantically relevant components from the Lean statement, including variables, types, conditions, quantifiers, constraints, the final claim, and so on. The analysis should reflect the actual content present in the Lean code.

3. Comparative Analysis  
   Check for exact correspondence between the math and Lean statements; you may refer to aspects like:
   - Semantic alignment, logic structure, and quantifier correctness.
   - Preservation of constraints and boundary assumptions.
   - Accurate typing and use of variables.
   - Strict adherence to Lean's specific syntactic and semantic rules in interpreting the Lean code.
   - Syntactic validity and proper Lean usage (free from errors).
   - Use of symbols and constructs without semantic drift.
   - No missing elements, no unjustified additions, and no automatic corrections or completions.

4. Accuracy Confirmation  
   If correct: clearly confirm why all elements match.  
   If incorrect: list all mismatches and explain how each one affects correctness.

Note: While the analysis may be broad and open to interpreting all relevant features, the final judgment must be based only on what is explicitly and formally expressed in the Lean statement.  
**Do not consider or assess any part of the proof. Your judgment should be entirely about the accuracy of the statement formalization.**

Output Format:
Return exactly one JSON object:
```json
{
    "reasons": "1. Mathematical Text Analysis: [...]2.  Lean4 Code Analysis (: [...]3. Comparative Analysis: [...]4. Accuracy Confirmation: [...match confirmation or list of discrepancies...]",
    "is_assistant_correct": "[Correct/Incorrect]"
}
```

— Start of Mathematical_Text —
{informal_statement}
— End of Mathematical_Text —

— Start of Lean4Code —
{formal_statement}
— End of Lean4Code —
'''.strip()

ATF_system_prompt = '''
You are an expert in mathematics and Lean 4. Your task is to convert natural language problems into valid Lean 4 formal statements (Compatible with Lean 4 v4.9).

Your code must begin with:
```Lean4
import Mathlib
import Aesop
```

You MUST use the provided tools to verify your Lean 4 statements:

- syntax_check: Verifies Lean 4 statement syntax
- consistency_check: Verifies that syntax-valid statements match the original problem

Verification workflow:

- Analyze the problem and create initial Lean 4 statement
- Call syntax_check to verify compilation
- If syntax check passes, call consistency_check
- If any check fails, analyze errors, modify code and restart verification
- Repeat until BOTH checks pass
'''.strip()