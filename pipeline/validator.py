from __future__ import annotations
import ast, json, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, Tuple, List, Optional
from llm.client import chat_for_stage, system
from pipeline.utils import safe_str as _safe_str
from config import ENABLE_LLM_SEMANTIC_CHECK

def _detect_model_var(code: str) -> str:
    """Return the variable name assigned from CpoModel() (e.g. 'mdl', 'model')."""
    m = re.search(r'(\w+)\s*=\s*CpoModel\s*\(', code)
    return m.group(1) if m else "mdl"

DOCPLEX_KEYWORDS = [
    "interval_var", "integer_var", "binary_var",
    "no_overlap", "alternative", "end_before_start", "start_before_start",
    "minimize", "maximize", "sum", "count", "element", "all_diff",
    "start_of", "end_of", "size_of",
    ".add(", ".solve(", "CpoModel",
]


_MATH_FUNC_TO_CODE = {
    "nooverlap": "no_overlap", "alternative": "alternative", "pack": "pack",
    "allowedassignments": "allowed_assignments", "scalprod": "scal_prod",
    "endbeforestart": "end_before_start", "startbeforestart": "start_before_start",
    "endbeforeend": "end_before_end", "alldiff": "all_diff",
    "ifthen": "if_then", "logical_or": "logical_or", "lexicographic": "lexicographic",
    "count": "count", "element": "element", "sequence_var": "sequence_var",
    "startof": "start_of", "endof": "end_of", "sizeof": "size_of",
}

_HARD_GLOBAL_OPS = {
    "no_overlap", "alternative", "pack", "sequence_var",
    "allowed_assignments", "all_diff",
}


def _strip_brackets(expr: str) -> str:
    """Strip array subscripts inside [...] so structural indices are not treated as constraint constants."""
    prev = None
    s = expr
    while prev != s:
        prev = s
        s = re.sub(r"\[[^\[\]]*\]", "[]", s)
    return s


def _numbers(text: str, strip_index: bool = False) -> set:
    """Extract numeric literals from text (normalized to a set of strings)."""
    src = _strip_brackets(text) if strip_index else text
    out = set()
    for tok in re.findall(r"-?\d+(?:\.\d+)?", src):
        try:
            f = float(tok)
        except ValueError:
            continue
        out.add(str(int(f)) if f.is_integer() else str(f))
    return out


def _count_add_statements(code: str) -> int:
    mvar = _detect_model_var(code)
    adds = len(re.findall(re.escape(mvar) + r"\.add\s*\(", code))
    obj_adds = len(re.findall(re.escape(mvar) + r"\.add\s*\(\s*" + re.escape(mvar)
                              + r"\.(?:minimize|maximize)\b", code))
    return max(adds - obj_adds, 0)


def constraint_coverage_check(structured: Dict, math_model: Dict, code: str) -> Dict:
    """Check whether constraints are fully mapped/covered across formal expression -> math model -> code."""
    f_cons = structured.get("constraints", []) or []
    m_cons = math_model.get("constraints_section", []) or []
    n_f, n_m = len(f_cons), len(m_cons)
    n_code = _count_add_statements(code)

    missing_f_in_m = max(n_f - n_m, 0)

    code_l = code.lower()
    ops_missing = []
    missing_global_ops: List[str] = []
    checkable = 0
    for c in m_cons:
        expr = str(c.get("math_expression", "")).lower()
        funcs = [code_kw for mk, code_kw in _MATH_FUNC_TO_CODE.items() if mk in expr]
        if not funcs:
            continue
        checkable += 1
        absent = [kw for kw in funcs if kw not in code_l]
        if absent:
            ops_missing.append(f"{c.get('id','?')}:{c.get('name','')}")
            for kw in absent:
                if kw in _HARD_GLOBAL_OPS and kw not in missing_global_ops:
                    missing_global_ops.append(kw)

    covered = checkable - len(ops_missing)
    cov_ratio = (covered / checkable) if checkable else 1.0
    coverage_ok = (missing_f_in_m == 0 and not ops_missing)

    return {
        "n_formal": n_f, "n_math": n_m, "n_code_add": n_code,
        "missing_formal_in_math": missing_f_in_m,
        "math_ops_missing_in_code": ops_missing,
        "missing_global_ops": missing_global_ops,
        "coverage_ratio": round(cov_ratio, 3),
        "coverage_ok": coverage_ok,
    }


def _percent_equivalents(num_str: str) -> set:
    """Given a numeric string, return the set of equivalent forms it may take in
    the code.

    E.g. "95" as a percentage is usually written "0.95" in code; "105" -> "1.05";
    conversely "0.95" may appear as "95". This avoids flagging 95% / 0.95 as
    "missing".
    """
    out = {num_str}
    try:
        f = float(num_str)
    except ValueError:
        return out
    for factor in (0.01, 100):
        cand = f * factor
        cs = str(int(cand)) if float(cand).is_integer() else str(cand)
        out.add(cs)
    return out


_NON_BINDING_MARKERS = (
    "%", "percent", "present but not enforced", "not enforced",
    "not binding", "non-binding", "relaxed",
    "nominal", "derive", "derived", "reference only", "informational",
)


def numeric_consistency_check(structured: Dict, math_model: Dict, code: str) -> Dict:
    """Check that salient numbers are consistent across formal expression / math
    model / code (guard against numeric drift).

    Important: this check is ADVISORY (a diagnostic signal); its verdict does not
    gate overall_valid, and it is multiply filtered to avoid false positives that
    would mislead a fix round into rewriting already-correct code:
      - skip numbers in percentage/non-binding/derived contexts (e.g. 95% written
        as 0.95 in code is correct);
      - a number and its x0.01 / x100 equivalents are treated as equal
        (percentage <-> decimal);
      - a number is "missing" only when neither it nor any equivalent form is in
        the code.
    """
    formal_nums: set = set()
    for p in (structured.get("parameters", []) or []):
        vsrc = str(p.get("value_or_source", ""))
        if any(mk in vsrc.lower() for mk in _NON_BINDING_MARKERS):
            continue
        formal_nums |= _numbers(vsrc)
    for c in (structured.get("constraints", []) or []):
        desc = str(c.get("description", ""))
        if any(mk in desc.lower() for mk in _NON_BINDING_MARKERS):
            continue
        formal_nums |= _numbers(desc, strip_index=True)

    m_text = " ".join(str(c.get("math_expression", "")) for c in
                      (math_model.get("constraints_section", []) or []))
    m_text += " " + str(math_model.get("objective_section", {}).get("math_expression", ""))
    math_nums = _numbers(m_text, strip_index=True)

    code_nums = _numbers(code)
    code_nums_expanded = set(code_nums)
    for cn in list(code_nums):
        code_nums_expanded |= _percent_equivalents(cn)

    salient = (formal_nums | math_nums) - {"0", "1"}

    missing_in_code = []
    for x in sorted(salient, key=lambda v: float(v)):
        forms = _percent_equivalents(x)
        if not (forms & code_nums_expanded):
            missing_in_code.append(x)

    scale = []
    for x in missing_in_code:
        fx = float(x)
        for factor in (10, 0.1):
            cand = fx * factor
            cs = str(int(cand)) if float(cand).is_integer() else str(cand)
            if cs in code_nums:
                scale.append(f"{x}→{cs}")
                break

    return {
        "formal_numbers": sorted(formal_nums, key=lambda x: float(x)),
        "math_numbers":   sorted(math_nums, key=lambda x: float(x)),
        "salient_numbers": sorted(salient, key=lambda x: float(x)),
        "missing_in_code": missing_in_code,
        "possible_scale_changes": scale,
        "consistency_ok": not missing_in_code,
    }


_CRITICAL_SCALAR_PARAMS = {
    "nbDemandAreas", "nbLocations", "num_vehicles", "vehicle_capacity",
    "depot", "node_to_skip", "n_words", "MAX_ALLOWED_COST", "MAX_COST",
    "MAX_UTILIZATION_RATIO",
}


def critical_scalar_assignment_check(structured: Dict, code: str) -> List[str]:
    """Hard-fail when code assigns an explicit current-instance scalar incorrectly."""
    expected = {}
    for p in structured.get("parameters", []) or []:
        name = str(p.get("name", "")).strip()
        var = re.split(r"[\[(]", name, 1)[0].strip()
        if var not in _CRITICAL_SCALAR_PARAMS:
            continue
        vsrc = str(p.get("value_or_source", "")).strip()
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*$", vsrc)
        if m:
            expected[var] = m.group(1)
    if not expected:
        return []

    issues = []
    for var, got in re.findall(
        r"^\s*([A-Za-z_]\w*)\s*=\s*(-?\d+(?:\.\d+)?)\s*$",
        code or "",
        flags=re.MULTILINE,
    ):
        want = expected.get(var)
        if want is None:
            continue
        if float(got) != float(want):
            issues.append(f"{var} assigned {got}, but current problem requires {want}")
    return issues


def syntax_check(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


# Conservative undefined-name check catches runtime NameError missed by ast.parse.
import builtins as _builtins

_PY_BUILTINS = set(dir(_builtins)) | {"__file__", "__name__", "__doc__"}


def _scope_bound_names(node) -> set:
    """Collect the names directly bound by a scope node (module/function/lambda), not descending into nested scopes."""
    bound: set = set()

    class _C(ast.NodeVisitor):
        def visit_FunctionDef(self, n): bound.add(n.name)
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_ClassDef(self, n): bound.add(n.name)
        def visit_Lambda(self, n): pass
        def visit_Import(self, n):
            for a in n.names: bound.add((a.asname or a.name).split(".")[0])
        def visit_ImportFrom(self, n):
            for a in n.names: bound.add(a.asname or a.name)
        def visit_Name(self, n):
            if isinstance(n.ctx, (ast.Store, ast.Del)): bound.add(n.id)
        def visit_arg(self, n): bound.add(n.arg)
        def visit_Global(self, n):
            for x in n.names: bound.add(x)
        def visit_Nonlocal(self, n):
            for x in n.names: bound.add(x)

    c = _C()
    for ch in ast.iter_child_nodes(node):
        c.visit(ch)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = node.args
        for arg in a.args + a.posonlyargs + a.kwonlyargs:
            bound.add(arg.arg)
        if a.vararg: bound.add(a.vararg.arg)
        if a.kwarg:  bound.add(a.kwarg.arg)
    return bound


def undefined_name_check(code: str) -> List[str]:
    """Return descriptions of names used in the code but defined in no enclosing scope (conservative, near-zero false positives)."""
    occ = _undefined_occurrences(code)
    seen, out = set(), []
    for n, l in occ:
        if n not in seen:
            seen.add(n)
            out.append(f"Undefined name '{n}' used at line {l} "
                       f"(never assigned/imported — will raise NameError at runtime).")
    return out


def undefined_names(code: str) -> set:
    """Return the set of undefined names in the code (names only, for the post-processing safety net to compare before/after)."""
    return {n for n, _ in _undefined_occurrences(code)}


def _undefined_occurrences(code: str) -> List[Tuple[str, int]]:
    """Core traversal: return [(undefined_name, line_no), ...]."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found: List[Tuple[str, int]] = []

    def check(node, stack):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _PY_BUILTINS:           return
            if any(node.id in s for s in stack):  return
            found.append((node.id, getattr(node, "lineno", 0)))
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comp_bound = set()
            for gen in node.generators:
                for t in ast.walk(gen.target):
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                        comp_bound.add(t.id)
            inner = stack + [comp_bound]
            for gen in node.generators:
                check(gen.iter, stack)
                for cond in gen.ifs: check(cond, inner)
            if isinstance(node, ast.DictComp):
                check(node.key, inner); check(node.value, inner)
            else:
                check(node.elt, inner)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            inner = stack + [_scope_bound_names(node)]
            body = [node.body] if isinstance(node, ast.Lambda) else node.body
            for b in body: check(b, inner)
            return
        for ch in ast.iter_child_nodes(node):
            check(ch, stack)

    module_scope = _scope_bound_names(tree)
    for stmt in tree.body:
        check(stmt, [module_scope])
    return found


def antipattern_check(code: str) -> List[str]:
    """Flag known-bad constructs that compile but fail at solve time or that
    contradict the knowledge-base template (e.g. invented toy data)."""
    issues = []
    if re.search(r"(presence_of|size_of|start_of|end_of|element)\s*\([^)]*\)\s*>>", code):
        issues.append("Invalid '>>' operator on CP expressions — use mdl.if_then(cond, expr).")
    if re.search(r"mdl\.\w+\([^\n]*\)\s*>>\s*\(", code):
        issues.append("Invalid '>>' on a CP expression — replace with mdl.if_then(...).")
    return issues


# Detect obvious scheduling/routing paradigm mismatches.
_SCHEDULING_MARKERS = ("interval_var", "no_overlap", "makespan",
                       "machine_operations", "end_before_start",
                       "start_of", "end_of")
_ROUTING_MARKERS = ("distance", "arc", "route", "depot", "subtour",
                    "tour", "flow", "demand", "capacity", "u[")


def _is_routing_type(problem_type: str) -> bool:
    p = (problem_type or "").lower()
    return ("vrp" in p) or ("rout" in p) or ("vehicle" in p) or ("traveling salesman" in p)


def paradigm_check(code: str, structured: Dict) -> List[str]:
    """When the problem type is VRP/routing, check whether the code was wrongly
    written as a job-shop scheduling model.

    Criteria (conservative, aiming for zero false positives):
      - type is routing (VRP), and
      - code has >= 2 scheduling features (interval_var / machine no_overlap /
        makespan ...), and
      - code has almost no routing features (distance / arc / route / depot /
        subtour / flow ...)
    If all hold, it is judged "modeled as job-shop" and a blocking error is
    returned.
    """
    ptype = structured.get("problem_type", "") or structured.get("matched_domain", "")
    if not _is_routing_type(ptype):
        return []
    low = code.lower()
    sched_hits = sum(1 for m in _SCHEDULING_MARKERS if m in low)
    route_hits = sum(1 for m in _ROUTING_MARKERS if m in low)
    if sched_hits >= 2 and route_hits <= 1:
        return [("Wrong modeling paradigm: this is a Vehicle Routing Problem (VRP) "
                 "but the code is written as a job-shop SCHEDULING model "
                 f"(scheduling markers={sched_hits}, routing markers={route_hits}). "
                 "Do NOT model VRP with interval_var / machine no_overlap / minimize "
                 "makespan. Re-model it as routing: binary arc variables arc[v][i][j], "
                 "service-coverage (each customer visited once), flow conservation, "
                 "depot departure/return, vehicle capacity, subtour elimination (e.g. "
                 "MTZ u[v][i]), and objective = minimize total travel distance "
                 "sum(arc[v][i][j]*distance_matrix[i][j]). Follow the VRP reference "
                 "template's data loading and variable structure.")]
    return []


def static_semantic_check(code: str) -> Tuple[bool, List[str]]:
    mvar = _detect_model_var(code)
    checks = [
        (r'from\s+docplex\.cp\.model\s+import\s+CpoModel', "Missing docplex.cp import"),
        (r'CpoModel\s*\(', "Missing model initialization: CpoModel()"),
        (re.escape(mvar) + r'\.add\s*\(', f"No {mvar}.add() calls found — constraints not added"),
        (re.escape(mvar) + r'\.solve\s*\(', f"Missing {mvar}.solve() call"),
    ]
    errors = [msg for pat, msg in checks if not re.search(pat, code)]
    errors += antipattern_check(code)
    # undefinedvariable → NameError feedback loopfix
    errors += undefined_name_check(code)
    return len(errors) == 0, errors


def _ngram_overlap(code: str, reference_tokens: List[str], n: int = 2) -> float:
    """Compute n-gram overlap between code tokens and reference tokens."""
    code_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+|[()[\].,=<>!+\-*/]', code)
    if not code_tokens or not reference_tokens:
        return 0.0
    code_ngrams = set(
        tuple(code_tokens[i:i+n]) for i in range(len(code_tokens) - n + 1)
    )
    ref_ngrams = set(
        tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens) - n + 1)
    )
    if not ref_ngrams:
        return 0.0
    return len(code_ngrams & ref_ngrams) / len(ref_ngrams)


def _keyword_match(code: str) -> float:
    """Score keyword coverage: fraction of expected docplex keywords present."""
    present = sum(1 for kw in DOCPLEX_KEYWORDS if kw in code)
    return present / len(DOCPLEX_KEYWORDS)


def _ast_structure_score(code: str) -> float:
    """
    Lightweight AST structural score: checks presence of expected node types
    consistent with well-formed docplex.cp programs.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    node_types = {type(n).__name__ for n in ast.walk(tree)}
    expected = {"ImportFrom", "Assign", "For", "Call", "If", "Expr"}
    coverage = len(expected & node_types) / len(expected)
    return coverage


def compute_code_quality_score(code: str, math_model: Dict) -> Dict:
    """
    Compute a CodeBLEU-inspired multi-dimensional quality score.
    Returns dict with individual component scores and combined score.

    Components (inspired by CodeBLEU metric, Paper §Experimental analysis):
      - syntax_ok:      AST parse success (binary)
      - keyword_match:  Docplex API keyword coverage
      - ast_structure:  AST node type diversity
      - ngram_overlap:  Token n-gram overlap with model spec tokens

    Note: Full CodeBLEU requires reference code; this is a self-contained proxy.
    """
    syn_ok, _ = syntax_check(code)

    # Build reference token set from math model spec
    ref_text = json.dumps(math_model, ensure_ascii=False)
    ref_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', ref_text)

    kw_score  = _keyword_match(code)
    ast_score = _ast_structure_score(code)
    ng_score  = _ngram_overlap(code, ref_tokens, n=2) if syn_ok else 0.0

    # Combined: weighted average matching paper's multi-dimensional assessment
    combined = (0.25 * float(syn_ok) +
                0.30 * kw_score +
                0.25 * ast_score +
                0.20 * ng_score)

    return {
        "syntax_ok":        syn_ok,
        "keyword_match":    round(kw_score, 4),
        "ast_structure":    round(ast_score, 4),
        "ngram_overlap":    round(ng_score, 4),
        "code_quality_score": round(combined, 4),
    }


def llm_semantic_check(code: str, math_model: Dict, structured: Dict) -> Dict:
    c_summary = json.dumps(math_model.get("constraints_section", []),
                           ensure_ascii=False)[:1000]
    obj_summary = json.dumps(math_model.get("objective_section", {}),
                             ensure_ascii=False)[:300]
    # NOTE: pass the FULL code — never truncate. Truncating made the reviewer see
    # a half-cut program and hallucinate "code is incomplete / X not defined"
    # false negatives, which then triggered destructive rewrites of correct code.
    prompt = f"""Check whether this docplex.cp code correctly implements the CP model.

Model constraints:
{c_summary}

Model objective:
{obj_summary}

Code:
```python
{code}
```

Return JSON:
{{
  "valid": true|false,
  "confidence": 0.0-1.0,
  "errors": [],
  "suggestions": [],
  "coverage_check": {{
    "constraints_covered": [],
    "constraints_missing": []
  }}
}}"""
    raw = chat_for_stage("validation", [{"role": "system", "content": system("validator")},
                {"role": "user",   "content": prompt}],
               json_mode=True, temperature=0.1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else {
            "valid": False, "errors": ["LLM response parse failed"], "suggestions": []
        }


def validate(code: str, math_model: Dict, structured: Dict,
             verbose: bool = True, llm_check: Optional[bool] = None) -> Dict:
    """
    Dual verification (Paper §Validation and iterative optimization module):
      (1) static analysis — syntax (ast.parse) + required docplex API patterns;
      (2) LLM semantic alignment — math model ↔ code, run by a general-purpose
          LLM API. This is the semantic leg of the paper's dual-verification
          mechanism and is now ENABLED BY DEFAULT (controlled by the config flag
          ENABLE_LLM_SEMANTIC_CHECK; pass llm_check=True/False to force it).

    Gating:
        overall_valid = syntax_ok AND static_ok       (executability gate, stable)
        semantic_ok   = LLM semantic alignment valid   (diagnostic + drives the
                        feedback loop's missing-constraint hints; advisory so an
                        occasional LLM false-negative cannot destroy correct code)

    NOTE: This function performs STATIC + LLM checks only. The third leg of the
    paper's dual verification — *dynamic* execution with the CP solver for
    real-time error interpretation and correction — is performed inside the
    feedback loop (pipeline/feedback.run_loop), where a runtime/solve error is
    fed back to the API fix pass. Keeping execution out of this function lets
    every non-loop caller (GUI/web "check code") validate without needing a
    working CPLEX install.
    """
    if llm_check is None:
        llm_check = ENABLE_LLM_SEMANTIC_CHECK
    print("\n" + "="*60)
    print("[Validation] Static check (syntax + required API patterns)")
    print("="*60)

    report = {
        "syntax_ok": False, "static_ok": False,
        "llm_result": {}, "all_errors": [], "all_suggestions": [],
        "advisory_notes": [],
        "overall_valid": False, "score": 0.0,
        "code_quality": {},
    }

    ok, msg = syntax_check(code)
    report["syntax_ok"] = ok
    if not ok:
        report["all_errors"].append(f"[Syntax] {msg}")
    if verbose:
        print(f"  {'✓' if ok else '✗'} Syntax check: {msg}")

    ok2, errs = static_semantic_check(code)
    report["static_ok"] = ok2
    report["all_errors"].extend(f"[Static] {e}" for e in errs)
    if verbose:
        if ok2:
            print(f"  ✓ Static semantic check: all required API patterns present")
        else:
            for e in errs:
                print(f"  ✗ {e}")

    paradigm_errs = paradigm_check(code, structured)
    if paradigm_errs:
        report["static_ok"] = False
        report["all_errors"].extend(f"[Paradigm] {e}" for e in paradigm_errs)
        report["paradigm_ok"] = False
        if verbose:
            for e in paradigm_errs:
                print(f"  ✗ [Paradigm] {e[:120]}...")
    else:
        report["paradigm_ok"] = True

    cq = compute_code_quality_score(code, math_model)
    report["code_quality"] = cq
    if verbose:
        print(f"  Code quality: keyword={cq['keyword_match']:.2f}  "
              f"ast={cq['ast_structure']:.2f}  "
              f"ngram={cq['ngram_overlap']:.2f}  "
              f"→ quality={cq['code_quality_score']:.2f}")

    cov = constraint_coverage_check(structured, math_model, code)
    report["constraint_coverage"] = cov
    report["coverage_ok"] = cov["coverage_ok"]
    if cov["missing_formal_in_math"]:
        report["advisory_notes"].append(
            f"[Coverage] math model may be missing {cov['missing_formal_in_math']} constraint(s) "
            f"(formal {cov['n_formal']} -> math {cov['n_math']})")
    for m in cov["math_ops_missing_in_code"]:
        report["advisory_notes"].append(f"[Coverage] constraint possibly not implemented: {m}")
    if verbose:
        sym = "✓" if cov["coverage_ok"] else "✗"
        print(f"  {sym} constraint coverage: formal {cov['n_formal']} -> math {cov['n_math']} "
              f"-> code add {cov['n_code_add']}  coverage={cov['coverage_ratio']:.2f}")
        if cov["math_ops_missing_in_code"]:
            print(f"    constraints missing in code: {', '.join(cov['math_ops_missing_in_code'][:6])}")

    num = numeric_consistency_check(structured, math_model, code)
    report["numeric_consistency"] = num
    report["numeric_ok"] = num["consistency_ok"]
    if num["missing_in_code"]:
        msg = f"[Numeric] salient values absent from code: {', '.join(num['missing_in_code'][:8])}"
        if num["possible_scale_changes"]:
            msg += f"  (possible scale change: {', '.join(num['possible_scale_changes'][:5])})"
        report["advisory_notes"].append(msg)
    if verbose:
        sym = "✓" if num["consistency_ok"] else "✗"
        print(f"  {sym} numeric consistency: salient={num['salient_numbers'][:8]}")
        if num["missing_in_code"]:
            print(f"    values missing/changed in code: {num['missing_in_code'][:8]}"
                  + (f"  possible scale change {num['possible_scale_changes'][:5]}"
                     if num["possible_scale_changes"] else ""))

    critical_scalar_errs = critical_scalar_assignment_check(structured, code)
    if critical_scalar_errs:
        report["static_ok"] = False
        report["all_errors"].extend(f"[Numeric] {e}" for e in critical_scalar_errs)
        if verbose:
            for e in critical_scalar_errs:
                print(f"  ✗ [Numeric] {e}")

    # LLM semantic alignment is advisory and never gates validity.
    if llm_check:
        print(f"  Running LLM semantic alignment check (advisory)...")
        try:
            llm = llm_semantic_check(code, math_model, structured)
        except Exception as e:
            llm = {"valid": False, "confidence": 0.0,
                   "errors": [f"LLM check failed: {e}"], "suggestions": [],
                   "coverage_check": {"constraints_covered": [], "constraints_missing": []}}
        report["llm_result"] = llm
        report["all_suggestions"].extend(llm.get("suggestions", []))
        if verbose:
            sym = "✓" if llm.get("valid") else "✗"
            print(f"  {sym} LLM check (advisory): valid={llm.get('valid')} "
                  f"confidence={llm.get('confidence', 0):.2f}")
            cov = llm.get("coverage_check", {})
            missing = [_safe_str(x) for x in cov.get("constraints_missing", [])]
            if missing:
                print(f"    Missing : {', '.join(missing[:6])}")

    score = (0.40 * float(report["syntax_ok"]) +
             0.40 * float(report["static_ok"]) +
             0.20 * cq["code_quality_score"])
    report["score"] = round(score, 3)

    report["overall_valid"] = report["syntax_ok"] and report["static_ok"]
    report["semantic_ok"] = bool(report.get("llm_result", {}).get("valid")) if llm_check else None
    report["mapping_ok"] = bool(report.get("coverage_ok") and report.get("numeric_ok"))

    if verbose:
        status = "✓ PASS" if report["overall_valid"] else "✗ NEEDS FIX"
        mp = "✓" if report["mapping_ok"] else "✗"
        print(f"\n  Score: {report['score']:.2f}/1.00  [{status}]  mapping-consistency[{mp}]")
    return report
