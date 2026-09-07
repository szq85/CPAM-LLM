from __future__ import annotations
import re, json, sys, os, ast
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, Optional, Tuple
from llm.client import chat_for_stage, system, chat_ft_with_api_fallback
from rag_fca.knowledge_base import KnowledgeBase
from config import (FT_STAGE3_SYSTEM, FT_STAGE3_USE_GOLD_LOADER,
                    FT_STAGE3_FIX_CONSTANTS, FT_STAGE3_USE_SKELETON_CONTRACT,
                    FT_STAGE3_POSTPROCESS_FIX)

_REFS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "data", "formal_expression_refs.json")
try:
    with open(_REFS_PATH, encoding="utf-8") as _f:
        _FORMAL_REFS: Dict[str, Dict] = json.load(_f)
except Exception:
    _FORMAL_REFS = {}

STYLE_RULES = """
STRICT CODE STYLE (match the training dataset exactly):
1. NO comments (no # lines, no docstrings) anywhere.
2. First line: from docplex.cp.model import CpoModel  (plus any other needed imports).
3. Read input data from the SAME file and SAME parser as the reference template;
   never fabricate / hard-code toy data.
4. Initialise the model once (e.g. mdl = CpoModel() OR model = CpoModel());
   use that same variable for .add(...) and .solve(...) consistently.
5. Define variables (interval_var / integer_var / binary_var ...).
6. Add EVERY constraint with <model>.add(...).
7. Objective with <model>.add(<model>.minimize(...)) / .maximize(...) when applicable.
8. End with: msol = <model>.solve(TimeLimit=30)
   if msol: print results; else: print("No solution found.")
For conditional constraints use <model>.if_then(cond, expr) — the Python '>>'
operator is INVALID on CP expressions.
9. Structural consistency between parsing and use: whatever shape you build when
   parsing the data file MUST match how you later unpack it. If each element of a
   list is a list of (machine, duration) tuples (e.g. you did job.append(choices)),
   then iterate it as `for op in job: for (m, d) in op:` — do NOT unpack it as
   `for (choices, ctype) in job`. Per-operation side data (e.g. tight/loose
   constraint flags) lives in its OWN parallel list (e.g. CONSTRAINTS[jx][ox]),
   not inside the choices element. A mismatch here raises
   "cannot unpack non-iterable int object" at runtime — verify it mentally before
   returning the code.
"""


def _base_template_block(structured: Dict, kb: KnowledgeBase) -> str:
    """
    Return the knowledge-base base template (canonical data-loading skeleton) for
    the problem type, as a reference block for the API paths. Empty string if none.
    """
    ptype = structured.get("problem_type", "")
    skeleton = structured.get("base_template") or (kb.base_template(ptype) if kb else "")
    if not skeleton:
        return ("## No base template available — read input from the data file named "
                "in the problem description; do NOT hard-code toy data.")
    return f"""## REFERENCE TEMPLATE (knowledge base — this problem type's base model).
KEEP its data-loading EXACTLY: read the SAME input file in the SAME format; do NOT
invent or hard-code toy data. KEEP its variable construction and control flow, then
ensure every constraint in the math model is added. Extend this template, do not
paraphrase it:

```python
{skeleton}
```"""


def _strip_fences(raw: str) -> str:
    m = re.search(r'```(?:python)?\n?(.*?)```', raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


# Loader grafting replaces fragile generated parsers with KB-proven parsers.

_MODEL_CREATE_RE = re.compile(r'^[ \t]*\w+[ \t]*=[ \t]*CpoModel[ \t]*\(', re.MULTILINE)


def _split_at_model(code: str):
    """Split at the first `<var> = CpoModel(` and return (loader_head, model_body).
    If no model-creation line is found, return (code, None)."""
    m = _MODEL_CREATE_RE.search(code or "")
    if not m:
        return code, None
    return code[:m.start()], code[m.start():]


def _loader_prelude(skeleton: str) -> str:
    """Take the data-loading/preprocessing block (before model creation) from base_template."""
    if not skeleton:
        return ""
    head, body = _split_at_model(skeleton)
    return head.rstrip() if body is not None else ""


_ROW_LEVEL_SCALAR_NAMES = {
    # Dataset slice sizes and row-specific switches. These may differ across
    # generated test rows even when the file parser is shared by one problem type.
    "nbDemandAreas", "nbLocations", "num_vehicles", "vehicle_capacity",
    "depot", "node_to_skip", "n_words", "MAX_ALLOWED_COST", "MAX_COST",
    "MAX_UTILIZATION_RATIO",
}


def _row_level_scalar_assignments(code: str) -> Dict[str, str]:
    """Extract generated row-level scalar assignments that a KB loader must not overwrite."""
    out: Dict[str, str] = {}
    for name, value in re.findall(
        r"^\s*([A-Za-z_]\w*)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*$",
        code or "",
        flags=re.MULTILINE,
    ):
        if name in _ROW_LEVEL_SCALAR_NAMES:
            out[name] = value
    return out


def _restore_row_level_scalars(grafted: str, original_code: str) -> str:
    """After grafting a generic KB loader, restore row-specific generated scalars."""
    scalars = _row_level_scalar_assignments(original_code)
    if not scalars:
        return grafted
    lines = grafted.split("\n")
    seen = set()
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)([A-Za-z_]\w*)(\s*=\s*)([-+]?\d+(?:\.\d+)?)(\s*)$", line)
        if not m:
            continue
        name = m.group(2)
        if name not in scalars:
            continue
        seen.add(name)
        if m.group(4) != scalars[name]:
            lines[i] = f"{m.group(1)}{name}{m.group(3)}{scalars[name]}{m.group(5)}"
    return "\n".join(lines)


def _graft_loader(code: str, prelude: str) -> str:
    """Replace only the fragile file parser while preserving row-level parameters."""
    if not prelude:
        return code
    _, body = _split_at_model(code)
    if body is None:
        return code
    grafted = prelude.rstrip() + "\n\n" + body.lstrip("\n")
    return _restore_row_level_scalars(grafted, code)


# Correct definite scalar constants copied from the formal expression.
_NUM_VALUE_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z%(].*)?$')


def _known_scalar_params(structured: Dict) -> Dict[str, str]:
    """Extract {code variable name: definite numeric string} from the formal-expression parameters."""
    out: Dict[str, str] = {}
    for p in structured.get("parameters", []) or []:
        name = str(p.get("name", "")).strip()
        vsrc = str(p.get("value_or_source", "")).strip()
        if not name or not vsrc:
            continue
        low = vsrc.lower()
        if any(tok in vsrc for tok in ("*", "/", "+", "[", "_")) or \
           any(low.endswith(ext) for ext in (".txt", ".data", ".fjs")) or \
           "file" in low or "comput" in low or "coordinate" in low:
            continue
        m = _NUM_VALUE_RE.match(vsrc)
        if not m:
            continue
        var = re.split(r'[\[(]', name, 1)[0].strip()   # I_load[k] → I_load
        if re.fullmatch(r'[A-Za-z_]\w*', var):
            out.setdefault(var, m.group(1))
    return out


def _harden_int_token_loader(code: str) -> Tuple[str, list]:
    """Fix bare int() parsing for data lines that contain categorical tokens."""
    fixes: list = []
    str_lits = set(re.findall(r"['\"]([a-zA-Z][a-zA-Z_]{1,15})['\"]", code or ""))
    KNOWN = ("tight", "loose", "none", "high", "low", "medium",
             "forward", "reverse", "left", "right")
    tokens = sorted({s for s in str_lits if s in KNOWN})
    if not tokens:
        return code, fixes
    if {"tight", "loose"} & set(tokens):
        tokens = sorted(set(tokens) | {"tight", "loose", "none"})
    toks_repr = ", ".join(f"'{t}'" for t in tokens)

    out_lines = []
    changed = 0
    guard_re = re.compile(r"if\s+\w+\s+(?:not\s+)?in\s*[\(\[]")
    bare_re = re.compile(r"\bint\(\s*(\w+)\s*\)(\s+for\s+\1\s+in\b[^\]]*\.split\(\))")
    two_int_unpack = re.compile(r"^\s*\w+\s*,\s*\w+\s*=\s*\[")   # `A, B = [int(v)...]`

    for line in code.split("\n"):
        if guard_re.search(line) or two_int_unpack.match(line):
            out_lines.append(line); continue
        def _repl(m):
            nonlocal changed
            changed += 1
            v = m.group(1)
            return f"(int({v}) if {v} not in ({toks_repr},) else {v}){m.group(2)}"
        out_lines.append(bare_re.sub(_repl, line))

    if changed:
        new_code = "\n".join(out_lines)
        fixes.append(f"loader hardening: protected {changed} int() cast(s) for string tokens {tokens}")
        return new_code, fixes
    return code, fixes


def _fix_fjsp_unpack(code: str) -> Tuple[str, bool]:
    """
    Fix a frequent FJSP runtime bug: in the gold loader each element of `job` is a
    choices list (`job.append(choices)`), and constraint types are stored
    separately in CONSTRAINTS. But the generated code sometimes writes the
    operation loop as `for ox, (choices, constraint_type) in enumerate(job):`,
    which raises "cannot unpack non-iterable int object" at runtime.

    This rewrites that bad unpack back to `for ox, op in enumerate(job):` and
    replaces references to `constraint_type` in the loop body with
    `CONSTRAINTS[jx][ox]` (the correct source of constraint type in the gold),
    keeping the choices name unchanged (the unpacked value is the choices itself).
    Pure output post-processing, conservative matching, changes only when the bad
    pattern is actually present.
    """
    pat = re.compile(
        r'for\s+(\w+)\s*,\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s+in\s+enumerate\(\s*job\s*\)\s*:'
    )
    head, body = _split_at_model(code)
    target = body if body is not None else code
    m = pat.search(target)
    if not m:
        return code, False

    ox_var, choices_name, ctype_name = m.group(1), m.group(2), m.group(3)
    jx_m = re.search(r'for\s+(\w+)\s*,\s*job\s+in\s+enumerate\(\s*JOBS\s*\)\s*:', target)
    jx_var = jx_m.group(1) if jx_m else "jx"

    # 1) Rewrite the bad unpack to iterate choices directly (the unpacked value is
    #    the choices list itself).
    target = pat.sub(f'for {ox_var}, {choices_name} in enumerate(job):', target, count=1)
    # 2) Replace references to the bad constraint-type name with CONSTRAINTS[jx][ox].
    target = re.sub(rf'\b{re.escape(ctype_name)}\b',
                    f'CONSTRAINTS[{jx_var}][{ox_var}]', target)
    return (head + target if body is not None else target), True


def _fix_membership_in_constraint(code: str) -> Tuple[str, bool]:
    """Rewrite an illegal Python membership test on a CP variable into a docplex-legal
    logical-OR / negation.

    The fine-tuned model sometimes writes a "value is one of a set" constraint as a
    Python `in` test, e.g. (observed on DNA terminal-base constraints):
        model.add(words[i][7] in [2, 3])          # CPO expression cannot be bool
        model.add(seq[k] not in [0, 1])
    docplex evaluates `<cpvar> in [...]` via Python's __bool__, raising
    "CpoException: CPO expression can not be used as boolean". The correct forms are
        model.add((words[i][7] == 2) | (words[i][7] == 3))
        model.add((seq[k] != 0) & (seq[k] != 1))

    Conservative: only rewrites a membership test whose right side is a literal list
    of integers AND that sits inside a `.add(...)` call (i.e. a CP constraint), so
    ordinary Python `for x in [...]` / `if k in [...]` control flow is untouched.
    """
    if ".add(" not in code:
        return code, False

    # Match: <lhs> [not] in [ <ints> ]   where lhs is a simple indexed/var expression.
    memb = re.compile(
        r'(?P<lhs>[A-Za-z_]\w*(?:\[[^\]\[]*\])*)'      # var or var[..][..]
        r'\s+(?P<neg>not\s+)?in\s*'
        r'\[(?P<vals>\s*-?\d+(?:\s*,\s*-?\d+)*\s*,?\s*)\]'
    )

    def _rewrite(m: re.Match) -> str:
        lhs  = m.group("lhs")
        neg  = bool(m.group("neg"))
        vals = [v.strip() for v in m.group("vals").split(",") if v.strip()]
        if not vals:
            return m.group(0)
        if neg:
            terms = " & ".join(f"({lhs} != {v})" for v in vals)
        else:
            terms = " | ".join(f"({lhs} == {v})" for v in vals)
        return f"({terms})"

    changed = False
    out_lines = []
    for line in code.splitlines(keepends=True):
        # Only touch lines that add a constraint and contain a literal-int membership.
        if ".add(" in line and memb.search(line):
            new_line = memb.sub(_rewrite, line)
            if new_line != line:
                changed = True
            out_lines.append(new_line)
        else:
            out_lines.append(line)
    if not changed:
        return code, False
    return "".join(out_lines), True


def _repair_fjsp_loader_constraint_pollution(code: str) -> Tuple[str, bool]:
    """Undo accidental CONSTRAINTS[jx][ox] rewrites inside the FJSP data loader."""
    if "jline.pop(0)" not in code or "CONSTRAINTS[" not in code:
        return code, False
    head, body = _split_at_model(code)
    if body is None:
        return code, False

    lines = head.split("\n")
    changed = False
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)CONSTRAINTS\[[^\n]+\]\[[^\n]+\]\s*=\s*jline\.pop\(0\)\s*$", line)
        if m:
            lines[i] = f"{m.group(1)}constraint = jline.pop(0)"
            changed = True
            continue
        m = re.match(r"^(\s*)constraints\.append\(\s*CONSTRAINTS\[[^\n]+\]\[[^\n]+\]\s*\)\s*$", line)
        if m:
            lines[i] = f"{m.group(1)}constraints.append(constraint)"
            changed = True

    if not changed:
        return code, False
    new_code = "\n".join(lines) + body
    try:
        compile(new_code, "<fjsp-loader-repair>", "exec")
    except SyntaxError:
        return code, False
    return new_code, True


def _detect_model_var(code: str) -> str:
    """Detect the model variable name: prefer `<var> = CpoModel(`, default 'mdl'."""
    m = re.search(r'(\w+)\s*=\s*CpoModel\s*\(', code or "")
    return m.group(1) if m else "mdl"


def _charging_similarity_fraction(structured: Dict, math_model: Dict) -> float:
    """Extract nearest/top percent for Charging Station site-diversity constraints."""
    texts = []
    for c in (structured.get("constraints", []) or []):
        texts.append(str(c.get("description", "")))
        texts.append(str(c.get("type", "")))
    for c in (math_model.get("constraints_section", []) or []):
        texts.append(str(c.get("name", "")))
        texts.append(str(c.get("description", "")))
        texts.append(str(c.get("math_expression", "")))
    blob = " ".join(texts).lower()
    for m in re.finditer(r"(?:nearest|top|closest|similar)[^0-9]{0,30}(\d+(?:\.\d+)?)\s*%", blob):
        val = float(m.group(1)) / 100.0
        if 0 < val <= 1:
            return val
    for m in re.finditer(r"\b0\.(\d+)\b", blob):
        val = float("0." + m.group(1))
        if 0 < val <= 0.5:
            return val
    return 0.10


def _has_charging_geographic_intent(structured: Dict, math_model: Dict, code: str) -> bool:
    ptype = str(structured.get("problem_type", "") or structured.get("matched_domain", "")).lower()
    if "charging" not in ptype:
        return False
    blob = code.lower()
    if any(k in blob for k in ("cost_sims", "pairs_to_prevent_opening", "profile_dist",
                               "cost_sim", "pair_distances", "_close_pairs",
                               "_cost_profiles", "_distances")):
        return True
    texts = []
    for c in (structured.get("constraints", []) or []):
        texts.append(str(c.get("type", "")))
        texts.append(str(c.get("description", "")))
    for c in (math_model.get("constraints_section", []) or []):
        texts.append(str(c.get("name", "")))
        texts.append(str(c.get("description", "")))
        texts.append(str(c.get("math_expression", "")))
    low = " ".join(texts).lower()
    return any(k in low for k in ("geographic", "diversity", "similar", "nearest", "cost_sim", "profile_dist"))


def _fix_charging_geographic_diversity(code: str, structured: Dict, math_model: Dict) -> Tuple[str, list]:
    """Rewrite Charging Station site-diversity into pure Python pair selection.

    LLMs often build `cost_sims` as CP expressions and then use them in a Python
    `if`, which fails before the model is solved. The diversity pair set is data
    preprocessing, so compute it from the parsed cost matrix in Python and only
    add normal linear constraints to the CP model.
    """
    fixes: list = []
    if not _has_charging_geographic_intent(structured, math_model, code):
        return code, fixes
    if not all(k in code for k in ("nbDemandAreas", "nbLocations", "cost", "open_station")):
        return code, fixes

    frac = _charging_similarity_fraction(structured, math_model)
    frac_s = f"{frac:.6g}"
    model_var = _detect_model_var(code)
    block = f"""pair_distances = []
for l1 in range(nbLocations):
    for l2 in range(l1 + 1, nbLocations):
        dist = sum(abs(cost[c][l1] - cost[c][l2]) for c in range(nbDemandAreas))
        pair_distances.append((dist, l1, l2))
pair_distances.sort(key=lambda x: x[0])
num_similar_pairs = max(1, int(len(pair_distances) * {frac_s}))
for _, l1, l2 in pair_distances[:num_similar_pairs]:
    {model_var}.add(open_station[l1] + open_station[l2] <= 1)"""
    block_lines = block.split("\n")

    head, body = _split_at_model(code)
    target = body if body is not None else code
    lines = target.split("\n")
    # Remove generated similarity logic before inserting the canonical block.
    trigger = re.compile(
        r"open_station\s*\[[^\]]+\]\s*\+\s*open_station\s*\[[^\]]+\]\s*<=\s*1"
        r"|\b_np\b|np\.linalg|pairwise|cost_sims?\b|max_sim\b|MAX_SIM\b"
        r"|pairs_to_prevent_opening|profile_dist|num_similar_pairs|pair_distances"
        r"|_cost_profiles|_distances|_threshold|_close_pairs"
        r"|import\s+numpy\s+as\s+_np"
    )

    def _bracket_delta(text: str) -> int:
        return (text.count("(") + text.count("[") + text.count("{")
                - text.count(")") - text.count("]") - text.count("}"))

    def _stmt_end(i: int) -> int:
        base_indent = len(re.match(r"(\s*)", lines[i]).group(1))
        end = i + 1
        depth = _bracket_delta(lines[i])
        while end < len(lines) and depth > 0:
            depth += _bracket_delta(lines[end])
            end += 1
        if lines[i].rstrip().endswith(":"):
            while end < len(lines):
                s = lines[end]
                if s.strip() == "" or len(re.match(r"(\s*)", s).group(1)) > base_indent:
                    end += 1
                else:
                    break
        return end

    ranges = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "":
            i += 1
            continue
        end = _stmt_end(i)
        block_text = "\n".join(lines[i:end])
        if trigger.search(block_text):
            ranges.append((i, end))
        i = end

    if not ranges:
        insert = next((i for i, line in enumerate(lines)
                       if re.match(r"\s*obj\s*=\s*", line)), None)
        if insert is None:
            return code, fixes
        target = "\n".join(lines[:insert] + block_lines + [""] + lines[insert:])
    else:
        rebuilt = []
        cursor = 0
        inserted = False
        for start, end in ranges:
            rebuilt.extend(lines[cursor:start])
            if not inserted:
                rebuilt.extend(block_lines)
                rebuilt.append("")
                inserted = True
            cursor = end
        rebuilt.extend(lines[cursor:])
        target = "\n".join(rebuilt)
    new_code = head + target if body is not None else target

    try:
        compile(new_code, "<charging-diversity-fix>", "exec")
    except SyntaxError:
        trig_idx = [i for i, ln in enumerate(lines) if trigger.search(ln)]
        if trig_idx:
            lo, hi = min(trig_idx), max(trig_idx)
            while hi + 1 < len(lines) and (
                    lines[hi + 1].startswith((" ", "\t")) or
                    trigger.search(lines[hi + 1]) or
                    "open_station[" in lines[hi + 1]):
                hi += 1
            rebuilt = lines[:lo] + block_lines + [""] + lines[hi + 1:]
            cand = head + "\n".join(rebuilt) if body is not None else "\n".join(rebuilt)
            try:
                compile(cand, "<charging-diversity-fix2>", "exec")
                new_code = cand
            except SyntaxError:
                return code, fixes
        else:
            return code, fixes
    if new_code != code:
        fixes.append("rewrote Charging geographic-diversity block as pure-Python nearest-similar-pair preprocessing")
    return new_code, fixes


# Small syntax repair used only when it makes the whole module parseable.
_OPENERS = {'(': ')', '[': ']', '{': '}'}
_CLOSERS = {')': '(', ']': '[', '}': '{'}


def _line_bracket_imbalance(line: str):
    """Return the stack of unclosed opening brackets in this physical line (in
    order), ignoring brackets inside strings/comments. Return None if there is an
    extra closing bracket (cannot be safely completed)."""
    stack = []
    i, n = 0, len(line)
    in_str = None
    while i < n:
        ch = line[i]
        if in_str:
            if ch == '\\':
                i += 2; continue
            if ch == in_str:
                in_str = None
            i += 1; continue
        if ch in ('"', "'"):
            in_str = ch
        elif ch == '#':
            break
        elif ch in _OPENERS:
            stack.append(ch)
        elif ch in _CLOSERS:
            if stack and stack[-1] == _CLOSERS[ch]:
                stack.pop()
            else:
                return None
        i += 1
    if in_str:
        return None
    return stack


def _repair_syntax(code: str, max_iter: int = 8) -> Tuple[str, bool]:
    """Complete "unclosed bracket" syntax errors. Only the physical line named by
    the parser is modified; closing brackets are added and the module re-parsed.
    Accepted only if the whole module goes from unparseable to parseable, otherwise
    the original code is returned."""
    import ast as _ast
    try:
        _ast.parse(code)
        return code, False
    except SyntaxError:
        pass

    work = code
    for _ in range(max_iter):
        try:
            _ast.parse(work)
            return work, (work != code)
        except SyntaxError as e:
            msg = (e.msg or "").lower()
            if "never closed" not in msg or not e.lineno:
                break
            lines = work.split('\n')
            li = e.lineno - 1
            if li < 0 or li >= len(lines):
                break
            stack = _line_bracket_imbalance(lines[li])
            if not stack:
                break
            lines[li] = lines[li] + ''.join(_OPENERS[c] for c in reversed(stack))
            new_work = '\n'.join(lines)
            if new_work == work:
                break
            work = new_work
    try:
        _ast.parse(work)
        return work, (work != code)
    except SyntaxError:
        return code, False


def _has_setup_intent(code: str, structured: Dict) -> bool:
    """Decide whether the code/problem involves a "sequence-dependent setup time"
    constraint.

    Loose detection: any trace of setup/transition/sequence_var in the generated
    code, or a mention of setup time in the problem's constraint descriptions, is
    taken as setup intent. This is broader than requiring both seq and transition,
    so it also covers the degenerate form where the model writes setup as a hand
    no_overlap([a,b]).
    """
    blob = code.lower()
    if any(k in blob for k in ("setup_time", "setup_times", "transition_matrix",
                               "transition_times", "sequence_var", "_seq")):
        return True
    for c in (structured.get("constraints", []) or []):
        d = (str(c.get("description", "")) + " " + str(c.get("type", ""))).lower()
        if "setup" in d or "sequence-dependent" in d or "changeover" in d:
            return True
    return False


def _has_invalid_setup_api(code: str) -> bool:
    """Detect docplex setup/no_overlap API shapes that are known to fail."""
    patterns = (
        r'\bsequence_var\s*\([^)]*\bsetup_time\s*=',
        r'\bsequence_var\s*\([^)]*\bsetup_times\s*=',
        r'\bno_overlap\s*\(\s*\w+\s*,\s*sequence_var\s*=',
        r'\bno_overlap\s*\(\s*ops\s*,',
        r'\btransition_matrix\s*=',
    )
    return any(re.search(pat, code or "", flags=re.DOTALL) for pat in patterns)


def _infer_setup_duration(code: str, structured: Dict) -> str:
    """Infer a scalar setup duration; default to the training-template value."""
    for pat in (
        r'\bsetup_time[s]?\s*=\s*\[\s*\(?\s*(\d+(?:\.\d+)?)',
        r'\bsetup_durations\b[^=]*=.*?\belse\s+(\d+(?:\.\d+)?)',
        r'\bsetup_trans\b.*?\b(\d+(?:\.\d+)?)',
    ):
        m = re.search(pat, code or "", flags=re.DOTALL)
        if m:
            return m.group(1)
    texts = []
    for c in (structured.get("constraints", []) or []):
        texts.append(str(c.get("description", "")))
        texts.append(str(c.get("type", "")))
    blob = " ".join(texts).lower()
    m = re.search(r'setup[^0-9]{0,40}(\d+(?:\.\d+)?)', blob)
    if m:
        return m.group(1)
    return "20"


# Canonical FJSP machine-level no_overlap blocks.
_FJSP_NOOVERLAP_PLAIN = """for lops in machine_operations:
    {mv}.add({mv}.no_overlap(lops))"""

_FJSP_NOOVERLAP_SETUP = """for m in range(NB_MACHINES):
    ops = machine_operations[m]
    if not ops:
        continue
    machine_seq = {mv}.sequence_var(
        ops,
        types=[job_number[op.get_name()] for op in ops],
        name="M{{}}_Seq".format(m))
    setup_durations = [[0 if j1 == j2 else {setup} for j2 in range(NB_JOBS)]
                       for j1 in range(NB_JOBS)]
    setup_trans = {mv}.transition_matrix(setup_durations)
    {mv}.add({mv}.no_overlap(machine_seq, setup_trans))"""


def _fix_docplex_setup_api(code: str, structured: Dict) -> Tuple[str, list]:
    """Rewrite FJSP machine-level no_overlap/setup blocks to valid docplex API."""
    fixes: list = []
    ptype = str(structured.get("problem_type", "")).lower()
    is_fjsp = ("aircraft" in ptype) or ("machine_operations" in code)
    if not is_fjsp:
        return code, fixes
    if "machine_operations" not in code:
        return code, fixes

    has_setup_intent = _has_setup_intent(code, structured)
    invalid_setup_api = _has_invalid_setup_api(code)

    # Keep valid sequence-dependent setup semantics intact.
    if has_setup_intent and not invalid_setup_api:
        return code, fixes

    model_var = _detect_model_var(code)
    lines = code.split('\n')

    def _is_machine_loop_head(i: int) -> bool:
        ln = lines[i]
        if not re.match(r'\s*for\s+', ln):
            return False
        head_uses_mops = 'machine_operations' in ln
        head_range_m   = bool(re.search(r'range\(\s*NB_MACHINES\s*\)', ln))
        if not (head_uses_mops or head_range_m):
            return False
        base_indent = len(re.match(r'(\s*)', ln).group(1))
        for j in range(i+1, min(i+15, len(lines))):
            s = lines[j]
            if s.strip() == "":
                continue
            cur_indent = len(re.match(r'(\s*)', s).group(1))
            if cur_indent <= base_indent:
                break
            if any(k in s for k in ('no_overlap', 'sequence_var', 'setup', 'transition')):
                return True
        return False

    head_i = next((i for i in range(len(lines)) if _is_machine_loop_head(i)), None)
    if head_i is None:
        return code, fixes

    indent = re.match(r'(\s*)', lines[head_i]).group(1)
    base_indent = len(indent)
    end = head_i + 1
    while end < len(lines):
        s = lines[end]
        if s.strip() == "" or len(re.match(r'(\s*)', s).group(1)) > base_indent:
            end += 1
        else:
            break

    use_setup = has_setup_intent
    tmpl = _FJSP_NOOVERLAP_SETUP if use_setup else _FJSP_NOOVERLAP_PLAIN
    setup_duration = _infer_setup_duration(code, structured)
    gold_block = tmpl.format(mv=model_var, setup=setup_duration)
    gold_lines = [indent + ln if ln else ln for ln in gold_block.split('\n')]

    _BLOCK_TRIGGER = re.compile(
        r'\.(no_overlap|sequence_var|transition_matrix)\s*\(|'
        r'\b(setup_time|setup_times|setup_durations|transition_times|'
        r'transition_matrix|machine_seq|setup_trans|setup_time_row|'
        r'setup_time_col)\b')

    def _block_should_drop(start: int, stop: int) -> bool:
        return any(_BLOCK_TRIGGER.search(lines[k]) for k in range(start, stop))

    drop = [False] * len(lines)
    for k in range(head_i, end):
        drop[k] = True

    i = 0
    n = len(lines)
    while i < n:
        if head_i <= i < end:
            i += 1; continue
        ln = lines[i]
        if ln.strip() == "":
            i += 1; continue
        cur_indent = len(re.match(r'(\s*)', ln).group(1))
        blk_start = i
        blk_stop = i + 1
        if ln.rstrip().endswith(":"):
            while blk_stop < n:
                s = lines[blk_stop]
                if s.strip() == "" or len(re.match(r'(\s*)', s).group(1)) > cur_indent:
                    blk_stop += 1
                else:
                    break
        if _block_should_drop(blk_start, blk_stop):
            for k in range(blk_start, blk_stop):
                drop[k] = True
            i = blk_stop
        else:
            i += 1

    rebuilt = []
    inserted = False
    for i in range(n):
        if head_i <= i < end:
            if not inserted:
                rebuilt.extend(gold_lines)
                inserted = True
            continue
        if drop[i]:
            continue
        rebuilt.append(lines[i])
    if not inserted:
        rebuilt = gold_lines + rebuilt
    new_code = "\n".join(rebuilt)

    try:
        compile(new_code, "<stage3-setupfix>", "exec")
    except SyntaxError:
        new_code2 = "\n".join(lines[:head_i] + gold_lines + lines[end:])
        try:
            compile(new_code2, "<stage3-setupfix>", "exec")
            new_code = new_code2
        except SyntaxError:
            return code, fixes

    if new_code != code:
        fixes.append(
            ("rewrote machine-level no_overlap as standard docplex setup-time form "
             "(sequence_var + transition_matrix)") if use_setup
            else "normalized machine-level no_overlap")
    return new_code, fixes


def _correct_known_constants(code: str, structured: Dict) -> Tuple[str, list]:
    """Correct a scalar value of the form `var = <literal>` or
    `var = [<literal>] * N` in the code to the definite value given by the formal
    expression. Only touches this simplest, safely identifiable scalar assignment
    to avoid collateral damage."""
    consts = _known_scalar_params(structured)
    if not consts:
        return code, []
    fixed = []
    lines = code.split("\n")
    for i, ln in enumerate(lines):
        for var, target in consts.items():
            # e.g.: var = 350 | var = [350] * NUM_PERIODS | var = [350]
            pat = re.compile(
                r'^(\s*' + re.escape(var) + r'\s*=\s*)'
                r'(\[\s*)?(-?\d+(?:\.\d+)?)(\s*\])?(\s*\*\s*\w+)?\s*$')
            m = pat.match(ln)
            if not m:
                continue
            cur = m.group(3)
            if cur == target:
                continue
            new_ln = (m.group(1) + (m.group(2) or "") + target +
                      (m.group(4) or "") + (m.group(5) or ""))
            lines[i] = new_ln
            fixed.append((var, cur, target))
            break
    return "\n".join(lines), fixed


def _parses(code: str) -> bool:
    """Return True iff `code` is parseable Python (used as the post-processing safety net)."""
    try:
        ast.parse(code or "")
        return True
    except SyntaxError:
        return False


def _undefined_names(code: str) -> set:
    """Lazy-load validator.undefined_names; return an empty set when unavailable (non-blocking)."""
    try:
        from pipeline.validator import undefined_names as _un
        return _un(code)
    except Exception:
        return set()


def _safe_apply(code: str, transform, label: str, verbose: bool) -> Tuple[str, bool]:
    """Apply a post-processing transform with "parse guard + semantic guard",
    solving the core problem of the original post-processing pipeline: a long chain
    of sequential regex rewrites with no safety net, where any step could break
    already-parseable/runnable code and still pass it downstream, only crashing at
    the solve stage (observed: the Charging geographic-diversity fix once inserted
    the correct block but left the bad `_np.array(...)` block in place -> repeated
    NameError).

    Rules (monotonically safe; every step may only "fix or preserve", never make
    the code worse):
      - transform raises          -> swallow, keep original code;
      - transform makes no change -> return as-is;
      - input parseable, output not -> deemed to break parseability, revert;
      - transform INTRODUCES a new undefined name (a NameError risk present in
        output but not input) -> revert;
      - otherwise                 -> accept.
    `transform(code)` returns `code` or `(code, info)`; info is the fix detail
    (list/truthy) used for printing. Returns (code, changed)."""
    before_ok = _parses(code)
    try:
        res = transform(code)
    except Exception as e:
        if verbose:
            print(f"  [{label}] skipped (post-processing error, original kept): {str(e)[:120]}")
        return code, False

    if isinstance(res, tuple):
        new_code = res[0]
        info = res[1] if len(res) > 1 else None
    else:
        new_code, info = res, None

    if not isinstance(new_code, str) or new_code == code:
        return code, False
    if before_ok and not _parses(new_code):
        if verbose:
            print(f"  [{label}] reverted (would break parseability, original kept)")
        return code, False
    # Reject transformations that introduce a new undefined name.
    if before_ok and _parses(new_code):
        new_undef = _undefined_names(new_code) - _undefined_names(code)
        if new_undef:
            if verbose:
                print(f"  [{label}] reverted (introduces new undefined name(s) {sorted(new_undef)},"
                      f" NameError; original kept)")
            return code, False

    if verbose:
        if isinstance(info, (list, tuple)) and info:
            for item in info:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    print(f"  [{label}] {item[0]}: {item[1]} → {item[2]}")
                else:
                    print(f"  [{label}] {item}")
        else:
            print(f"  [{label}] applied")
    return new_code, True


def run(
    structured: Dict,
    math_model: Dict,
    kb: KnowledgeBase,
    verbose: bool = True,
    feedback: str = "",
    current_code: str = "",
) -> str:
    is_fix = bool(feedback)
    print("\n" + "="*60)
    print(f"[Stage 3] CP Code Generation{'  [Fix Mode]' if is_fix else ''}")
    print("="*60)

    base_block = _base_template_block(structured, kb)

    if is_fix:
        prompt = f"""You are EDITING an existing docplex.cp program — NOT rewriting it from scratch.
Start from the EXACT code below and make the SMALLEST possible change that resolves
the feedback. Treat it like a patch/diff: keep every line that is not directly
implicated by the error.

ABSOLUTE RULES:
1. PRESERVE EVERY CONSTRAINT. Do NOT drop, merge, comment out, or simplify any
   constraint that is present in the code below OR listed in the math model — this
   includes time-window bounds, inter-job / cross-job precedence, capacity/budget
   bounds, and named global constraints (alternative / no_overlap / pack /
   sequence_var + transition_matrix setup). The fixed program must contain at LEAST
   as many `<model>.add(...)` constraints as the code below.
2. Keep the data-loading and overall structure intact (do NOT replace the file
   reader with hard-coded toy data).
3. If a construct is buggy, FIX it in place with the correct docplex API. Examples:
   - invalid `sequence_var(..., setup_time=...)` or `no_overlap(ops, sequence_var=...)`
     → `seq = mdl.sequence_var(ops, types=[...])`,
       `trans = mdl.transition_matrix(SETUP_MATRIX)`  (NEVER call
       `mdl.transition_matrix()` with no argument — it requires the matrix/size),
       `mdl.add(mdl.no_overlap(seq, trans))`.
   - never delete the constraint to make an error go away.
4. If the solver returned NO SOLUTION, relax ONLY the single conflicting bound the
   feedback identifies; do NOT remove other constraints to force a solution.

## Current code (edit THIS, keep all its constraints):
```python
{current_code}
```

## Feedback / issues to fix:
{feedback}

## Mathematical model (reference):
{json.dumps(math_model, indent=2)[:1800]}

{base_block}

{STYLE_RULES}

Return ONLY corrected Python code. No markdown. No comments."""
        raw = chat_for_stage(
            "feedback_fix",
            [{"role": "system", "content": system("code_fixer")},
             {"role": "user",   "content": prompt}],
            temperature=0.1,
        )

    else:
        if verbose:
            print(f"  Input to Stage-3 FT model — Math Model keys: "
                  f"{list(math_model.keys())}"
                  + ("  [+skeleton contract]" if FT_STAGE3_USE_SKELETON_CONTRACT else ""))

        mm_json = json.dumps(math_model, ensure_ascii=False)

        if FT_STAGE3_USE_SKELETON_CONTRACT:
            user_content = f"""{mm_json}

{base_block}

DATA / STRUCTURE CONTRACT (must follow):
- Reuse the reference template's data-loading and parsing EXACTLY (same input
  file, same parser, same data structures). Do NOT replace it with hard-coded
  toy data and do NOT change the shape of the parsed structures.
- Build the model on top of that skeleton, then add EVERY constraint and the
  objective from the math model above.
- Keep parsing and use structurally consistent: if each `job` element is a list
  of (machine, duration) choices, iterate `for op in job: for (m, d) in op:` —
  never `for (choices, ctype) in job` (that raises
  "cannot unpack non-iterable int object"). Per-operation flags such as
  tight/loose live in their OWN parallel list (e.g. CONSTRAINTS[jx][ox]).
- Output Python code only: no markdown, no comments, no prose."""
        else:
            user_content = mm_json

        ft_messages = [
            {"role": "system", "content": FT_STAGE3_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        api_messages = ft_messages

        raw = chat_ft_with_api_fallback(
            stage         = "code_generation",
            ft_messages   = ft_messages,
            api_messages  = api_messages,
            api_json_mode = False,
            temperature   = 0.0,
        )

    code = _strip_fences(raw)

    steps = []

    if FT_STAGE3_USE_GOLD_LOADER:
        ptype    = structured.get("problem_type", "")
        skeleton = structured.get("base_template") or (kb.base_template(ptype) if kb else "")
        loader   = _loader_prelude(skeleton)
        if loader:
            steps.append((lambda c, _ld=loader: _graft_loader(c, _ld),
                          "std-loader-graft"))

    if FT_STAGE3_POSTPROCESS_FIX:
        steps.append((_harden_int_token_loader, "loader-hardening"))
        steps.append((_repair_fjsp_loader_constraint_pollution, "FJSP-loader-fix"))
        steps.append((_fix_fjsp_unpack, "unpack-fix"))
        steps.append((_fix_membership_in_constraint, "membership-fix"))
        steps.append((lambda c: _fix_docplex_setup_api(c, structured), "api-fix"))
        steps.append((lambda c: _fix_charging_geographic_diversity(c, structured, math_model),
                      "geo-diversity-fix"))

    if FT_STAGE3_FIX_CONSTANTS:
        ptype = (structured.get("matched_domain")
                 or structured.get("problem_type") or "")
        ref = {}
        try:
            ref = (kb.formal_expression_ref(ptype) if kb else {}) or {}
        except Exception:
            ref = {}
        if not ref:
            ref = _FORMAL_REFS.get(ptype, {})
        const_source = {"parameters": list(structured.get("parameters", []) or [])
                                      + list(ref.get("parameters", []) or [])}
        steps.append((lambda c, _cs=const_source: _correct_known_constants(c, _cs), "constant-correction"))

    steps.append((_repair_syntax, "syntax-repair"))

    for transform, label in steps:
        code, _changed = _safe_apply(code, transform, label, verbose)

    if verbose:
        lines = code.split('\n')
        print(f"\n  Generated {len(lines)} lines of code"
              + ("" if _parses(code) else "  (still unparseable; left to feedback loop)") + ":")
        print("  " + "─"*56)
        for ln in lines[:80]:
            print(f"  {ln}")
        if len(lines) > 80:
            print(f"  ... [{len(lines)-80} more lines]")
        print("  " + "─"*56)
    return code
