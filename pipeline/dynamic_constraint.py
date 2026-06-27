from __future__ import annotations
import json, re, sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, List, Tuple
from llm.client import chat_for_stage, system
from rag_fca.knowledge_base import KnowledgeBase
from config import TOP_K_RETRIEVAL

CONSTRAINT_ID_MAP = {
    "time_window": "1", "temporal": "1", "window": "1", "deadline": "1",
    "capacity":    "2",
    "resource":    "3", "machine": "3",
    "precedence":  "4", "sequence": "4",
    "global":      "5", "all_diff": "5", "gc_content": "5",
    "dynamic":     "6", "fault":    "6", "reconfigur": "6",
}


def _constraint_id(nl_desc: str) -> str:
    desc_lower = nl_desc.lower()
    for kw, cid in CONSTRAINT_ID_MAP.items():
        if kw in desc_lower:
            return cid
    return "new"


def _new_label(existing_label: str, new_id: str) -> str:
    if existing_label in ("base", "", "nan"):
        return new_id
    parts = [p.strip() for p in existing_label.split(",") if p.strip()]
    if new_id not in parts and new_id != "new":
        parts.append(new_id)
    return ",".join(sorted(set(parts), key=lambda x: int(x) if x.isdigit() else 99))


# ─────────────────────────────────────────────────────────────
# Deterministic code-level helpers
# ─────────────────────────────────────────────────────────────

def _find_objective_line(lines: List[str]) -> int:
    """Index of the objective line. Recognizes both the wrapped form
    `mdl.add(mdl.minimize(...))` and the bare form `model.minimize(...)` /
    `mdl.maximize(...)`, for either the `mdl.` or `model.` naming."""
    import re
    # wrapped: x.add(x.minimize( / x.add(x.maximize(
    wrapped = re.compile(r'\b(?:mdl|model)\.add\(\s*(?:mdl|model)\.(?:minimize|maximize)\b')
    # bare:    x.minimize( / x.maximize(   (not already inside an add(...))
    bare = re.compile(r'\b(?:mdl|model)\.(?:minimize|maximize)\s*\(')
    # first pass: the wrapped form is the most explicit objective marker
    for i, ln in enumerate(lines):
        if wrapped.search(ln):
            return i
    # second pass: a bare minimize/maximize call
    for i, ln in enumerate(lines):
        if bare.search(ln):
            return i
    return -1


def _insert_before_objective(existing_code: str, new_lines: List[str]) -> str:
    """
    Inject new_lines into existing_code immediately before the objective line.
    existing_code is preserved exactly; only the new lines are inserted before
    the minimize/maximize call. The injected lines take the indentation of the
    objective line so they sit at the right scope.
    """
    lines   = existing_code.split("\n")
    obj_idx = _find_objective_line(lines)

    if obj_idx == -1:
        # Fallback: insert before solve()
        for i, ln in enumerate(lines):
            if ".solve(" in ln:
                obj_idx = i
                break
        if obj_idx == -1:
            print("  [DynConstraint] ⚠  Could not locate objective or solve() line — "
                  "appending new constraint lines at end of code.")
            return existing_code + "\n" + "\n".join(new_lines)

    # Match the indentation of the line we insert before, so the new lines sit
    # at the same scope (top level for a top-level objective).
    anchor = lines[obj_idx]
    indent = anchor[:len(anchor) - len(anchor.lstrip())]
    injected = [indent + ln if ln.strip() else ln for ln in new_lines]

    result = lines[:obj_idx] + injected + lines[obj_idx:]
    return "\n".join(result)


def _job_index_note(nl_desc: str) -> str:
    mentioned = sorted(set(int(n) for n in
                           re.findall(r'\bJob\s+(\d+)\b', nl_desc, re.IGNORECASE)))
    lines = [
        "Python 0-based job indexing:",
        "  'Job 1' = job_operations[0]",
        "  'Job 2' = job_operations[1]",
        "  'Job 3' = job_operations[2]",
        "  General: 'Job N' = job_operations[N-1]",
    ]
    if mentioned:
        for jn in mentioned:
            lines.append(f"  --> Job {jn} = job_operations[{jn-1}]")
    return "\n".join(lines)


def _clean_new_lines(raw: str) -> List[str]:
    """
    Extract only executable mdl.add(...) lines from LLM output.
    Strips markdown, comments, explanations.
    """
    # Remove code fences
    raw = re.sub(r'```(?:python)?', '', raw)
    raw = raw.replace('```', '')

    result = []
    for ln in raw.splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        # Skip import lines, variable definitions, comments, print statements
        if (stripped.startswith('#') or
                stripped.startswith('import ') or
                stripped.startswith('from ') or
                stripped.startswith('print(') or
                stripped.startswith('msol') or
                stripped.startswith('if msol') or
                stripped.startswith('else:')):
            continue
        # Accept lines that are clearly constraint additions or for/if wrappers.
        # Use `stripped` for all checks so indentation level doesn't matter.
        if ('mdl.add(' in stripped or 'model.add(' in stripped or
                stripped.startswith('for ') or
                stripped.startswith('if ')):
            result.append(ln)

    return result


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def adapt_constraint(
    new_constraint_nl: str,
    existing_code: str,
    existing_structured: Dict,
    existing_math_model: Dict,
    kb: KnowledgeBase,
    verbose: bool = True,
) -> Tuple[str, Dict, Dict]:
    """
    Localized model update: inject new constraint into existing code.
    Returns (updated_code, updated_structured, updated_math_model).
    """
    print("\n" + "="*60)
    print("[Dynamic Adaptation] Adding New Constraint")
    print("="*60)
    print(f"  New constraint: {new_constraint_nl[:120]}")

    # Step 1: concept-lattice RAG — formal template(s) for the new constraint
    rag = kb.rag_fca_retrieve(
        new_constraint_nl + " " + existing_structured.get("problem_type", ""),
        top_k=TOP_K_RETRIEVAL,
    )
    tmpls = rag["formal_templates"]
    examples_text = "\n".join(
        f"- {t['name']}: {t['math']}  |  cp: {t['cp']}" for t in tmpls[:6]
    ) or "(none)"

    # Step 2: Ask LLM ONLY for the new constraint lines
    new_c_id     = f"C{len(existing_structured.get('constraints', [])) + 1}"
    index_note   = _job_index_note(new_constraint_nl)
    ptype        = existing_structured.get("problem_type", "")

    # Show LLM the relevant variable names from existing code
    var_lines = [ln.strip() for ln in existing_code.splitlines()
                 if "job_operations" in ln or "interval_var" in ln.lower()
                 or "all_operations" in ln or "machine_operations" in ln][:6]
    var_context = "\n".join(var_lines) or "(see code)"

    prompt = f"""You are adding ONE new constraint to an existing docplex.cp model.

Problem type: {ptype}
New constraint: {new_constraint_nl}

{index_note}

Relevant variables already defined in the code:
{var_context}

Docplex syntax reference:
{examples_text}

TASK: Return ONLY the Python lines (mdl.add statements and any surrounding for/if loops)
that implement the new constraint. Do NOT return the full program.
Do NOT include imports, model definition, existing constraints, objective, or solve call.
Use job_operations[N-1] for 'Job N' per the index table above.
No comments. No markdown.

Example output format:
for op in job_operations[1]:
    mdl.add(mdl.start_of(op) >= 10)
    mdl.add(mdl.end_of(op) <= 700)
"""

    try:
        raw = chat_for_stage(
            "dynamic_constraint",
            [{"role": "system", "content":
                "You output ONLY new Python constraint lines for a docplex.cp model. "
                "No full program. No markdown. No comments. No imports. "
                "Only mdl.add() statements and their for/if wrappers."},
             {"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except RuntimeError as e:
        # Network / API failure: do NOT crash the session — keep the existing
        # code unchanged and report the issue so the user can retry.
        print(f"  ⚠  Could not reach the model to add the constraint: {str(e)[:120]}")
        print("     Existing code is unchanged. Check your network/API and try option [2] again.")
        return existing_code, existing_structured, existing_math_model

    new_lines = _clean_new_lines(raw)

    if not new_lines:
        # Fallback: try to extract any mdl.add line from raw
        fallback = [ln for ln in raw.splitlines() if "mdl.add(" in ln or "model.add(" in ln]
        new_lines = fallback if fallback else [f"# Could not generate constraint for: {new_constraint_nl[:60]}"]

    # Step 3: Deterministic injection into existing code
    updated_code = _insert_before_objective(existing_code, new_lines)

    # Step 4: Formal expression
    raw_formal = chat_for_stage(
        "nl_structuring",
        [{"role": "system", "content": system("math_modeler")},
         {"role": "user", "content": (
             f"Express this constraint formally using CPLEX math notation.\n"
             f"Problem type: {ptype}\n"
             f"Constraint: {new_constraint_nl}\n"
             f"Respond in JSON: {{\"id\":\"{new_c_id}\",\"type\":\"\","
             f"\"description\":\"\",\"formal_expression\":\"\",\"docplex_expression\":\"\"}}"
         )}],
        json_mode=True, temperature=0.1,
    )
    try:
        new_c_formal = json.loads(raw_formal)
    except Exception:
        new_c_formal = {
            "id": new_c_id, "type": "new",
            "description": new_constraint_nl[:120],
            "formal_expression": "", "docplex_expression": "\n".join(new_lines),
        }

    # Step 5: Update structured + math model dicts
    updated_struct = copy.deepcopy(existing_structured)
    updated_struct.setdefault("constraints", []).append({
        "id":                new_c_formal.get("id", new_c_id),
        "type":              new_c_formal.get("type", "new"),
        "description":       new_c_formal.get("description", new_constraint_nl[:120]),
        "formal_expression": new_c_formal.get("formal_expression", ""),
    })

    updated_model = copy.deepcopy(existing_math_model)
    updated_model.setdefault("constraints_section", []).append({
        "id":                 new_c_formal.get("id", new_c_id),
        "name":               new_c_formal.get("type", "new constraint"),
        "math_expression":    new_c_formal.get("formal_expression", ""),
        "docplex_expression": new_c_formal.get("docplex_expression", "\n".join(new_lines)),
        "description":        new_c_formal.get("description", new_constraint_nl[:120]),
    })

    if verbose:
        print(f"\n  Formal     : {new_c_formal.get('formal_expression','')[:80]}")
        print(f"\n  Injected lines ({len(new_lines)}):")
        for ln in new_lines:
            print(f"    + {ln}")

        lines_up = updated_code.split('\n')
        obj_idx  = _find_objective_line(lines_up)
        print(f"\n  Total lines: {len(lines_up)}")
        if obj_idx != -1:
            # Verify injected lines sit before objective
            for ln in new_lines:
                idx = next((i for i, l in enumerate(lines_up) if l == ln), -1)
                if idx != -1:
                    pos = "BEFORE objective ✓" if idx < obj_idx else "AFTER objective ⚠"
                    print(f"  Line {idx+1}: {pos}")

    return updated_code, updated_struct, updated_model