from __future__ import annotations
import json, sys, os, re
from difflib import SequenceMatcher
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, Optional
from llm.client import chat_ft_with_api_fallback
from rag_fca.knowledge_base import KnowledgeBase
from config import FT_STAGE2_SYSTEM

# The 6 fields that form the Formal Expression (must match Excel training format exactly).
# Stage 1 output carries extra RAG-FCA metadata fields that are useful downstream
# (Stage 3 checklist, base template, etc.) but must NOT reach the Stage 2 model —
# they were absent from the training data and would shift the input distribution.
FORMAL_EXPR_KEYS = [
    "problem_type",
    "problem_summary",
    "decision_variables",
    "parameters",
    "constraints",
    "objective",
]


# ── constraintnormalize Stage-3 training distribution ────────────────────────
#
# fine-tuned Stage-3 math_expression generated code only
# math modelconstraintalreadyset code alreadyvalidate
# Stage-1/2 constraint e.g. "Geographic Separation"
# "similar(i,j) in top 10%" Stage-3 API
#
# here training set"eachproblemeachxconstraint" onlygenerate
# **anyalreadyvariant** alreadymatchvariantleave unchanged
# numeric value "training distribution" anycontent
_CANON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "data", "canonical_constraints.json")
try:
    with open(_CANON_PATH, encoding="utf-8") as _f:
        _CANON_CONSTRAINTS: Dict[str, Dict[str, list]] = json.load(_f)
except Exception as _e:
    print(f"  [Stage 2] ⚠ Could not load canonical_constraints.json ({_e}).")
    _CANON_CONSTRAINTS = {}


def _norm_expr(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def _denumber(s: str) -> str:
    """Numeric-free structural fingerprint: compares only the form skeleton, ignoring threshold/numeric differences."""
    return re.sub(r"-?\d+(?:\.\d+)?", "#", _norm_expr(s))


def _name_tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", str(s or "").lower()))


def _match_canonical_name(name: str, table: Dict[str, list]) -> Optional[str]:
    """Match a (possibly drifted) constraint name to the standard training name.

    Exact match first; otherwise fuzzy-match via token overlap + string
    similarity, e.g. 'Terminal GC Stability' -> 'Terminal GC Constraint'.
    """
    if name in table:
        return name
    nt = _name_tokens(name)
    if not nt:
        return None
    best, best_score = None, 0.0
    for cand in table:
        ct = _name_tokens(cand)
        # Jaccard
        jac = len(nt & ct) / len(nt | ct) if (nt | ct) else 0.0
        ratio = SequenceMatcher(None, name.lower(), cand.lower()).ratio()
        score = max(jac, ratio)
        # twoinfo orstring
        shared_informative = len((nt & ct) - {"constraint", "constraints"})
        if (shared_informative >= 2 or ratio >= 0.8) and score > best_score:
            best, best_score = cand, score
    return best if best_score >= 0.5 else None


def canonicalize_constraints(result: Dict, verbose: bool = True) -> Dict:
    """Pull constraint math_expressions whose STRUCTURE has truly drifted back to
    the standard training form.

    Safe policy (avoid breaking already-correct constraints):
      1. if the generated expression == any known training form for this type,
         leave it unchanged (it is a valid training form).
      2. otherwise match it by name to a standard constraint; replace only when
         the generated form also differs from the standard on its numeric-free
         structural skeleton (not merely threshold values). A pure numeric
         difference (e.g. time window 500-600 vs 100-700) is a valid variant and
         is left unchanged.
    """
    ptype = result.get("model_title", "")
    table = _CANON_CONSTRAINTS.get(ptype)
    if not table:
        return result

    # typeallalready & numeric value
    known_exact, known_struct = set(), set()
    for variants in table.values():
        for v in variants:
            known_exact.add(_norm_expr(v))
            known_struct.add(_denumber(v))

    fixed = []
    for c in result.get("constraints_section", []) or []:
        cur = str(c.get("math_expression", ""))
        if _norm_expr(cur) in known_exact:
            continue  # alreadyvalid → leave unchanged
        name = str(c.get("name", "")).strip()
        canon_name = _match_canonical_name(name, table)
        if not canon_name:
            continue
        canonical = table[canon_name][0]
        # onlynumeric value → validnumeric valuevariant → generate
        if _denumber(cur) == _denumber(canonical):
            continue
        # butskeletontypexalreadyconstraintskeleton
        # onlynumeric value/variant leave unchanged
        if _denumber(cur) in known_struct:
            continue
        c["math_expression"] = canonical  # →
        if name != canon_name:
            c["name"] = canon_name
        fixed.append((name, cur[:45], canon_name, canonical[:45]))

    if fixed and verbose:
        for old_name, old_expr, new_name, new_expr in fixed:
            arrow = f"{old_name} → {new_name}" if old_name != new_name else new_name
            print(f"  [constraint-normalize] {arrow}: '{old_expr}...' -> '{new_expr}...'")
    return result


def _to_formal_expression(structured: Dict) -> Dict:
    """
    Extract the 6-field Formal Expression from the Stage-1 structured dict.

    Format mirrors the 'Formal Expression' column in CP_weitiao140_with_formal_expression.xlsx:
      problem_type       : str
      problem_summary    : str
      decision_variables : list[{name, type, description, domain}]
      parameters         : list[{name, description, value_or_source}]
      constraints        : list[{id, type, description}]
      objective          : {direction, expression, description}
    """
    return {k: structured[k] for k in FORMAL_EXPR_KEYS if k in structured}


def run(structured: Dict, kb: KnowledgeBase, verbose: bool = True) -> Dict:
    print("\n" + "="*60)
    print("[Stage 2] Mathematical Model Generation")
    print("="*60)

    # ── Build the 6-field Formal Expression (Stage-2 model input) ──────────────
    formal_expr = _to_formal_expression(structured)

    if verbose:
        print(f"  Input to Stage-2 model — Formal Expression ({len(formal_expr)} fields):")
        for k in FORMAL_EXPR_KEYS:
            v = formal_expr.get(k)
            if isinstance(v, list):
                print(f"    {k}: [{len(v)} item(s)]")
            elif isinstance(v, dict):
                print(f"    {k}: direction={v.get('direction','')}  "
                      f"expression={str(v.get('expression',''))[:50]}")
            else:
                print(f"    {k}: {str(v)[:70]}")

    # ── Both FT model and API fallback receive identical messages ───────────────
    # system : FT_STAGE2_SYSTEM  (the system prompt the FT model was trained with)
    # user   : Formal Expression JSON string (6 fields, no extra metadata)
    messages = [
        {"role": "system", "content": FT_STAGE2_SYSTEM},
        {"role": "user",   "content": json.dumps(formal_expr, ensure_ascii=False)},
    ]

    raw = chat_ft_with_api_fallback(
        stage         = "math_modeling",
        ft_messages   = messages,
        api_messages  = messages,   # same clean format for API fallback
        api_json_mode = True,       # API needs this flag to guarantee JSON output
    )

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(m.group()) if m else {"raw_response": raw}

    # constraintdistribution ensure Stage-3 =
    result = canonicalize_constraints(result, verbose=verbose)

    if verbose:
        print(f"  Model       : {result.get('model_title')}")
        print(f"  Type        : {result.get('model_type')} | "
              f"{result.get('complexity_note','')[:60]}")
        print(f"  Parameters  : {len(result.get('parameters_section', []))}")
        print(f"  Variables   : {len(result.get('variables_section', []))}")
        for v in result.get("variables_section", []):
            print(f"    {v.get('symbol')} [{v.get('var_type')}] — "
                  f"{v.get('description','')[:60]}")
        print(f"  Constraints : {len(result.get('constraints_section', []))}")
        for c in result.get("constraints_section", []):
            print(f"    [{c.get('id')}] {c.get('name')} : "
                  f"{c.get('math_expression','')[:70]}")
        obj = result.get("objective_section", {})
        print(f"  Objective   : {obj.get('direction','').upper()} "
              f"{obj.get('math_expression','')[:70]}")
    return result
