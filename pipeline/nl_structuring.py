from __future__ import annotations
import json, sys, os, re, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, List
from llm.client import chat_for_stage, system
from rag_fca.knowledge_base import KnowledgeBase
from config import TOP_K_RETRIEVAL, STAGE1_MODE

# Fields that constitute the Formal Expression (Stage 2 model input format)
FORMAL_EXPR_KEYS = [
    "problem_type",
    "problem_summary",
    "decision_variables",
    "parameters",
    "constraints",
    "objective",
]

# ── Canonical formal-expression references (one per problem type) ─────────────
# Extracted from CP_weitiao140_with_formal_expression.xlsx (the FT training data).
# Used as few-shot examples so Stage 1 output matches training conventions.
_REFS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "data", "formal_expression_refs.json")
try:
    with open(_REFS_PATH, encoding="utf-8") as _f:
        _FORMAL_REFS: Dict[str, Dict] = json.load(_f)
except Exception as _e:
    print(f"  [Stage 1] ⚠ Could not load formal_expression_refs.json ({_e}); "
          f"falling back to format-only guidance.")
    _FORMAL_REFS = {}

# ── Controlled constraint-'type' vocabulary + canonical parameter order ───────
# Extracted from the FT training set (CP_weitiao140_with_formal_expression.xlsx).
# The fine-tuned Stage-2 model keys on these EXACT constraint 'type' keywords;
# if Stage 1 invents a new keyword (e.g. 'module_usage_balancing') or picks a
# wrong generic one (e.g. 'time_window' for a switching constraint), the FT model
# receives an out-of-distribution constraint type and mis-models it.
_VOCAB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "data", "constraint_type_vocab.json")
try:
    with open(_VOCAB_PATH, encoding="utf-8") as _f:
        _TYPE_VOCAB: Dict[str, Dict] = json.load(_f)
except Exception as _e:
    print(f"  [Stage 1] ⚠ Could not load constraint_type_vocab.json ({_e}).")
    _TYPE_VOCAB = {}


_TYPE_SIGNATURES: Dict[str, List] = {
    "VRP": [("vehicle", 3), ("depot", 3), ("routing", 3), ("subtour", 3),
            ("truck", 3), ("fleet", 2), ("route", 2), ("customer", 2),
            ("arc", 2), ("deliver", 1), ("logistics", 1), ("warehouse", 1),
            ("dispatch", 1), ("visited exactly once", 2), ("travel distance", 2)],
    "Aircraft Skin Processing": [("aircraft", 3), ("skin", 3), ("job shop", 3),
            ("makespan", 3), ("processing time", 2), ("operation", 1),
            ("machine", 1), ("precedence", 1)],
    "Battery Pack Design": [("battery", 3), ("module", 2), ("voltage", 2),
            ("series column", 2), ("cell", 1), ("pack", 1), ("current", 1)],
    "Charging Station Location": [("charging", 2), ("demand area", 3),
            ("facility location", 3), ("station", 2), ("facility", 1)],
    "DNA Sequence Design": [("dna", 3), ("nucleotide", 3), ("gc content", 3),
            ("guanine", 2), ("cytosine", 2), ("hamming", 2), ("sequence", 1)],
}


def _detect_problem_type(nl_text: str) -> str:
    """Detect the problem type from NL keywords; return only on a clear win,
    otherwise return ''.

    The main decision is delegated to fca.detect_scenario_strong (the per-domain
    strong keywords of the 5 domains do not overlap and are unambiguous). Key
    point: cross-domain generic words like 'vehicle' are NOT a strong keyword of
    any domain, so charging-station and battery-pack problems that mention
    'electric vehicle(s)' are no longer misclassified as VRP. If no strong keyword
    matches, a weighted-signature pass adds a conservative decision (return '' when
    uncertain, deferring to RAG-FCA).
    """
    low = (nl_text or "").lower()

    # Prefer domain-specific strong keywords before the weighted fallback.
    from rag_fca.fca import detect_scenario_strong
    strong = detect_scenario_strong(low)
    if strong:
        return strong

    # 2) otherwise
    scores = {t: sum(w for kw, w in sig if kw in low)
              for t, sig in _TYPE_SIGNATURES.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] < 3:
        return ""  # No domain has enough evidence.
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 2:
        return ""  # The top scores are too close to disambiguate.
    return ranked[0][0]


def _vocab_block(matched_domain: str) -> str:
    """Hard constraint block: restrict constraint 'type' to the training vocabulary and give the parameter order."""
    spec = _TYPE_VOCAB.get(matched_domain)
    if not spec:
        return ""
    types = ", ".join(spec.get("constraint_types", []))
    porder = ", ".join(spec.get("param_order", []))
    lines = [
        f"## ★ CONTROLLED VOCABULARY for '{matched_domain}' (training set — MANDATORY) ★",
        "The fine-tuned Stage-2 model was trained ONLY on these constraint 'type' keywords.",
        f"  Allowed constraint 'type' values (use EXACTLY one of these, verbatim): [{types}]",
        "  • For EVERY constraint — including any extra/variant constraint described in the",
        "    problem — set 'type' to the SINGLE closest keyword from the list above.",
        "  • NEVER invent a new 'type' name (e.g. do NOT output 'module_usage_balancing',",
        "    'spatial_block_activation_limit', etc.). Map it to the closest allowed keyword",
        "    (e.g. a usage/most-vs-least balance → 'load_balance'; a per-period state-change",
        "    limit → 'switching'; a spatial/neighbourhood activation limit → 'locality').",
        "  • Do NOT relabel a constraint as a generic 'time_window' unless it truly is one.",
    ]
    if porder:
        lines.append(f"  Parameters: list them in EXACTLY this order: [{porder}]")
    return "\n".join(lines)


def _all_refs_overview_block(refs: Dict[str, Dict]) -> str:
    """
    Compact one-look reference of ALL base problem types' canonical conventions.
    Always shown so the model picks the right house-style even when the RAG
    domain match is imperfect or empty (mitigates Stage-1 distribution shift).
    """
    if not refs:
        return ""
    lines = [
        "## ★ CANONICAL FORMAL-EXPRESSION STYLE — base problem types (from knowledge base) ★",
        "The fine-tuned downstream models were trained on these EXACT conventions.",
        "Find the row matching this problem's type and reproduce its variable-naming,",
        "constraint 'type' vocabulary and objective format — only adapt the VALUES:",
    ]
    for ptype, ref in refs.items():
        var_names  = ", ".join(v.get("name", "") for v in ref.get("decision_variables", []))
        cons_types = ", ".join(c.get("type", "") for c in ref.get("constraints", []))
        obj_expr   = ref.get("objective", {}).get("expression", "")
        lines.append(f"  - [{ptype}]")
        lines.append(f"      decision_variable names : {var_names}")
        lines.append(f"      constraint 'type' words : {cons_types}")
        lines.append(f"      objective expression    : {obj_expr}")
    return "\n".join(lines)


def _canonical_example_block(matched_domain: str, ref: Dict = None) -> str:
    """
    Build a few-shot block showing the canonical formal expression for the
    matched problem type. This pins the LLM to the EXACT training conventions
    (variable naming, constraint 'type' vocabulary, granularity, objective format).

    `ref` is the gold Formal Expression (preferably sourced from the knowledge
    base); falls back to the bundled refs file when not supplied.
    """
    if not ref:
        ref = _FORMAL_REFS.get(matched_domain)
    if not ref:
        return ""

    var_names    = ", ".join(v.get("name", "") for v in ref.get("decision_variables", []))
    cons_types   = ", ".join(c.get("type", "") for c in ref.get("constraints", []))
    obj_expr     = ref.get("objective", {}).get("expression", "")

    return f"""## ★ CANONICAL FORMAL EXPRESSION for '{matched_domain}' — FOLLOW THIS FORMAT EXACTLY ★
The fine-tuned downstream model expects this EXACT style. You MUST reproduce its
conventions (only adapt the numeric/textual VALUES to the new problem):
  - decision_variable NAMING convention : {var_names}
  - constraint 'type' VOCABULARY (use these exact keywords): {cons_types}
  - constraint GRANULARITY: keep each constraint SEPARATE (e.g. tight and loose
    temporal constraints are TWO distinct entries, not merged into one)
  - objective 'expression' FORMAT: {obj_expr}

Reference example (reproduce this structure and conventions):
```json
{json.dumps(ref, ensure_ascii=False, indent=2)}
```
"""


def _concept_block(rag: Dict) -> str:
    """Render the concept-lattice retrieval as STRUCTURE (matched concepts and
    their constraint intents), not natural-language paragraphs."""
    lines = []
    dom = rag.get("matched_domain", "")
    if dom:
        lines.append(f"Matched domain (by concept-lattice similarity, Eq.15): {dom}")
    for i, c in enumerate(rag.get("matched_concepts", []), 1):
        lines.append(
            f"[Concept {i}] type={c['problem_type']}  score={c['score']}  "
            f"covers {c['extent_size']} problems (e.g. ids {c['example_ids']})")
        lines.append(f"    constraint intent B: {{{', '.join(c['constraints'])}}}")
    return "\n".join(lines) if lines else "(no matching concept)"


def _implicit_hint(certain: List[str], possible: List[str]) -> str:
    """Build a hint block about implicit constraints for the LLM prompt."""
    lines = []
    if certain:
        lines.append("## Certain implicit constraints (auto-detected by RAG-FCA, Eq.22):")
        lines.append("These constraints are present in ALL similar problems — include them.")
        for c in certain[:8]:
            lines.append(f"  - {c}")
    if possible:
        lines.append("## Possible implicit constraints (rough-set suggestions, Eq.21):")
        lines.append("These constraints appear in SOME similar problems — consider them.")
        for c in possible[:8]:
            lines.append(f"  - {c}")
    return "\n".join(lines) if lines else ""


def _attach_rag_metadata(result: Dict, rag: Dict, certain_implicit: List[str],
                         possible_implicit: List[str], matched_domain: str) -> Dict:
    """Attach FCA-retrieved metadata onto the structured result (shared by both Stage-1 modes)."""
    result["certain_implicit"]     = certain_implicit
    result["possible_implicit"]    = possible_implicit
    result["implicit_constraints"] = certain_implicit + possible_implicit
    result["formal_templates"]     = rag["formal_templates"]
    result["matched_domain"]       = matched_domain
    result["base_template"]        = rag["base_template"]
    result["_rag_concepts"] = [{"problem_type": c["problem_type"],
                                "constraints": c["constraints"],
                                "example_ids": c["example_ids"]}
                               for c in rag["matched_concepts"]]
    return result


_CONSTRAINT_CUE_RE = re.compile(
    r"\b(constraint|constraints|required|require|requires|must|shall|should|"
    r"not exceed|no more than|at most|at least|within|between|limited|limit|"
    r"enforced|imposed|specified|controlled)\b", re.IGNORECASE)

_NON_MODEL_SENTENCE_RE = re.compile(
    r"\b(write a python program|docplex|cp optimizer|input data|file format|"
    r"read from|located in|first two integers|followed by|finally)\b", re.IGNORECASE)


def _constraint_sentences(nl_text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", nl_text or "")
    return [p.strip() for p in parts if p and _CONSTRAINT_CUE_RE.search(p)]


def _covered_by_known_constraint(sentence: str, problem_type: str) -> bool:
    """Whether this constraint sentence is already covered by the domain catalog (base templates / optional regex)."""
    if _NON_MODEL_SENTENCE_RE.search(sentence):
        return True
    try:
        from rag_fca import constraints as _con
    except Exception:
        return False
    spec = _con.CATALOG.get(problem_type, {})
    for opt in spec.get("optional", []):
        if re.search(opt.get("regex", ""), sentence, flags=re.IGNORECASE):
            return True
    low = sentence.lower()
    base_hints = {
        "Charging Station Location": (
            "strategically deployed", "predicted charging demand", "fully served",
            "service coverage", "maximum service capacity", "candidate station",
            "optimal subset", "activate as charging stations"),
        "Aircraft Skin Processing": (
            "machine can process only one operation", "predefined sequence",
            "temporal constraints between consecutive operations",
            "temporal rules between consecutive operations"),
        "Battery Pack Design": (
            "dynamically configured", "same in every series column",
            "parallel structure remains balanced", "voltage", "current",
            "active modules", "energy loss"),
        "DNA Sequence Design": (
            "stringent biochemical constraints", "design objective",
            "all the above biochemical", "word list", "gc content",
            "hamming distance", "reverse complement"),
        "VRP": ("visited exactly once", "depot", "route flow", "capacity",
                "subtour", "flow conservation"),
    }
    return any(h in low for h in base_hints.get(problem_type, ()))


def _novel_constraint_candidates(nl_text: str, problem_type: str) -> List[str]:
    return [s for s in _constraint_sentences(nl_text)
            if not _covered_by_known_constraint(s, problem_type)]


def _dedup_constraints(cons: List[Dict], verbose: bool = False) -> List[Dict]:
    """Deterministic de-duplication: remove semantically equivalent duplicate
    constraints (the same constraint stated twice). Equality is decided by a
    normalized signature: same type + same numeric sequence + same key indices in
    the description. Keeps the first occurrence. API-free first line of defense
    so duplicates never reach Stage-2/3."""
    def _sig(c: Dict) -> tuple:
        t = str(c.get("type", "")).strip().lower()
        d = str(c.get("description", "")).lower()
        nums = tuple(re.findall(r"\d+(?:\.\d+)?", d))
        # Include entity indices so constraints on different entities stay distinct.
        idx = tuple(sorted(set(re.findall(
            r"\b(?:job|operation|op|customer|node|station|module|period|column|row)\s*\d+", d))))
        return (t, nums, idx)

    seen, out, dropped = {}, [], 0
    for c in cons:
        s = _sig(c)
        if s in seen:
            dropped += 1
            continue
        seen[s] = True
        out.append(c)
    if dropped and verbose:
        print(f"  [Stage 1 self-check] de-duplicated {dropped} semantically duplicate constraint(s)")
    return out


def _verify_formal_against_nl(result: Dict, nl_text: str, matched_domain: str,
                              verbose: bool = True) -> Dict:
    """Stage-1 API self-check: after deterministic compose, call the API to verify
    the KB-mapped constraint descriptions against the original NL, and fix three
    issue classes:
      (1) numeric mismatch  -- a number in a constraint description / parameter
          value disagrees with the NL (budget bound, utilization %, similarity %,
          capacity, time-window edge, ...); fixed to the NL value;
      (2) missing constraint -- required by the NL but absent (added with a
          controlled type keyword);
      (3) extra constraint   -- present but not asked for by the NL (removed).

    Discipline (keeps the alignment fixes intact): the API only REPORTS diffs; the
    framework applies them DETERMINISTICALLY -- numeric fixes only rewrite the
    number in the description, deletions only remove API-named ids, additions use
    controlled vocab; out-of-vocab types are dropped. Fail-safe: on API error or
    empty audit the composed result is kept. Gated by STAGE1_VERIFY (default on)."""
    try:
        from config import STAGE1_VERIFY
    except Exception:
        STAGE1_VERIFY = True
    if not STAGE1_VERIFY:
        return result

    vocab = _TYPE_VOCAB.get(matched_domain, {}).get("constraint_types", [])
    cons_text = json.dumps([{k: c.get(k) for k in ("id", "type", "description")}
                            for c in result.get("constraints", [])],
                           ensure_ascii=False, indent=2)
    param_text = json.dumps([{k: p.get(k) for k in ("name", "value_or_source")}
                             for p in result.get("parameters", [])],
                            ensure_ascii=False, indent=2)
    prompt = f"""You are auditing a formal model that was assembled from a knowledge base
against the ORIGINAL natural-language (NL) problem. The base structure is correct;
your ONLY job is to catch where the assembled constraints/parameters DIVERGE from
the NL — specifically: wrong numbers, a missing constraint, an extra constraint, or
the SAME constraint written twice (duplicates).

Problem type: {matched_domain}

Original NL problem:
{nl_text}

Assembled constraints (id, type, description):
{cons_text}

Assembled parameters (name, value_or_source):
{param_text}

Allowed constraint 'type' vocabulary (use ONLY these for any addition):
{json.dumps(vocab, ensure_ascii=False)}

Check four things and return STRICT JSON:
{{
  "numeric_fixes": [
    {{"id": "<constraint id whose description has a wrong number>",
      "description": "<the SAME constraint, description rewritten with the NL-correct number(s)>"}}
  ],
  "param_fixes": [
    {{"name": "<parameter name>", "value_or_source": "<NL-correct value>"}}
  ],
  "missing_constraints": [
    {{"type": "<closest allowed type>", "description": "<constraint clearly required by the NL but absent above, incl. its exact numbers>"}}
  ],
  "extra_constraint_ids": [ "<id of a constraint present above but NOT supported by the NL>" ],
  "duplicate_constraint_ids": [ "<id of a constraint that is a DUPLICATE of another one above (same meaning / same bound on the same object); list the id(s) to REMOVE, keeping one>" ]
}}

Rules:
- Compare NUMBERS carefully: bounds, percentages, capacities, budgets, counts,
  time-window edges, thresholds. If the NL says 40 / 45000 / 95% / 5% but the
  description says 30 / 30000 / 90% / 10%, that is a numeric_fix.
- DUPLICATES: if two constraints express the SAME restriction on the SAME object
  (e.g. two "Job 1 starts in [20,100]" entries, or two "Job 4 after Job 3" entries
  even if one says 'last operation' and the other 'second operation'), list the
  redundant id(s) in duplicate_constraint_ids so only ONE remains. If the two
  differ in their numbers/objects they are NOT duplicates.
- Only report a missing_constraint if the NL UNAMBIGUOUSLY states it and it is not
  already represented (by any type) above. Use the closest ALLOWED type.
- Only report an extra_constraint_id if the NL clearly does NOT ask for it.
- Do NOT rename types, do NOT touch correct constraints, do NOT paraphrase for
  style. Return empty lists when everything matches. Output ONLY the JSON."""

    try:
        raw = chat_for_stage(
            "nl_structuring",
            [{"role": "system", "content": system("formal_verifier")},
             {"role": "user", "content": prompt}],
            json_mode=True)
        audit = json.loads(raw)
    except Exception as e:
        if verbose:
            print(f"  [Stage 1] self-check skipped (API error, keeping composed result): {str(e)[:120]}")
        return result
    if not isinstance(audit, dict):
        return result

    allowed = set(vocab)
    cons = list(result.get("constraints", []))
    by_id = {c.get("id"): c for c in cons}
    n_num = n_param = n_add = n_del = 0

    for fx in (audit.get("numeric_fixes") or []):
        if not isinstance(fx, dict):
            continue
        cid, desc = fx.get("id"), str(fx.get("description", "")).strip()
        if cid in by_id and desc:
            if by_id[cid].get("description") != desc:
                by_id[cid]["description"] = desc
                n_num += 1

    params = list(result.get("parameters", []))
    pnames = {p.get("name") for p in params}
    for fx in (audit.get("param_fixes") or []):
        if not isinstance(fx, dict):
            continue
        nm, val = fx.get("name"), fx.get("value_or_source")
        if nm in pnames and val not in (None, ""):
            for p in params:
                if p.get("name") == nm and str(p.get("value_or_source")) != str(val):
                    p["value_or_source"] = str(val)
                    n_param += 1
    result["parameters"] = params

    extra_ids = set(audit.get("extra_constraint_ids") or [])
    if extra_ids:
        kept = [c for c in cons if c.get("id") not in extra_ids]
        n_del = len(cons) - len(kept)
        cons = kept

    dup_ids = set(audit.get("duplicate_constraint_ids") or [])
    n_dup = 0
    if dup_ids:
        kept = [c for c in cons if c.get("id") not in dup_ids]
        n_dup = len(cons) - len(kept)
        cons = kept

    # (2) missing constraint -> add with controlled vocab only.
    # Discipline: for the 5 KNOWN catalog types, _compose_formal_expression already
    # emitted the complete base + detected-optional constraint set. Re-adding an LLM
    # "missing" constraint here would duplicate a composed one with different wording
    # (which the signature de-dup can't catch) and over-constrain the model. So skip
    # the missing-constraint injection for known types; still apply it for genuinely
    # unknown problems. (Numeric fixes, param fixes and extra/duplicate removal above
    # remain active for all types — they only correct or prune, never duplicate.)
    try:
        from rag_fca import constraints as _con
        _is_known = matched_domain in set(_con.CATALOG.keys())
    except Exception:
        _is_known = False
    if not _is_known:
        for mc in (audit.get("missing_constraints") or []):
            if not isinstance(mc, dict):
                continue
            ctype = str(mc.get("type", "")).strip()
            desc = str(mc.get("description", "")).strip()
            if not desc:
                continue
            if allowed and ctype not in allowed:
                if verbose:
                    print(f"  [Stage 1 self-check] dropped out-of-vocab added constraint type='{ctype}'")
                continue
            cons.append({"id": "", "type": ctype, "description": desc})
            n_add += 1

    # (3c) Remove duplicates introduced by API verification or fallback logic.
    before = len(cons)
    cons = _dedup_constraints(cons, verbose=verbose)
    n_dup += before - len(cons)

    # Renumber constraints contiguously after edits.
    for i, c in enumerate(cons, start=1):
        c["id"] = f"C{i}"
    result["constraints"] = cons

    if verbose:
        if n_num or n_param or n_add or n_del or n_dup:
            print(f"  [Stage 1 self-check] NL audit: {n_num} numeric fix(es), {n_param} "
                  f"param fix(es), +{n_add} missing, -{n_del} extra, -{n_dup} duplicate")
        else:
            print(f"  [Stage 1 self-check] NL audit: constraints and values consistent, no change")
    return result


def _append_novel_constraints_if_any(result: Dict, nl_text: str,
                                     matched_domain: str, verbose: bool = True) -> Dict:
    """Hybrid fallback: known variants are handled by deterministic compose; only
    if the NL contains a genuinely new constraint sentence not covered by the
    catalog do we call the LLM once to return just the uncovered constraints,
    appended with the closest controlled type. Standard variants of the 5 known
    types do not trigger this, staying in-distribution."""
    novel_candidates = _novel_constraint_candidates(nl_text, matched_domain)
    if not novel_candidates:
        return result

    # Discipline: for the 5 KNOWN catalog types, _compose_formal_expression already
    # produced the COMPLETE, in-distribution constraint set (anchor + base + detected
    # optional variants, with real numbers). The novel-constraint LLM escape-hatch is
    # ONLY for genuinely unknown problem types. Letting it fire on a known type is
    # actively harmful: lower-abstraction phrasings (L1/L2) contain a semi-structured
    # "Constraints: A, B, C ..." preamble whose clauses don't match the coverage
    # hints, so the LLM re-emits them as extra constraints that DUPLICATE the composed
    # ones (e.g. charging: a second activation_load / pack / capacity), producing a
    # self-contradictory, over-constrained model that comes back no_solution. The
    # signature de-dup cannot catch them because the wording/indices differ. So we
    # skip the escape-hatch entirely for catalog types and trust the deterministic
    # composition, which is what gave the good results on earlier runs.
    try:
        from rag_fca import constraints as _con
        _known_types = set(_con.CATALOG.keys())
    except Exception:
        _known_types = set()
    if matched_domain in _known_types:
        return result

    vocab = _TYPE_VOCAB.get(matched_domain, {}).get("constraint_types", [])
    known_text = json.dumps(result.get("constraints", []), ensure_ascii=False, indent=2)
    prompt = f"""You are checking whether an optimization problem contains constraints
NOT already covered by the known formal expression.

Problem type: {matched_domain}

Known/detected constraints already covered:
{known_text}

Allowed type vocabulary for this problem type:
{json.dumps(vocab, ensure_ascii=False)}

Natural-language problem:
{nl_text}

Candidate sentences that appear not covered:
{json.dumps(novel_candidates, ensure_ascii=False, indent=2)}

Return strict JSON:
{{
  "novel_constraints": [
    {{"type": "closest allowed type; if none fits use custom_constraint",
      "description": "precise formal meaning incl. numeric values, indices, bounds"}}
  ],
  "parameters": [ {{"name": "", "description": "", "value_or_source": ""}} ]
}}

Rules:
- Return an empty novel_constraints list if every constraint is already covered.
- Do not repeat or paraphrase known constraints.
- Prefer the closest ALLOWED type so the downstream fine-tuned model stays in its
  training vocabulary; use custom_constraint only when no allowed type fits."""

    try:
        raw = chat_for_stage(
            "nl_structuring",
            [{"role": "system", "content": system("nl_structurer")},
             {"role": "user", "content": prompt}],
            json_mode=True)
        extra = json.loads(raw)
    except Exception as e:
        if verbose:
            print(f"  [Stage 1] novel-constraint check skipped: {str(e)[:120]}")
        return result

    novel = extra.get("novel_constraints", [])
    if not isinstance(novel, list) or not novel:
        return result

    constraints = list(result.get("constraints", []))
    next_id = len(constraints) + 1
    for item in novel:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description", "")).strip()
        if not desc:
            continue
        ctype = str(item.get("type", "") or "custom_constraint").strip()
        constraints.append({"id": f"C{next_id}", "type": ctype, "description": desc})
        next_id += 1
    result["constraints"] = constraints
    result["_novel_constraints"] = novel

    # Parameter handling here MUST obey the same distribution-alignment discipline
    # as _compose_formal_expression: NEVER add a non-gold parameter name. The LLM
    # novel-check, when fed a semi-structured "Parameters: ..." preamble (as in the
    # lower-abstraction L1/L2 phrasings), tends to echo it back as a parallel set of
    # renamed/duplicate params (num_demand_areas, service_cost, fixed_cost, ...).
    # Merging those in pollutes the Stage-2 input with names the fine-tuned model
    # never saw, producing broken code. So we ONLY override the value of a parameter
    # name that already exists; any other ("novel") parameter is dropped.
    params = list(result.get("parameters", []))
    existing_names = {p.get("name") for p in params}
    for p in extra.get("parameters", []) or []:
        if isinstance(p, dict) and p.get("name") in existing_names:
            _merge_parameter(params, p)   # Existing gold parameter: safe value override.
    result["parameters"] = params
    if verbose:
        print(f"  [Stage 1] appended {len(novel)} novel constraint(s) beyond the catalog (LLM fallback)")
    return result


OPTIONAL_FORMAL_CONSTRAINTS: Dict[str, Dict[int, List[Dict[str, str]]]] = {
    "Aircraft Skin Processing": {
        1: [{"type": "time_window",
             "description": "Job 1 must start between time 20 and time 100."}],
        2: [{"type": "time_window",
             "description": "All operations of Job 2 must start after time 10."},
            {"type": "time_window",
             "description": "All operations of Job 2 must end before time 500."}],
        3: [{"type": "precedence",
             "description": "Inter-job precedence: all operations of Job 3 start only after all operations of Job 2 complete."}],
        4: [{"type": "precedence",
             "description": "Inter-job precedence: first operation of Job 4 starts after completion of second operation of Job 3."}],
        5: [{"type": "time_window",
             "description": "Operation-level temporal constraint: end time of first operation of Job 5 is in [500, 600]."}],
        6: [{"type": "sequence",
             "description": "Sequence-dependent setup time of 20 units when the same machine switches between different jobs."}],
    },
    "Battery Pack Design": {
        1: [{"type": "load_balance",
             "description": "Difference between most-used and least-used modules across all periods is at most 2."}],
        2: [{"type": "switching",
             "description": "Between consecutive periods, no more than 6 module switch states may change."}],
        3: [{"type": "locality",
             "description": "In every 2x2 block of neighboring module positions, at most 2 modules may be active in the same period."}],
        4: [{"type": "degradation",
             "description": "Module at row 1 and column 1 is faulty and must remain offline in every period."}],
    },
    "Charging Station Location": {
        1: [{"type": "scale",
             "description": "Problem scale is expanded from 30 to 40 demand areas."}],
        2: [{"type": "cost_cap",
             "description": "Transportation cost from each demand area to its assigned station must not exceed 20 units."}],
        3: [{"type": "fixed_cost_bound",
             "description": "Sum of fixed costs of open stations must be in [20000, 30000]."}],
        4: [{"type": "utilization",
             "description": "Operational load of every opened station must not exceed 90% of design capacity."}],
        5: [{"type": "geographic_spread",
             "description": "At most one of any two sites in the nearest 10% of cost-similar pairs may be selected."}],
    },
    "DNA Sequence Design": {
        1: [{"type": "end_gc",
             "description": "Each DNA sequence must end with C or G for terminal stability."}],
        2: [{"type": "global_balance",
             "description": "Across all generated sequences, the total counts of any two bases differ by at most 2."}],
        3: [{"type": "no_repeat_pair",
             "description": "No dinucleotide pattern may repeat in overlapping positions within a sequence."}],
        4: [{"type": "forbidden_pattern",
             "description": "The forbidden motif ACTG must not appear in any generated sequence."}],
    },
    "VRP": {
        2: [{"type": "time_window",
             "description": "Ready time, due time, and service time are enforced for each customer visit."}],
        4: [{"type": "exclusion",
             "description": "A specified customer node is excluded from the delivery network."}],
    },
}

OPTIONAL_PARAMETER_OVERRIDES: Dict[str, Dict[int, List[Dict[str, str]]]] = {
    "Charging Station Location": {
        1: [{"name": "nbDemandAreas", "description": "Number of demand areas.",
             "value_or_source": "40"}],
        2: [{"name": "MAX_ALLOWED_COST", "description": "Maximum service cost per assigned demand area.",
             "value_or_source": "20"}],
        3: [{"name": "MIN_FIXED_COST", "description": "Minimum fixed-cost budget for opened stations.",
             "value_or_source": "20000"},
            {"name": "MAX_FIXED_COST", "description": "Maximum fixed-cost budget for opened stations.",
             "value_or_source": "30000"}],
        4: [{"name": "MAX_UTILIZATION_RATIO", "description": "Maximum allowed station utilization ratio.",
             "value_or_source": "0.9"}],
        5: [{"name": "SIMILARITY_PERCENTILE", "description": "Nearest cost-profile percentile used for site-diversity pairs.",
             "value_or_source": "0.10"}],
    },
    "VRP": {
        3: [{"name": "vehicle_capacity", "description": "Capacity per vehicle.",
             "value_or_source": "180"}],
        4: [{"name": "node_to_skip", "description": "Customer node excluded from service.",
             "value_or_source": "5"}],
    },
}


def _infer_optional_ids(nl_text: str, problem_type: str) -> List[int]:
    """Use the catalog regex signatures to deterministically infer which optional constraint ids this variant enables from the NL."""
    try:
        from rag_fca import constraints as _con
    except Exception:
        return []
    spec = _con.CATALOG.get(problem_type, {})
    ids: List[int] = []
    for opt in spec.get("optional", []):
        oid = opt.get("id")
        if oid is None:
            continue
        if re.search(opt.get("regex", ""), nl_text or "", flags=re.IGNORECASE):
            ids.append(int(oid))
    return sorted(set(ids))


def _relaxed_base_names(nl_text: str, problem_type: str) -> set:
    """Return the set of base constraint names this variant should REMOVE (CATALOG.relax matched and NL matches)."""
    try:
        from rag_fca import constraints as _con
    except Exception:
        return set()
    spec = _con.CATALOG.get(problem_type, {})
    out = set()
    for opt in spec.get("optional", []):
        relax = opt.get("relax")
        if relax and re.search(opt.get("regex", ""), nl_text or "", flags=re.IGNORECASE):
            out.add(relax)
    return out


_BASE_NAME_TO_GOLD_TYPE = {
    "vehicle_capacity": "capacity",
}


def _merge_parameter(params: List[Dict], override: Dict) -> None:
    """Replace a gold parameter value by name (only value_or_source/description); append if absent."""
    name = override.get("name", "")
    for p in params:
        if p.get("name") == name:
            p.update(override)
            return
    params.append(copy.deepcopy(override))


def _num_token(s: str) -> str:
    return str(s or "").replace(",", "").strip()


def _dynamic_parameter_overrides(nl_text: str, problem_type: str,
                                 optional_ids: List[int]) -> List[Dict[str, str]]:
    """Regex-extract the per-row scalar real values that distinguish variants (override template defaults)."""
    text = nl_text or ""
    low = text.lower()
    out: List[Dict[str, str]] = []

    if problem_type == "Charging Station Location":
        if 1 in optional_ids:
            m = re.search(r"(?:number of )?demand areas (?:is |are )?(?:increased|expanded)[^.]*", text, re.I)
            nums = re.findall(r"\d+", m.group(0)) if m else []
            out.append({"name": "nbDemandAreas", "description": "Number of demand areas.",
                        "value_or_source": _num_token(nums[-1]) if nums else "40"})
        if 2 in optional_ids:
            m = re.search(r"transportation cost.*?(?:not exceed|no more than|at most)\s*([\d,]+)", text, re.I)
            out.append({"name": "MAX_ALLOWED_COST", "description": "Maximum service cost per assigned demand area.",
                        "value_or_source": _num_token(m.group(1)) if m else "20"})
        if 3 in optional_ids:
            m = re.search(r"budget range of\s*([\d,]+)\s*(?:to|-|and)\s*([\d,]+)", text, re.I) \
                or re.search(r"(?:between|within)\s*([\d,]+)\s*(?:to|-|and)\s*([\d,]+).*?budget", text, re.I)
            lo, hi = (_num_token(m.group(1)), _num_token(m.group(2))) if m else ("20000", "30000")
            out += [{"name": "MIN_FIXED_COST", "description": "Minimum fixed-cost budget for opened stations.", "value_or_source": lo},
                    {"name": "MAX_FIXED_COST", "description": "Maximum fixed-cost budget for opened stations.", "value_or_source": hi}]
        if 4 in optional_ids:
            m = re.search(r"(?:actual load capacity|operational load|load).*?(?:not exceed|no more than|at most)\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
            pct = float(m.group(1)) / 100.0 if m else 0.90
            out.append({"name": "MAX_UTILIZATION_RATIO", "description": "Maximum allowed station utilization ratio.",
                        "value_or_source": f"{pct:g}"})
        if 5 in optional_ids:
            m = re.search(r"(?:nearest|top|ranking in the nearest)\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
            pct = float(m.group(1)) / 100.0 if m else 0.10
            out.append({"name": "SIMILARITY_PERCENTILE", "description": "Nearest cost-profile percentile used for site-diversity pairs.",
                        "value_or_source": f"{pct:g}"})

    if problem_type == "VRP":
        # Extract the vehicle capacity only when it is stated in the NL text.
        m = re.search(r"capacity(?: value)?(?: of| is| set to)?\s*(\d+)", low, re.I)
        if m:
            out.append({"name": "vehicle_capacity", "description": "Capacity per vehicle.",
                        "value_or_source": _num_token(m.group(1))})
        if 4 in optional_ids:
            m = re.search(r"(?:node|location|customer)\s*(\d+).*?(?:excluded|skipped)|(?:excluded|skipped).*?(?:node|location|customer)\s*(\d+)", low, re.I)
            val = _num_token(next((g for g in m.groups() if g), "5")) if m else "5"
            out.append({"name": "node_to_skip", "description": "Customer node excluded from service.",
                        "value_or_source": val})

    return out


def _variant_constraint_descriptions(nl_text: str, problem_type: str,
                                     optional_ids: List[int]) -> Dict[int, List[str]]:
    """Generate each optional constraint's description from NL-extracted real
    numbers (overriding the template default). Returns {optional_id: [desc, ...]}
    in the same order as OPTIONAL_FORMAL_CONSTRAINTS. Only customizes constraints
    whose numbers may vary per variant; others keep the template default."""
    text, low = nl_text or "", (nl_text or "").lower()
    out: Dict[int, List[str]] = {}

    if problem_type == "Charging Station Location":
        if 1 in optional_ids:
            m = re.search(r"(?:from\s*(\d+)\s*to\s*(\d+))", text)
            frm, to = (m.group(1), m.group(2)) if m else ("30", "40")
            out[1] = [f"Problem scale is expanded from {frm} to {to} demand areas."]
        if 2 in optional_ids:
            m = re.search(r"transportation cost[^.]*?(?:not exceed|no more than|at most)\s*([\d,]+)", text, re.I)
            v = (m.group(1).replace(",", "") if m else "20")
            out[2] = [f"Transportation cost from each demand area to its assigned station must not exceed {v} units."]
        if 3 in optional_ids:
            m = re.search(r"budget range of\s*([\d,]+)\s*(?:to|-|and)\s*([\d,]+)", text, re.I) \
                or re.search(r"(?:between|within)\s*([\d,]+)\s*(?:to|-|and)\s*([\d,]+)", text, re.I)
            lo, hi = (m.group(1).replace(",", ""), m.group(2).replace(",", "")) if m else ("20000", "30000")
            out[3] = [f"Sum of fixed costs of open stations must be in [{lo}, {hi}]."]
        if 4 in optional_ids:
            m = re.search(r"(?:not exceed|no more than|at most)\s*(\d+(?:\.\d+)?)\s*%\s*of[^.]*(?:design )?capacity", text, re.I)
            pct = m.group(1) if m else "90"
            out[4] = [f"Operational load of every opened station must not exceed {pct}% of design capacity."]
        if 5 in optional_ids:
            m = re.search(r"nearest\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
            pct = m.group(1) if m else "10"
            out[5] = [f"At most one of any two sites in the nearest {pct}% of cost-similar pairs may be selected."]

    elif problem_type == "VRP":
        if 4 in optional_ids:
            m = re.search(r"(?:node|location|customer)\s*(\d+)[^.]*(?:excluded|skipped)|(?:excluded|skipped)[^.]*(?:node|location|customer)\s*(\d+)", low, re.I)
            node = (next((g for g in m.groups() if g), "5") if m else "5")
            out[4] = [f"Customer node {node} is excluded from the delivery network."]

    elif problem_type == "Aircraft Skin Processing":
        if 6 in optional_ids:
            m = re.search(r"setup time of\s*(\d+)\s*units", text, re.I)
            v = m.group(1) if m else "20"
            out[6] = [f"Sequence-dependent setup time of {v} units when the same machine switches between different jobs."]

    return out


def _strip_relaxed_phrases(summary: str, relaxed: set) -> str:
    """When a variant relaxes/removes a base constraint, strip the conflicting
    wording from the summary so the summary stays consistent with the constraint
    list. E.g. after VRP capacity is relaxed, the summary should no longer say
    'respecting capacity constraints', otherwise the fine-tuned model would re-add
    the capacity constraint from the summary -> no-solution that the loop cannot
    fix."""
    if not summary or not relaxed:
        return summary
    s = summary
    phrase_map = {
        "vehicle_capacity": [
            r",?\s*respecting (?:vehicle )?capacity constraints?",
            r",?\s*subject to (?:vehicle )?capacity (?:constraints?|limits?)",
            r",?\s*within (?:vehicle )?capacity",
            r",?\s*(?:while )?(?:not exceeding|honou?ring) (?:vehicle )?capacity",
        ],
    }
    for name in relaxed:
        for pat in phrase_map.get(name, []):
            s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*,\s*,", ",", s)
    s = re.sub(r"\s*,\s*\.", ".", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _compose_formal_expression(ref: Dict, nl_text: str, problem_type: str,
                               verbose: bool = False) -> Dict:
    """Deterministically compose the 6-field Formal Expression that Stage-2 expects.

      The base shape is verbatim from gold; optional ids append controlled-template
      constraints and parameter overrides; a CATALOG.relax-matched variant REMOVES
      the corresponding base constraint (no out-of-vocab type); per-row real numbers
      extracted by regex override the template defaults.
    """
    result = copy.deepcopy({k: ref[k] for k in FORMAL_EXPR_KEYS if k in ref})
    optional_ids = _infer_optional_ids(nl_text, problem_type)
    relaxed      = _relaxed_base_names(nl_text, problem_type)
    gold_param_names = {p.get("name") for p in result.get("parameters", [])}

    if relaxed:
        drop_types = {_BASE_NAME_TO_GOLD_TYPE.get(n) for n in relaxed} - {None}
        if drop_types:
            kept = [c for c in result.get("constraints", []) if c.get("type") not in drop_types]
            if verbose and len(kept) != len(result.get("constraints", [])):
                print(f"  [compose] variant relaxation -> removed base constraint type={sorted(drop_types)}")
            result["constraints"] = kept

    dyn_desc = _variant_constraint_descriptions(nl_text, problem_type, optional_ids)
    constraints = list(result.get("constraints", []))
    for oid in optional_ids:
        templates = OPTIONAL_FORMAL_CONSTRAINTS.get(problem_type, {}).get(oid, [])
        custom    = dyn_desc.get(oid)
        for j, item in enumerate(templates):
            desc = item["description"]
            if custom and j < len(custom) and custom[j]:
                desc = custom[j]  # Prefer the NL-derived description when available.
            constraints.append({"id": "", "type": item["type"], "description": desc})
    result["constraints"] = constraints

    params = list(result.get("parameters", []))
    def _override_existing(ov):
        nm = ov.get("name")
        if nm in gold_param_names:
            _merge_parameter(params, ov)  # Override only an existing gold parameter.
        elif verbose:
            print(f"  [compose] skipped non-gold parameter '{nm}' (its value lives in the "
                  f"constraint description; avoids distribution drift)")
    for oid in optional_ids:
        for ov in OPTIONAL_PARAMETER_OVERRIDES.get(problem_type, {}).get(oid, []):
            _override_existing(ov)
    for ov in _dynamic_parameter_overrides(nl_text, problem_type, optional_ids):
        _override_existing(ov)
    result["parameters"] = params

    if relaxed:
        for p in result.get("parameters", []):
            if p.get("name") in relaxed:
                d = p.get("description", "").rstrip(".")
                if "reference only" not in d.lower():
                    p["description"] = d + " (reference only; NOT enforced as a constraint)."

    # Renumber constraints contiguously after applying the variant.
    for i, c in enumerate(result.get("constraints", []), start=1):
        c["id"] = f"C{i}"

    if relaxed:
        result["problem_summary"] = _strip_relaxed_phrases(
            result.get("problem_summary", ""), relaxed)
    # Keep the label as metadata; it is not part of the six Stage-2 fields.
    result["constraint_label"] = (",".join(str(i) for i in optional_ids)
                                  if (optional_ids or relaxed) else "base")
    return result


def _variant_diff_prompt(nl_text: str, gold: Dict, matched_domain: str) -> str:
    """Prompt: make the LLM output only the variant's DIFF relative to the gold
    base, not a from-scratch rewrite of the whole formal expression.

    This is key to aligning Stage-1 -> Stage-2. Asking the LLM to emit the whole
    Formal Expression drifts the distribution even with few-shot/controlled vocab
    (writing type 'capacity' as 'capacity_relaxed', rewriting summary/description,
    dropping value_or_source, etc.); the fine-tuned Stage-2 model has only seen the
    gold distribution, so any input drift breaks downstream.

    Solution: the base is left untouched, verbatim from gold; the model only
    decides the variant's minimal diff (param-value override / constraint-desc
    override / appended constraint / removed constraint / objective override).
    _apply_variant_diff then deterministically reapplies it onto the gold base, so
    the output stays in-distribution."""
    allowed_types = []
    spec = _TYPE_VOCAB.get(matched_domain) or {}
    allowed_types = spec.get("constraint_types", [])
    return f"""You are given a CANONICAL base Formal Expression (the exact training-distribution
form) for a '{matched_domain}' problem, plus a NEW problem description that is a
VARIANT of that base. Do NOT rewrite the whole Formal Expression. Instead, output
ONLY the MINIMAL DIFF the variant introduces relative to the base.

## Canonical base Formal Expression (the reference distribution — do NOT restate it):
```json
{json.dumps(gold, ensure_ascii=False, indent=2)}
```

## New problem description (a variant of the base):
{nl_text}

## How to express the diff (strict JSON, ONLY these keys; use [] / {{}} when empty):
{{
  "param_value_overrides": {{ "<parameter name exactly as in base>": "<new value_or_source string>" }},
  "constraint_desc_overrides": {{ "<constraint id, e.g. C4>": "<new description>" }},
  "append_constraints": [ {{"id": "C<n>", "type": "<one of allowed types>", "description": "<...>"}} ],
  "remove_constraint_ids": [ "<constraint id to drop, e.g. C4>" ],
  "objective_override": {{ "direction": "<minimize|maximize|feasibility|''>", "expression": "<or ''>", "description": "<or ''>" }}
}}

## RULES:
  1. Keep the base's variable names, base constraint 'type' keywords, parameter
     order, and objective format UNCHANGED — only emit what the variant changes.
  2. For any appended/variant constraint, 'type' MUST be one of the allowed
     keywords for this problem type: {allowed_types}. Never invent a new keyword.
  3. If the variant only RELAXES/REMOVES a base constraint (e.g. "capacity is not
     enforced"), put that constraint's id in "remove_constraint_ids" — do NOT
     rename its 'type' to something like 'capacity_relaxed'.
  4. If the variant changes a numeric value (e.g. capacity 200→180, areas 30→40),
     put it in "param_value_overrides" keyed by the exact base parameter name.
  5. If nothing changes for a section, return an empty object/list for it.
  6. Output ONLY the JSON diff, nothing else."""


def _apply_variant_diff(gold: Dict, diff: Dict, verbose: bool = False) -> Dict:
    """Deterministically reapply the LLM's variant diff onto the gold base,
    producing an in-distribution Formal Expression.

    Accepts only controlled minimal edits; anything that cannot be aligned to the
    base is safely ignored, never introducing out-of-distribution
    fields/keywords downstream."""
    result = copy.deepcopy(gold)
    diff = diff or {}

    # 1) Override values for parameters that already exist in the base.
    pov = diff.get("param_value_overrides") or {}
    if isinstance(pov, dict):
        for p in result.get("parameters", []):
            nm = p.get("name")
            if nm in pov and pov[nm]:
                p["value_or_source"] = str(pov[nm])

    # 2) Match constraint IDs and replace descriptions without changing their types.
    cdo = diff.get("constraint_desc_overrides") or {}
    if isinstance(cdo, dict):
        for c in result.get("constraints", []):
            cid = c.get("id")
            if cid in cdo and cdo[cid]:
                c["description"] = str(cdo[cid])

    # 3) Remove base constraints requested by the variant.
    rm = set(diff.get("remove_constraint_ids") or [])
    if rm:
        result["constraints"] = [c for c in result.get("constraints", [])
                                 if c.get("id") not in rm]

    # 4) Append only constraints whose types are in the controlled vocabulary.
    allowed = set()
    spec = _TYPE_VOCAB.get(result.get("problem_type", "")) or {}
    allowed = set(spec.get("constraint_types", []))
    for nc in (diff.get("append_constraints") or []):
        if not isinstance(nc, dict):
            continue
        ctype = str(nc.get("type", "")).strip()
        if allowed and ctype not in allowed:
            if verbose:
                print(f"  [variant-align] dropped out-of-vocab constraint type='{ctype}'")
            continue
        result.setdefault("constraints", []).append({
            "id": nc.get("id") or f"C{len(result['constraints'])+1}",
            "type": ctype,
            "description": str(nc.get("description", "")),
        })

    # 5) Apply an optional objective override.
    oo = diff.get("objective_override") or {}
    if isinstance(oo, dict):
        obj = result.setdefault("objective", {})
        for k in ("direction", "expression", "description"):
            if oo.get(k):
                obj[k] = oo[k]

    # Renumber constraints contiguously after applying the diff.
    for i, c in enumerate(result.get("constraints", []), start=1):
        c["id"] = f"C{i}"
    return result


def _print_formal_expression(result: Dict) -> None:
    """Stage-1 display: print the Formal Expression (the Stage-2 input) only."""
    print("\n  ---- Formal Expression (Stage-2 input) ----")
    print(f"  Problem type : {result.get('problem_type')}")
    print(f"  Summary      : {result.get('problem_summary', '')}")
    dvs = result.get("decision_variables", [])
    print(f"  Decision Variables ({len(dvs)}):")
    for v in dvs:
        print(f"    {v.get('name','')} [{v.get('type','')}]: {v.get('description','')}")
    ps = result.get("parameters", [])
    print(f"  Parameters ({len(ps)}):")
    for p in ps:
        print(f"    {p.get('name','')}: {p.get('value_or_source','')}  — {p.get('description','')}")
    cs = result.get("constraints", [])
    print(f"  Constraints ({len(cs)}):")
    for c in cs:
        print(f"    [{c.get('id','')}] {c.get('type','')}: {c.get('description','')}")
    obj = result.get("objective", {})
    print(f"  Objective    : {str(obj.get('direction','')).upper()} {obj.get('expression','')}")


def run(nl_text: str, kb: KnowledgeBase, verbose: bool = True) -> Dict:
    print("\n" + "="*60)
    print("[Stage 1] Natural Language Structuring"
          f"{'  [KB mode — no LLM]' if STAGE1_MODE == 'kb' else ''}")
    print("="*60)

    rag = kb.rag_fca_retrieve(nl_text, top_k=TOP_K_RETRIEVAL)

    certain_implicit  = rag["certain_constraints"]
    possible_implicit = rag["possible_constraints"]
    matched_domain    = rag.get("matched_domain", "")

    # RAG-FCA localizes by constraint overlap only and can mislabel the type (e.g.
    # VRP -> job-shop). Correct it with an NL keyword detector: when the detector
    # clearly disagrees with FCA, trust the detector and RE-RETRIEVE so the whole
    # chain locks onto the correct type. Otherwise changing only matched_domain is
    # not enough: the certain/possible constraints, formal_templates, base_template
    # and matched_concepts would still come from the wrong domain and pull the
    # output back. With a problem_type_filter re-retrieval, all metadata comes from
    # the correct type.
    detected_domain = _detect_problem_type(nl_text)
    if detected_domain and detected_domain != matched_domain:
        rag = kb.rag_fca_retrieve(nl_text, top_k=TOP_K_RETRIEVAL,
                                  problem_type_filter=detected_domain)
        certain_implicit  = rag["certain_constraints"]
        possible_implicit = rag["possible_constraints"]
        matched_domain    = rag.get("matched_domain", detected_domain)

    implicit_hint  = _implicit_hint(certain_implicit, possible_implicit)

    # Canonical Formal-Expression references now live in the knowledge base; fall
    # back to the bundled refs file if the KB has none loaded.
    kb_refs = {}
    try:
        kb_refs = kb.all_formal_expression_refs() or {}
    except Exception:
        kb_refs = {}
    if not kb_refs:
        kb_refs = _FORMAL_REFS

    matched_ref     = kb_refs.get(matched_domain, {})

    # ── KB mode: no LLM; use the KB's canonical Formal Expression for this type ──
    #   Applies when the problem is one of the 5 known types with a stored 6-field
    #   form. That form is verbatim from the training set, identical in distribution
    #   to the Stage-2 fine-tuned input, and needs no cloud API.
    if STAGE1_MODE == "kb":
        if matched_ref:
            result = copy.deepcopy(matched_ref)
            # Keep only the 6 standard fields; no extra keys leak into Stage-2 input
            result = {k: result[k] for k in FORMAL_EXPR_KEYS if k in result}
            result = _attach_rag_metadata(result, rag, certain_implicit,
                                          possible_implicit, matched_domain)
            if verbose:
                _print_formal_expression(result)
            return result
        # Unknown type -> fall back to the LLM path instead of failing outright.

    # ── Align mode (default): deterministic compose first + LLM increment ───────
    #   Key fix: do not let the LLM freely generate the whole Formal Expression
    #   (that drifts type/summary/value_or_source away from the Stage-2 training
    #   distribution). Two steps:
    # (1) Compose the base template and detected variants deterministically.
    # (2) Use the LLM only for genuinely novel constraint wording, keeping all
    # generated types and fields within the training distribution.
    if STAGE1_MODE == "align" and matched_ref:
        result = _compose_formal_expression(matched_ref, nl_text, matched_domain,
                                            verbose=verbose)
        result = _append_novel_constraints_if_any(result, nl_text, matched_domain,
                                                  verbose=verbose)
        # API self-check: audit the KB-mapped constraints/parameters against the
        # original NL (numeric correctness, missing/extra constraints) and fix
        # deterministically. This is in addition to the novel-constraint fallback.
        result = _verify_formal_against_nl(result, nl_text, matched_domain,
                                           verbose=verbose)
        # Unconditional deterministic de-duplication: even if STAGE1_VERIFY is off,
        # or the compose/novel step introduced a duplicate, guarantee the constraint
        # list handed to Stage-2 has no duplicates (ids renumbered contiguously).
        _deduped = _dedup_constraints(result.get("constraints", []), verbose=verbose)
        for _i, _c in enumerate(_deduped, start=1):
            _c["id"] = f"C{_i}"
        result["constraints"] = _deduped
        _label = result.get("constraint_label", "base")
        result = {k: result[k] for k in FORMAL_EXPR_KEYS if k in result}
        result["constraint_label"] = _label
        result = _attach_rag_metadata(result, rag, certain_implicit,
                                      possible_implicit, matched_domain)
        if verbose:
            _print_formal_expression(result)
        return result

    refs_overview   = _all_refs_overview_block(kb_refs)
    # ★ The canonical few-shot example pins output to training conventions.
    canonical_block = _canonical_example_block(matched_domain, matched_ref)
    if verbose:
        if canonical_block:
            print(f"  Using canonical formal-expression template for '{matched_domain}' "
                  f"(KB has {len(kb_refs)} references)")
        else:
            print(f"  ⚠ No canonical template for '{matched_domain}' — relying on the "
                  f"{len(kb_refs)}-type style overview only")

    vocab_block = _vocab_block(matched_domain)

    user_msg = f"""Analyze the following optimization problem description and extract its components as a Formal Expression.

{refs_overview}

{canonical_block}

{vocab_block}

## Concept-lattice retrieval (RAG-FCA — structural, not example text):
{_concept_block(rag)}

{implicit_hint}

## Problem description to analyze:
{nl_text}

## RULES (critical for downstream model compatibility):
  1. Use the SAME decision-variable naming convention as the canonical example above.
  2. constraint 'type' MUST be one of the CONTROLLED VOCABULARY keywords listed
     above for this problem type. Do NOT invent new type names — map any extra or
     variant constraint to the single closest allowed keyword.
  3. Keep constraints at the SAME granularity (e.g. tight and loose temporal
     constraints are SEPARATE entries with type 'time_window').
  4. Write the objective 'expression' in the SAME format as the canonical example
     (e.g. 'minimize(max([endOf(x[j][o]) for all j,o]))', not 'max_{{j,s}} endOf(...)').
  5. List 'parameters' in EXACTLY the canonical order shown above, and keep the
     base constraints (C1..Cn) identical to the canonical example; append any
     variant constraint AFTER them with the closest allowed 'type' keyword.

## Output (strict JSON — exactly these keys, matching the canonical example structure):
{{
  "problem_type": "{matched_domain or 'Aircraft Skin Processing | DNA Sequence Design | VRP | Battery Pack Design | Charging Station Location | Other'}",
  "problem_summary": "one-sentence summary",
  "decision_variables": [
    {{"name": "", "type": "interval_var|integer_var|binary_var|interval_var (optional)|etc", "description": "", "domain": ""}}
  ],
  "parameters": [
    {{"name": "", "description": "", "value_or_source": ""}}
  ],
  "constraints": [
    {{"id": "C1", "type": "use the canonical vocabulary", "description": ""}}
  ],
  "objective": {{
    "direction": "minimize|maximize|feasibility",
    "expression": "match canonical format",
    "description": ""
  }},
  "implicit_constraints": {json.dumps(certain_implicit + possible_implicit)},
  "certain_implicit": {json.dumps(certain_implicit)},
  "possible_implicit": {json.dumps(possible_implicit)}
}}"""

    raw = chat_for_stage("nl_structuring",
                         [{"role": "system", "content": system("nl_structurer")},
                          {"role": "user",   "content": user_msg}], json_mode=True)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(m.group()) if m else {"raw_response": raw}

    # Always enforce FCA-derived implicit constraint fields regardless of what
    # the LLM wrote, so downstream stages always see the FCA analysis results.
    result = _attach_rag_metadata(result, rag, certain_implicit,
                                  possible_implicit, matched_domain)

    if verbose:
        print(f"\n  Problem type : {result.get('problem_type')}")
        print(f"  Summary      : {result.get('problem_summary', '')[:80]}")
        print(f"  Variables    : {len(result.get('decision_variables', []))}")
        for v in result.get("decision_variables", []):
            print(f"    {v.get('name')} [{v.get('type')}]")
        print(f"  Constraints  : {len(result.get('constraints', []))}")
        for c in result.get("constraints", []):
            print(f"    [{c.get('id')}] {c.get('type')} — {c.get('description','')[:55]}")
        obj = result.get("objective", {})
        print(f"  Objective    : {obj.get('direction','').upper()} {obj.get('expression','')[:70]}")
        # Show the clean formal expression that will be passed to Stage 2
        formal_expr_preview = {k: result[k] for k in FORMAL_EXPR_KEYS if k in result}
        print(f"\n  ── Formal Expression (Stage-2 input, {len(formal_expr_preview)} keys) ──")
    return result
