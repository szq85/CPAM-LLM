"""
chaos_augment.py
================
Usage
-----
  # prove the code-perturbation is safe on the whole base set:
  python chaos_augment.py --selftest \
      --stage2 stage2_formal2model_fixed.json \
      --stage3 stage3_model2code_fixed.json

  # build an augmented corpus (T iterations, optional sample cap):
  python chaos_augment.py --augment --iterations 1 --limit 200 \
      --stage2 stage2_formal2model_fixed.json \
      --stage3 stage3_model2code_fixed.json \
      --out-stage2 stage2_augmented.json \
      --out-stage3 stage3_augmented.json
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import hashlib
import io
import json
import keyword
import random
import re
import tokenize
from typing import Callable, Dict, List, Optional, Tuple

# =========================================================================== #
# CONFIG — set input / output paths and run parameters here, then run the
# script directly:  python chaos_augment.py
# Command-line flags (if given) still override these values.
# =========================================================================== #
STAGE2_INPUT  = "sft_data/stage2_formal2model_fixed.json"     # base stage-2 (spec -> model)
STAGE3_INPUT  = "sft_data/stage3_model2code_fixed.json"       # base stage-3 (model -> code)
STAGE2_OUTPUT = "sft_data/stage2_formal2model_augmented.json"  # augmented stage-2 out
STAGE3_OUTPUT = "sft_data/stage3_model2code_augmented.json"    # augmented stage-3 out

ITERATIONS    = 20        # number of chaotic-augmentation passes T
LIMIT         = None     # cap #base samples (None = all); useful for a quick demo
ENABLE_REORDER = True    # allow safe reordering of independent model.add(...) lines
DEDUP         = True     # drop duplicates of D_0 / each other and no-op perturbations
RUN_SELFTEST  = False    # True -> only run the T_code safety self-test (no output)
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Paper hyper-parameters (Eq. 1-2). Defaults kept small so perturbation is
# meaningful but conservative; tune per the paper's a_max/b_max/g_max.
# --------------------------------------------------------------------------- #
CHAOS_X0: float = 0.6      # initial value x_0  (sensitive dependence => any value works)
CHAOS_R: float = 3.99      # mapping parameter r in [3.5, 4]
ALPHA_MAX: float = 0.35    # max text  perturbation rate
BETA_MAX: float = 0.40     # max model perturbation rate
GAMMA_MAX: float = 0.50    # max code  perturbation strength

_PROTECTED = set(keyword.kwlist) | set(dir(builtins)) | {
    # soft keywords / dunder-ish names we never touch
    "match", "case", "type", "_", "__file__", "__name__", "__doc__",
}


# =========================================================================== #
# Eq. 1 - 2 : chaotic sequence and perturbation parameters
# =========================================================================== #
def logistic_sequence(n: int, x0: float = CHAOS_X0, r: float = CHAOS_R) -> List[float]:
    """Eq.1: return n successive values of x_{n+1} = r*x_n*(1-x_n)."""
    seq, x = [], x0
    for _ in range(n):
        x = r * x * (1.0 - x)
        seq.append(x)
    return seq


def perturbation_params(x: float) -> Tuple[float, float, float]:
    """Eq.2: map a chaotic value x in (0,1) to (alpha, beta, gamma)."""
    return ALPHA_MAX * x, BETA_MAX * x, GAMMA_MAX * x


# =========================================================================== #
# Eq. 3 : T_lang  -- text-level perturbation (semantics-preserving)
# =========================================================================== #
# Synonyms are intentionally domain-safe: they vary surface wording without
# changing the optimisation meaning. Numbers, symbols and structural JSON keys
# are never touched.
_SYNONYM_MAP: Dict[str, List[str]] = {
    "minimize": ["reduce", "minimise"], "maximize": ["increase", "maximise"],
    "constraint": ["restriction", "condition"], "constraints": ["restrictions", "conditions"],
    "objective": ["goal", "target"], "schedule": ["plan", "timetable"],
    "machine": ["resource", "processor"], "machines": ["resources", "processors"],
    "vehicle": ["truck", "carrier"], "depot": ["warehouse", "hub"],
    "sequence": ["series", "chain"], "duration": ["processing time", "length"],
    "capacity": ["limit", "throughput"], "demand": ["requirement", "need"],
    "optimal": ["best", "ideal"], "feasible": ["valid", "admissible"],
    "assign": ["allocate", "place"], "assigned": ["allocated", "placed"],
    "complete": ["finish", "finalise"], "process": ["handle", "execute"],
    "each": ["every", "per"], "ensure": ["guarantee", "make sure"],
    "minimum": ["smallest", "least"], "maximum": ["largest", "greatest"],
}

# Free-text fields inside the formal-spec / model JSON that are safe to reword.
# Anything not in this set (symbols, math_expression, domains, numeric values)
# is left exactly as-is.
_TEXT_FIELDS = {"problem_summary", "description", "complexity_note"}


def _perturb_text(text: str, alpha: float, rng: random.Random) -> str:
    """Synonym substitution + light, number-safe word reordering at rate alpha."""
    if not text or alpha <= 0:
        return text
    words = text.split()
    if not words:
        return text
    # synonym substitution on ~alpha of the tokens
    n_sub = max(1, int(len(words) * alpha))
    for i in rng.sample(range(len(words)), min(n_sub, len(words))):
        key = words[i].lower().strip(".,;:!?()[]")
        if key in _SYNONYM_MAP:
            repl = rng.choice(_SYNONYM_MAP[key])
            # preserve leading capitalisation
            if words[i][:1].isupper():
                repl = repl[:1].upper() + repl[1:]
            words[i] = words[i].replace(words[i].strip(".,;:!?()[]"), repl)
    # gentle adjacency swaps on non-numeric tokens only (never reorder numbers)
    if alpha > 0.2 and len(words) > 8:
        for _ in range(max(1, int(len(words) * alpha * 0.25))):
            i = rng.randrange(len(words) - 1)
            a, b = words[i], words[i + 1]
            if not any(ch.isdigit() for ch in a + b):
                words[i], words[i + 1] = b, a
    return " ".join(words)


def _walk_text_fields(obj, alpha: float, rng: random.Random):
    """Recursively reword only the whitelisted free-text fields of a JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _TEXT_FIELDS and isinstance(v, str):
                obj[k] = _perturb_text(v, alpha, rng)
            else:
                _walk_text_fields(v, alpha, rng)
    elif isinstance(obj, list):
        for item in obj:
            _walk_text_fields(item, alpha, rng)


def T_lang(spec_json: str, alpha: float, rng: random.Random) -> str:
    """
    Eq.3: perturb the natural-language content of a JSON spec string while
    keeping the JSON structure, all symbols and all numeric values intact.
    Falls back to the original string if it is not valid JSON.
    """
    try:
        obj = json.loads(spec_json)
    except (json.JSONDecodeError, TypeError):
        return spec_json
    _walk_text_fields(obj, alpha, rng)
    return json.dumps(obj, ensure_ascii=False)


# =========================================================================== #
# Eq. 4 : T_model -- model-level perturbation (semantics-preserving)
# =========================================================================== #
def _rename_symbol_in_model(model: dict, old: str, new: str) -> None:
    """Consistently rename a decision-variable base symbol across all string
    fields of the model (symbol lists, math_expression, objective)."""
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(old) + r"(?![A-Za-z0-9_])")

    def sub_strings(o):
        if isinstance(o, dict):
            return {k: sub_strings(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sub_strings(v) for v in o]
        if isinstance(o, str):
            return pat.sub(new, o)
        return o

    new_model = sub_strings(model)
    model.clear()
    model.update(new_model)


def T_model(model_json: str, beta: float, rng: random.Random) -> str:
    """
    Eq.4: perturb the mathematical model via constraint/section reordering and
    optional consistent variable/index renaming, preserving semantics.

    Input/Output are the JSON *string* form of the model (this is exactly the
    text stored in stage2.output and stage3.instruction).
    """
    try:
        model = json.loads(model_json)
    except (json.JSONDecodeError, TypeError):
        return model_json
    if not isinstance(model, dict):
        return model_json

    # (a) reorder constraints; renumber positional ids C1,C2,... (order-free)
    cons = model.get("constraints_section")
    if isinstance(cons, list) and len(cons) > 1 and beta > 0.10:
        rng.shuffle(cons)
        for idx, c in enumerate(cons, start=1):
            if isinstance(c, dict) and isinstance(c.get("id"), str) and re.fullmatch(r"C\d+", c["id"]):
                c["id"] = f"C{idx}"

    # (b) reorder parameter / variable listings (purely presentational order)
    for sec in ("parameters_section", "variables_section"):
        lst = model.get(sec)
        if isinstance(lst, list) and len(lst) > 1 and beta > 0.20:
            rng.shuffle(lst)

    # (c) optional consistent symbol suffixing on one decision variable
    if beta > 0.30:
        vs = model.get("variables_section")
        if isinstance(vs, list) and vs:
            v = rng.choice(vs)
            sym = v.get("symbol", "") if isinstance(v, dict) else ""
            base = re.match(r"^([A-Za-z][A-Za-z0-9]*)", sym or "")
            if base:
                old = base.group(1)
                suffix = rng.choice(["_v", "_s", "_p"])
                new = old + suffix
                # only if the new base does not already occur as a symbol prefix
                existing = " ".join(
                    x.get("symbol", "") for x in vs if isinstance(x, dict)
                )
                if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(new), existing):
                    _rename_symbol_in_model(model, old, new)

    return json.dumps(model, ensure_ascii=False)


# =========================================================================== #
# Eq. 5 : T_code -- SAFE code-level perturbation for CP (docplex) programs
# =========================================================================== #
def _classify_identifiers(tree: ast.AST) -> Tuple[set, set]:
    """Return (renameable_local_names, all_reserved_names).

    renameable_local_names: locally bound variables / loop & comprehension
        targets that are NOT protected and NOT referenced inside any f-string.
    all_reserved_names: every identifier that appears anywhere (used to keep
        generated names collision-free) plus builtins/keywords.
    """
    bound, used, imported, fstring = set(), set(), set(), set()

    def add_target(t):
        if isinstance(t, ast.Name):
            bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List, ast.Starred)):
            for e in getattr(t, "elts", []) or [getattr(t, "value", None)]:
                if e is not None:
                    add_target(e)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                imported.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.Assign,)):
            for tgt in node.targets:
                add_target(tgt)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            add_target(node.target)
        elif isinstance(node, ast.For):
            add_target(node.target)
        elif isinstance(node, ast.comprehension):
            add_target(node.target)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            add_target(node.optional_vars)
        elif isinstance(node, ast.JoinedStr):  # f-strings: collect referenced names
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    fstring.add(sub.id)

    renameable = (bound - _PROTECTED - imported - fstring)
    reserved = used | imported | _PROTECTED
    return renameable, reserved


# readable preferred renames; only used when target is free of collisions
_STYLE_RENAME: Dict[str, List[str]] = {
    "mdl": ["model", "cp_model", "prob"], "model": ["mdl", "cp_model", "prob"],
    "i": ["idx", "ii", "a"], "j": ["jdx", "jj", "b"], "k": ["kdx", "kk", "c"],
    "jx": ["j_idx", "jpos"], "ox": ["o_idx", "opos"], "cx": ["c_idx", "cpos"],
    "x": ["xv", "var_x"], "y": ["yv", "var_y"], "m": ["mc", "mach"],
    "d": ["dur", "dd"], "solution": ["sol", "result", "msol"],
    "words": ["seqs", "tokens"], "word_list": ["vocab", "lexicon"],
}


def _fresh_name(old: str, reserved: set, rng: random.Random) -> Optional[str]:
    """Pick a new identifier for `old` that collides with nothing in `reserved`."""
    for cand in _STYLE_RENAME.get(old, []):
        if cand.isidentifier() and cand not in reserved:
            return cand
    for suf in ("_v", "_v2", "_x", "_n", "_2", "_3"):
        cand = old + suf
        if cand.isidentifier() and cand not in reserved:
            return cand
    # last resort: numbered
    for n in range(2, 50):
        cand = f"{old}_{n}"
        if cand not in reserved:
            return cand
    return None


def _apply_token_rename(src: str, rmap: Dict[str, str]) -> str:
    """Rename NAME tokens via the token stream. Never edits strings/comments;
    skips attribute accesses (a NAME immediately after '.')."""
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    edits: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
    prev_op_dot = False
    for t in toks:
        if t.type == tokenize.NAME and t.string in rmap and not prev_op_dot:
            edits.append((t.start, t.end, rmap[t.string]))
        # track whether the previous *significant* token was a '.'
        if t.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                      tokenize.DEDENT, tokenize.COMMENT, tokenize.ENCODING):
            continue
        prev_op_dot = (t.type == tokenize.OP and t.string == ".")
    lines = src.splitlines(keepends=True)
    for (sr, sc), (er, ec), new in sorted(edits, key=lambda e: (e[0][0], e[0][1]), reverse=True):
        ln = sr - 1
        # single-line identifiers only (always true for NAME tokens)
        lines[ln] = lines[ln][:sc] + new + lines[ln][ec:]
    return "".join(lines)


def _rename_variables(code: str, gamma: float, rng: random.Random) -> str:
    """Safely rename a gamma-scaled subset of local variables; compile-guarded."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    renameable, reserved = _classify_identifiers(tree)
    if not renameable:
        return code
    pool = sorted(renameable)
    rng.shuffle(pool)
    n = max(1, int(round(gamma * len(pool))))
    rmap: Dict[str, str] = {}
    reserved = set(reserved)
    for name in pool[:n]:
        new = _fresh_name(name, reserved, rng)
        if new:
            rmap[name] = new
            reserved.add(new)
    if not rmap:
        return code
    try:
        out = _apply_token_rename(code, rmap)
        compile(out, "<aug-rename>", "exec")
        return out
    except (SyntaxError, tokenize.TokenError, ValueError, IndentationError):
        return code  # fall back: never emit broken code


def _perturb_comments(code: str, gamma: float, rng: random.Random) -> str:
    """Reposition / trim comments (semantics-free). Operates line-by-line, never
    touches code tokens; compile-guarded as a final safety net."""
    lines = code.splitlines(keepends=True)
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # a standalone full-line comment: maybe drop, or fold into the next
        # statement as a trailing inline comment.
        if stripped.startswith("#"):
            roll = rng.random()
            comment_text = stripped[1:].strip()
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            nxt_code = nxt.split("#", 1)[0].rstrip("\n")
            can_fold = (
                gamma > 0.25
                and nxt_code.strip()
                and not nxt_code.strip().startswith("#")
                and "#" not in nxt  # don't stack onto an already-commented line
                and nxt.rstrip("\n") == nxt_code  # next line has no string with '#'
                and len(nxt_code) + len(comment_text) < 95
            )
            if gamma > 0.35 and roll < gamma * 0.35:
                i += 1  # drop this comment line
                continue
            if can_fold and roll < 0.5:
                nl = "\n" if nxt.endswith("\n") else ""
                out.append(f"{nxt_code}  # {comment_text}{nl}")
                i += 2
                continue
        out.append(line)
        i += 1
    result = "".join(out)
    try:
        compile(result, "<aug-comments>", "exec")
        return result
    except (SyntaxError, ValueError, IndentationError):
        return code


def _reorder_constraint_adds(code: str, gamma: float, rng: random.Random) -> str:
    """Shuffle runs of consecutive, single-line, independent `*.add(...)`
    statements. Adding constraints in a different order yields an equivalent CP
    model. Strictly guarded: same indentation, no interleaving lines, identical
    multiset of lines preserved, and final compile()."""
    if gamma <= 0.20:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # line numbers that are single-line `<obj>.add(...)` expression statements
    add_lines = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "add"
                and getattr(node, "lineno", None) == getattr(node, "end_lineno", None)):
            add_lines.add(node.lineno)

    if len(add_lines) < 2:
        return code

    lines = code.splitlines(keepends=True)
    indent_of = lambda s: len(s) - len(s.lstrip())
    i, n = 0, len(lines)
    changed = False
    while i < n:
        if (i + 1) in add_lines:
            j = i
            base_indent = indent_of(lines[i])
            block_idx = []
            while j < n and (j + 1) in add_lines and indent_of(lines[j]) == base_indent:
                block_idx.append(j)
                j += 1
            if len(block_idx) >= 2:
                block = [lines[b] for b in block_idx]
                order = list(range(len(block)))
                rng.shuffle(order)
                if order != list(range(len(block))):
                    for pos, b in enumerate(block_idx):
                        lines[b] = block[order[pos]]
                    changed = True
            i = j
        else:
            i += 1

    if not changed:
        return code
    result = "".join(lines)
    try:
        compile(result, "<aug-reorder>", "exec")
        return result
    except (SyntaxError, ValueError, IndentationError):
        return code


def T_code(code: str, gamma: float, rng: random.Random,
           enable_reorder: bool = True) -> str:
    """
    Eq.5: structure-aware, semantics-preserving code perturbation for docplex CP
    programs. Guaranteed to return code that `compile()`s (falls back to the
    original on any failure).

    Steps, each individually compile-guarded:
      1. safe local-variable renaming   (gamma-scaled, AST+tokenize, collision-free)
      2. comment repositioning / trimming
      3. independent `*.add(...)` constraint reordering   (optional)
    """
    if not isinstance(code, str) or gamma <= 0:
        return code
    original = code
    try:
        compile(original, "<aug-src>", "exec")
    except (SyntaxError, ValueError, IndentationError):
        return original  # un-parseable source: do not perturb

    code = _rename_variables(code, gamma, rng)
    code = _perturb_comments(code, gamma, rng)
    if enable_reorder:
        code = _reorder_constraint_adds(code, gamma, rng)

    # final validation gate
    try:
        compile(code, "<aug-final>", "exec")
        return code
    except (SyntaxError, ValueError, IndentationError):
        return original


# =========================================================================== #
# Eq. 7 : validation gate  (CheckCompilation; ManualReview is a human step)
# =========================================================================== #
def check_compilation(code: str) -> bool:
    """v_compile in Algorithm 1: True iff the code compiles."""
    try:
        compile(code, "<validate>", "exec")
        return True
    except (SyntaxError, ValueError, IndentationError):
        return False


def _is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def is_valid_triple(spec_json: str, model_json: str, code: str) -> bool:
    """Reject an augmented sample as erroneous unless the code compiles AND both
    the spec and the model are valid JSON (defensive against a bad reframer)."""
    return check_compilation(code) and _is_valid_json(spec_json) and _is_valid_json(model_json)


def _canon_json(s: str) -> str:
    """Canonical form of a JSON string for hashing: dict keys sorted and
    whitespace normalised, but LIST ORDER preserved (so a genuine constraint /
    section reordering still counts as a distinct sample). Non-JSON returned
    as-is."""
    try:
        return json.dumps(json.loads(s), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        return s


def triple_signature(spec_json: str, model_json: str, code: str) -> str:
    """Stable hash of a (spec, model, code) triple, for deduplication. JSON
    components are canonicalised first, so two samples that differ only in key
    order or whitespace collide, while reordered lists and renamed/edited code
    remain distinct."""
    h = hashlib.md5()
    h.update(_canon_json(spec_json).encode("utf-8")); h.update(b"\x00")
    h.update(_canon_json(model_json).encode("utf-8")); h.update(b"\x00")
    h.update(code.encode("utf-8"))
    return h.hexdigest()


# =========================================================================== #
# Eq. 6 : LLM stylistic reframing hook (pluggable; identity when offline)
# =========================================================================== #
def llm_reframe(spec_json: str, model_json: str, code: str,
                reframer: Optional[Callable[[str, str, str], Tuple[str, str, str]]] = None
                ) -> Tuple[str, str, str]:
    """Eq.6: optional LLM reframing on top of the structural perturbations.
    Provide `reframer(spec, model, code) -> (spec, model, code)` to enable it;
    otherwise the structurally-perturbed triple is returned unchanged."""
    if reframer is None:
        return spec_json, model_json, code
    try:
        d, m, c = reframer(spec_json, model_json, code)
        return d or spec_json, m or model_json, (c if check_compilation(c) else code)
    except Exception:
        return spec_json, model_json, code


# =========================================================================== #
# Algorithm 1 : orchestration  (D_SFT = D_0 U accepted augmentations)
# =========================================================================== #
def augment_dataset(
    stage2: List[dict],
    stage3: List[dict],
    iterations: int = 1,
    x0: float = CHAOS_X0,
    r: float = CHAOS_R,
    reframer: Optional[Callable] = None,
    enable_reorder: bool = True,
    limit: Optional[int] = None,
    dedup: bool = True,
    verbose: bool = True,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Run chaotic augmentation. Returns (aug_stage2, aug_stage3, stats) where the
    two lists stay index-aligned and the base records (D_0) come first.

    The shared model object produced by T_model is written identically to the
    new stage2.output and stage3.instruction so the two files remain consistent.
    """
    assert len(stage2) == len(stage3), "stage2/stage3 must be aligned 1:1"
    base = list(zip(stage2, stage3))
    if limit is not None:
        base = base[:limit]

    out2: List[dict] = [copy.deepcopy(s2) for s2, _ in base]   # D_0
    out3: List[dict] = [copy.deepcopy(s3) for _, s3 in base]

    # Seed the dedup set with every base triple so an augmented sample can never
    # duplicate D_0 (this also drops "no-op" perturbations that fell back to the
    # original). Augmented samples are then checked against each other too.
    seen = set()
    if dedup:
        for s2, s3 in base:
            seen.add(triple_signature(s2.get("instruction", ""), s2.get("output", ""),
                                      s3.get("output", "")))

    x = x0
    produced = accepted = invalid = duplicate = 0

    for k in range(1, iterations + 1):
        for i, (s2, s3) in enumerate(base):
            # advance the single shared chaotic sequence (Algorithm 1, line 5)
            x = r * x * (1.0 - x)
            alpha, beta, gamma = perturbation_params(x)
            # derive a deterministic per-step seed from the chaotic state
            rng = random.Random(int((x * 1e9)) ^ (k * 1_000_003) ^ (i * 7919))

            produced += 1
            spec_in = s2.get("instruction", "")
            model_in = s2.get("output", "")        # == s3["instruction"]
            code_in = s3.get("output", "")

            # structural perturbations (Eq. 3-5)
            spec_p = T_lang(spec_in, alpha, rng)
            model_p = T_model(model_in, beta, rng)
            code_p = T_code(code_in, gamma, rng, enable_reorder=enable_reorder)

            # stylistic reframing (Eq. 6, optional)
            spec_p, model_p, code_p = llm_reframe(spec_p, model_p, code_p, reframer)

            # validation gate (Eq. 7): drop erroneous samples (code must compile,
            # spec and model must be valid JSON)
            if not is_valid_triple(spec_p, model_p, code_p):
                invalid += 1
                continue

            # deduplication: drop exact duplicates of D_0 or of an already-kept
            # augmented sample (also removes no-op perturbations)
            if dedup:
                sig = triple_signature(spec_p, model_p, code_p)
                if sig in seen:
                    duplicate += 1
                    continue
                seen.add(sig)

            # keep stage2.output and stage3.instruction identical (linkage)
            out2.append({"instruction": spec_p, "input": s2.get("input", ""), "output": model_p})
            out3.append({"instruction": model_p, "input": s3.get("input", ""), "output": code_p})
            accepted += 1

        if verbose:
            print(f"  iteration {k}: produced={produced} accepted={accepted} "
                  f"invalid={invalid} duplicate={duplicate}")

    stats = {
        "base_samples": len(base),
        "iterations": iterations,
        "dedup": dedup,
        "produced": produced,
        "accepted": accepted,
        "invalid_dropped": invalid,
        "duplicate_dropped": duplicate,
        "accept_rate": round(accepted / produced, 4) if produced else 0.0,
        "final_stage2": len(out2),
        "final_stage3": len(out3),
        "unique_total": len(out2),
    }
    return out2, out3, stats


# =========================================================================== #
# self-test : prove T_code never breaks the base CP code (any gamma)
# =========================================================================== #
def selftest(stage3: List[dict], gammas=(0.1, 0.25, 0.4, 0.5, 0.8, 1.0)) -> dict:
    base_codes = [r.get("output", "") for r in stage3]
    base_ok = sum(check_compilation(c) for c in base_codes)
    print(f"base codes: {len(base_codes)} | compile OK: {base_ok}")

    results = {}
    for g in gammas:
        ok = changed = broke = 0
        for idx, c in enumerate(base_codes):
            if not check_compilation(c):
                continue  # skip any base sample that itself doesn't compile
            rng = random.Random(1234 + idx)
            out = T_code(c, g, rng)
            if check_compilation(out):
                ok += 1
            else:
                broke += 1
            if out != c:
                changed += 1
        results[g] = {"compiled": ok, "broken": broke, "changed": changed}
        print(f"  gamma={g:<4}: compiled={ok}/{base_ok}  broken={broke}  changed={changed}")
        assert broke == 0, f"T_code produced non-compiling code at gamma={g}!"
    print("PASS: T_code is safe at every tested gamma (0 broken).")
    return results


# --------------------------------------------------------------------------- #
def _load(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================================== #
# Dataset export from KB records
#
# This builds an SFT dataset from knowledge-base records and is the
# implementation behind KnowledgeBase.export_sft_dataset.
# A KB record is a dict with at least: "nl_text", "problem_type", "cp_code".
# Because chaos_map.py re-exports this module with `import *`, the function
# is also importable as augmentation.chaos_map.export_sft_dataset.
# =========================================================================== #
def _records_to_stage_pairs(records: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Map KB records to the index-aligned (stage2, stage3) lists that
    augment_dataset consumes. KB records carry a natural-language spec
    (nl_text) and solver code (cp_code) but no separate formal-model field, so
    the model slot is a small JSON object derived from the record. The spec and
    model are emitted as JSON strings, as required by the validity gate."""
    stage2, stage3 = [], []
    for r in records:
        if not r.get("cp_code"):
            continue
        spec_json = json.dumps({"description": r.get("nl_text", "")}, ensure_ascii=False)
        model_json = json.dumps(
            {"problem_type": r.get("problem_type", ""), "constraint": r.get("constraint", "")},
            ensure_ascii=False,
        )
        stage2.append({"instruction": spec_json, "input": "", "output": model_json})
        stage3.append({"instruction": model_json, "input": "", "output": r["cp_code"]})
    return stage2, stage3


def export_sft_dataset(
    records: List[dict],
    output_path: str,
    include_augmented: bool = True,
    n_aug_iterations: int = 1,
    use_llm_reframe: bool = False,
) -> int:
    """Export a supervised fine-tuning (SFT) dataset from KB records.

    Each output item is an instruction/input/output triple in LLaMA-Factory
    format that maps a natural-language specification to the solver code. When
    include_augmented is True, chaotic-mapping augmentation adds extra samples.

    Returns the number of samples written.
    """
    stage2, stage3 = _records_to_stage_pairs(records)

    if include_augmented and stage2:
        # use_llm_reframe is accepted for API compatibility; a concrete LLM
        # reframer can be passed through augment_dataset's `reframer` argument
        # when one is configured. With none wired in, augmentation is structural.
        _, aug3, _ = augment_dataset(
            stage2, stage3,
            iterations=n_aug_iterations,
            reframer=None,
            verbose=False,
        )
        rows = aug3
    else:
        rows = stage3

    # stage3 items already have the spec->code shape we want for SFT:
    # instruction = formal model, output = solver code. Re-key to a clean
    # instruction/input/output record for the SFT file.
    sft = [
        {"instruction": row.get("instruction", ""),
         "input": row.get("input", ""),
         "output": row.get("output", "")}
        for row in rows
    ]
    _save(sft, output_path)
    return len(sft)


def main():
    ap = argparse.ArgumentParser(
        description="Chaos-mapping augmentation (CPAM-LLM, Eq.1-8). "
                    "Edit the CONFIG block at the top, or override with the flags below.")
    ap.add_argument("--stage2", default=STAGE2_INPUT, help="base stage-2 input json")
    ap.add_argument("--stage3", default=STAGE3_INPUT, help="base stage-3 input json")
    ap.add_argument("--out-stage2", default=STAGE2_OUTPUT, help="augmented stage-2 output json")
    ap.add_argument("--out-stage3", default=STAGE3_OUTPUT, help="augmented stage-3 output json")
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--limit", type=int, default=LIMIT,
                    help="cap number of base samples (for a quick demo)")
    ap.add_argument("--no-reorder", dest="reorder", action="store_false", default=ENABLE_REORDER,
                    help="disable independent constraint-add reordering")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false", default=DEDUP,
                    help="disable deduplication (keep duplicate / no-op samples)")
    ap.add_argument("--x0", type=float, default=CHAOS_X0)
    ap.add_argument("--r", type=float, default=CHAOS_R)
    ap.add_argument("--selftest", action="store_true", default=RUN_SELFTEST,
                    help="only run the T_code safety self-test (no augmentation output)")
    args = ap.parse_args()

    s2 = _load(args.stage2)
    s3 = _load(args.stage3)
    print(f"loaded stage2={len(s2)} ({args.stage2}) | stage3={len(s3)} ({args.stage3})")

    if args.selftest:
        print("\n=== SELF-TEST: T_code safety on base corpus ===")
        selftest(s3)
        return

    print("\n=== AUGMENT ===")
    out2, out3, stats = augment_dataset(
        s2, s3,
        iterations=args.iterations,
        x0=args.x0, r=args.r,
        enable_reorder=args.reorder,
        dedup=args.dedup,
        limit=args.limit,
    )
    _save(out2, args.out_stage2)
    _save(out3, args.out_stage3)
    print("stats:", json.dumps(stats, indent=2))
    print(f"wrote {args.out_stage2} and {args.out_stage3}")


if __name__ == "__main__":
    main()