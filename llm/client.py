from __future__ import annotations
import sys, os, time, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    API_BASE_URL, API_KEY, API_MODEL,
    FT_BASE_URL,  FT_API_KEY, FT_MODEL,
    FT_STAGE_MODELS, FT_STAGE_ENDPOINTS,
    MAX_TOKENS, TEMPERATURE, STAGE_ROUTING,
    FT_REQUEST_TIMEOUT, API_REQUEST_TIMEOUT,
)

LAST_BACKEND: dict = {}

# Cumulative token usage across all LLM calls in the current process. Each API
# response carries a "usage" block ({prompt_tokens, completion_tokens,
# total_tokens}); _call accumulates it for run-level reporting.
TOKEN_USAGE: dict = {"prompt_tokens": 0, "completion_tokens": 0,
                     "total_tokens": 0, "calls": 0}


def reset_token_usage() -> None:
    """Zero the cumulative token counter (call at the start of a run)."""
    TOKEN_USAGE.update(prompt_tokens=0, completion_tokens=0,
                       total_tokens=0, calls=0)


def get_token_usage() -> dict:
    """Return a copy of the cumulative token usage so far."""
    return dict(TOKEN_USAGE)


def _accumulate_usage(resp_json: dict) -> None:
    """Add one API response's token usage into the cumulative counter."""
    usage = (resp_json or {}).get("usage") or {}
    pt = int(usage.get("prompt_tokens", 0) or 0)
    ct = int(usage.get("completion_tokens", 0) or 0)
    tt = int(usage.get("total_tokens", 0) or (pt + ct))
    TOKEN_USAGE["prompt_tokens"]     += pt
    TOKEN_USAGE["completion_tokens"] += ct
    TOKEN_USAGE["total_tokens"]      += tt
    TOKEN_USAGE["calls"]             += 1


# ─────────────────────────────────────────────────────────────
# Internal HTTP call
# ─────────────────────────────────────────────────────────────

def _call(base_url: str, api_key: str, model: str,
          messages: list, temperature: float,
          max_tokens: int, json_mode: bool,
          max_retries: int = 3, backoff: float = 1.5,
          timeout: int = API_REQUEST_TIMEOUT) -> str:
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload: dict = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                f"{base_url}/chat/completions",
                headers=headers, json=payload, timeout=timeout,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.exceptions.RequestException(
                    f"HTTP {r.status_code}")
            r.raise_for_status()
            _resp = r.json()
            _accumulate_usage(_resp)
            return _resp["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            last_err = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (400, 401, 403, 404):
                break
            if attempt < max_retries:
                wait = backoff ** attempt
                print(f"  [LLM] transient error (attempt {attempt}/{max_retries}): "
                      f"{str(e)[:80]} — retrying in {wait:.1f}s")
                time.sleep(wait)
    raise RuntimeError(f"[LLM] {last_err}")


# ─────────────────────────────────────────────────────────────
# Fine-tuned model selection and endpoint helpers.
# ─────────────────────────────────────────────────────────────

def _ft_model_for_stage(stage: str) -> str:
    """
    Return the vLLM adapter name for the current stage.
    Looks up FT_STAGE_MODELS first; falls back to FT_MODEL if undefined.

    Mapping (kept in sync with --lora-modules in start_ft_server.sh):
      math_modeling   -> "stage2"
      code_generation -> "stage3"
    """
    return FT_STAGE_MODELS.get(stage, FT_MODEL)


def _ft_url_for_stage(stage: str) -> str:
    """Return the fine-tuned service URL for the current stage (Stage2->6006, Stage3->6008)."""
    url, model = FT_STAGE_ENDPOINTS.get(stage, (FT_BASE_URL, FT_MODEL))
    return url


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def chat(
    messages: list,
    temperature: float  = TEMPERATURE,
    max_tokens: int     = MAX_TOKENS,
    json_mode: bool     = False,
    backend: str        = "api",
    stage: str          = "",  # fine-tuned adapter key
) -> str:
    """
    Call the LLM with the specified backend.

    When backend="ft":
      - if stage is defined in FT_STAGE_MODELS, use the matching adapter name
      - otherwise use FT_MODEL as fallback
      - if FT_BASE_URL is empty, fall back to the cloud API automatically
    """
    if backend == "ft":
        ft_url = _ft_url_for_stage(stage)
        if not ft_url:
            print("  [LLM] ⚠  Fine-tuned endpoint not configured "
                  "— falling back to API.")
            backend = "api"
        else:
            ft_model = _ft_model_for_stage(stage)
            try:
                return _call(ft_url, FT_API_KEY, ft_model,
                             messages, temperature, max_tokens, json_mode,
                             timeout=FT_REQUEST_TIMEOUT)
            except RuntimeError as e:
                print(f"  [LLM] ⚠  Fine-tuned model ({ft_model}) error ({e}) — falling back to API.")
                backend = "api"

    return _call(API_BASE_URL, API_KEY, API_MODEL,
                 messages, temperature, max_tokens, json_mode,
                 timeout=API_REQUEST_TIMEOUT)


def get_backend(stage: str) -> str:
    """Return the backend name ('api' or 'ft') for the given stage."""
    return STAGE_ROUTING.get(stage, "api")


def chat_ft_with_api_fallback(
    stage: str,
    ft_messages: list,
    api_messages: list,
    api_json_mode: bool = False,
    temperature: float  = TEMPERATURE,
    max_tokens: int     = MAX_TOKENS,
) -> str:
    """
    Core routing function: call the fine-tuned model first; on failure fall back
    to the cloud API with the full prompt.

    stage decides both:
      1. whether to take the ft path (STAGE_ROUTING[stage] == "ft")
      2. which adapter to use (FT_STAGE_MODELS[stage])

    Args:
        stage        : pipeline stage key (e.g. "math_modeling" / "code_generation")
        ft_messages  : concise messages for the fine-tuned model (system = training system_prompt)
        api_messages : rich-prompt messages for the cloud API (with RAG context, etc.)
        api_json_mode: whether to require JSON output from the API
        temperature  : sampling temperature
        max_tokens   : max tokens
    """
    backend = STAGE_ROUTING.get(stage, "api")

    # ── Path 1: ft not configured → API directly ─────────────────────────────
    ft_url = _ft_url_for_stage(stage)
    if backend != "ft" or not ft_url:
        label = "api (ft not configured)" if backend == "ft" else "api"
        print(f"  [LLM:{stage}] → {label}")
        LAST_BACKEND[stage] = "api(ft-not-configured)" if backend == "ft" else "api"
        return _call(API_BASE_URL, API_KEY, API_MODEL,
                     api_messages, temperature, max_tokens, api_json_mode,
                     timeout=API_REQUEST_TIMEOUT)

    # ── Path 2: try ft with stage-specific adapter; on error fall back to API ─
    # The fine-tuned model gets a long read timeout (FT_REQUEST_TIMEOUT) so the
    # pipeline WAITS for it instead of switching to the API too quickly.
    ft_model = _ft_model_for_stage(stage)
    print(f"  [Backend:{stage}] FT  adapter={ft_model}  @ {ft_url}  "
          f"(timeout={FT_REQUEST_TIMEOUT}s)")
    try:
        result = _call(ft_url, FT_API_KEY, ft_model,
                       ft_messages, temperature, max_tokens, False,
                       timeout=FT_REQUEST_TIMEOUT)
        print(f"  [Backend:{stage}] FT  ✓  ({len(result)} chars)")
        LAST_BACKEND[stage] = "ft"
        return result
    except RuntimeError as e:
        print(f"  [Backend:{stage}] FT  ✗  {str(e)[:100]}")
        print(f"  [Backend:{stage}] → Fallback to API ({API_MODEL})")
        LAST_BACKEND[stage] = "api(ft-fallback)"
        return _call(API_BASE_URL, API_KEY, API_MODEL,
                     api_messages, temperature, max_tokens, api_json_mode,
                     timeout=API_REQUEST_TIMEOUT)


def chat_for_stage(
    stage: str,
    messages: list,
    temperature: float = TEMPERATURE,
    max_tokens: int    = MAX_TOKENS,
    json_mode: bool    = False,
) -> str:
    """
    Route the call to the backend configured for the given stage in STAGE_ROUTING.
    Prints which backend and adapter are actually used.
    """
    backend  = STAGE_ROUTING.get(stage, "api")
    ft_model = _ft_model_for_stage(stage)
    ft_url   = _ft_url_for_stage(stage)
    label    = f"ft(adapter={ft_model})" if backend == "ft" and ft_url else "api"
    print(f"  [LLM:{stage}] → {label}")
    return chat(messages, temperature=temperature,
                max_tokens=max_tokens, json_mode=json_mode,
                backend=backend, stage=stage)


def system(role: str) -> str:
    return SYSTEM_PROMPTS.get(role, "You are a helpful assistant.")


# ─────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "nl_structurer": (
        "You are an expert in constraint programming (CP) and combinatorial optimization. "
        "Extract and structure the key components from natural language problem descriptions. "
        "Respond strictly in JSON with keys: problem_type, problem_summary, "
        "decision_variables, parameters, constraints, objective, implicit_constraints."
    ),
    "variant_differ": (
        "You are an expert in constraint programming. You are given a CANONICAL base "
        "Formal Expression and a NEW problem that is a VARIANT of it. Output ONLY the "
        "minimal structured DIFF the variant introduces. Do NOT restate or rewrite the "
        "whole expression, do NOT invent new constraint 'type' keywords. Respond strictly "
        "in JSON with exactly the diff keys requested."
    ),
    "formal_verifier": (
        "You are auditing an assembled formal optimization model against the original "
        "natural-language problem. The base structure is already correct; your ONLY task "
        "is to flag DIVERGENCES from the NL: numbers that disagree (bounds, percentages, "
        "capacities, budgets, counts, time-window edges, thresholds), a constraint the NL "
        "clearly requires but is missing, a constraint present that the NL does not ask "
        "for, or the SAME constraint written twice (a duplicate — same restriction on the "
        "same object). Do NOT rename constraint types, do NOT rewrite correct constraints "
        "for style, do NOT invent new type keywords. Trust the NL for all numeric values. "
        "Respond strictly in JSON with exactly the audit keys requested; use empty lists "
        "when everything matches."
    ),
    "math_modeler": (
        "You are a mathematical modeling expert for IBM CPLEX CP Optimizer (docplex.cp). "
        "Convert structured problem components into a formal CP mathematical model. "
        "In math_expression fields, use CPLEX CP Optimizer mathematical function names exactly: "
        "endOf(x), startOf(x), sizeOf(x) for interval expressions; "
        "noOverlap(t) for no-overlap; alternative(a,[b1..bn]) for machine selection; "
        "endBeforeStart(a,b), startBeforeStart(a,b) for precedence; "
        "minimize(expr), maximize(expr) for objective; "
        "sum([...]), count(a,v), element(a,i), scalProd(x,c) for arithmetic; "
        "ifThen(cond,e), pack(load,where,weight) for conditional/bin-packing; "
        "allowedAssignments(vars,table), allDiff(vars) for global constraints. "
        "Respond in JSON with keys: model_title, model_type, parameters_section, "
        "variables_section, constraints_section, objective_section, complexity_note."
    ),
    "cp_code_generator": (
        "You are a Python expert for IBM CPLEX CP Optimizer (docplex.cp). "
        "Generate complete, executable Python code using docplex.cp that implements the given CP model. "
        "STRICT STYLE RULES: "
        "(1) NO comments or inline # notes anywhere in the code. "
        "(2) Start with 'from docplex.cp.model import CpoModel' and any other needed imports. "
        "(3) Use 'mdl = CpoModel()' for model initialization. "
        "(4) Use mdl.add(...) for all constraints. "
        "(5) End with: msol = mdl.solve(TimeLimit=30) "
        "    if msol: print('Solution found.') and print relevant results. "
        "    else: print('No solution found.') "
        "(6) Return ONLY the Python code. No markdown, no code fences, no explanations."
    ),
    "dynamic_constraint": (
        "You output ONLY new Python constraint lines for a docplex.cp model. "
        "No full program. No markdown. No comments. No imports. "
        "Only mdl.add() statements and their for/if wrappers."
    ),
    "validator": (
        "You are a CPLEX CP Optimizer code reviewer. "
        "Check whether the given Python code correctly implements the described CP model. "
        "Respond in JSON: {valid: bool, confidence: float, errors: [...], "
        "suggestions: [...], coverage_check: {constraints_covered: [...], constraints_missing: [...]}}"
    ),
    "code_modifier": (
        "You are a CPLEX CP Optimizer Python expert. "
        "The user wants to modify specific parts of an existing docplex.cp program. "
        "Apply ONLY the requested changes — do not restructure or add comments. "
        "Return the COMPLETE updated Python code. No markdown. No comments. No code fences."
    ),
    "code_fixer": (
        "You are a Python debugging expert for CPLEX CP Optimizer (docplex.cp). "
        "Fix the provided code based on the error messages and missing constraints listed. "
        "Return ONLY the corrected Python code. No comments. No markdown."
    ),
}
