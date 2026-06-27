"""
CPAM-LLM Web Application

A browser UI for the CPAM-LLM pipeline whose layout and colour scheme mirror the
reference "Modou Intelligent Manufacturing / SCHEDULING" app:

  • a welcome screen with scene chips,
  • a two-panel workspace: left = conversation (basic problem + added
    constraints, each with "Copy to Modeling"), right = Model area
    (constraint list, "Generate Code"),
  • a code view with Upload / Example Dataset / Run Solver.

It keeps ALL of the existing pipeline steps — RAG-FCA structuring, math model,
CP code generation, validation, solving, add-constraint, and save-to-KB.

Run:
    python webapp/app.py
then open http://127.0.0.1:5000
"""
from __future__ import annotations
import os, sys, io, json, uuid, threading, traceback
from contextlib import redirect_stdout, redirect_stderr

# project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, render_template

DATA_DIR = os.path.join(ROOT, "data")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.json.ensure_ascii = False

# ── In-memory session store (single-user local app) ─────────────────────────
# session_id -> {"result": <pipeline result dict>, "scene": str}
SESSIONS: dict = {}
_KB = None
_KB_LOCK = threading.Lock()


def kb():
    global _KB
    with _KB_LOCK:
        if _KB is None:
            from rag_fca.knowledge_base import get_knowledge_base
            _KB = get_knowledge_base()
    return _KB


SCENES = [
    {"id": "Aircraft Skin Processing", "label": "Aircraft Skin Processing", "icon": "✈"},
    {"id": "DNA Sequence Design",      "label": "DNA Sequence Design",      "icon": "🧬"},
    {"id": "Battery Pack Design",      "label": "Battery Pack Design",      "icon": "🔋"},
    {"id": "VRP",                      "label": "Logistics Planning (VRP)", "icon": "🚚"},
    {"id": "Charging Station Location","label": "Charging Station Location","icon": "⚡"},
]

SCENE_SAMPLES = {
    "Aircraft Skin Processing":
        "This problem arises from aircraft manufacturing where multiple skin components need to be "
        "processed through a series of operations. In the flexible job shop scheduling environment, "
        "each job (representing an aircraft skin component) must be processed according to a predefined "
        "sequence of operations. Each operation can be performed on multiple candidate machines with "
        "different processing times, providing flexibility in machine selection. However, each machine "
        "can process only one operation at a time, ensuring no resource conflicts. The system "
        "automatically handles temporal constraints between consecutive operations: tight constraints "
        "require intervals of 15 time units or less, while loose constraints require intervals of 30 "
        "time units or more. The objective is to minimize the makespan, which represents the maximum "
        "completion time across all jobs, thus optimizing the overall production efficiency. Write a "
        "Python program using the docplex.cp module (IBM CP Optimizer) to solve this flexible job shop "
        "scheduling problem. The input data should be read from a file named "
        "\"fjsp_location_newdataset_1.fjs\" located in the data subdirectory. The file format is as "
        "follows: the first line contains the number of jobs and machines; subsequent lines contain job "
        "information with the number of operations, and for each operation, the number of candidate "
        "machines followed by machine-duration pairs, and temporal constraint types ('tight', 'loose', "
        "or 'none').",
    "DNA Sequence Design":
        "This problem arises from molecular biology and genetic engineering where specific DNA "
        "sequences must be designed to meet stringent biochemical constraints. All designed DNA "
        "sequences must be selected from a predefined word list that contains valid nucleotide "
        "combinations, ensuring biological feasibility. The GC content must be exactly 50% for each "
        "sequence to maintain optimal thermodynamic stability. The Hamming distance between any two "
        "different DNA sequences must be at least 4, providing sufficient discrimination between "
        "distinct sequences. Additionally, the Hamming distance between any DNA sequence and its "
        "reverse complement must be at least 4 to prevent self-hybridization. The design objective is "
        "to construct a set of 6 DNA sequences, each precisely 8 bases in length, that simultaneously "
        "satisfies all the above biochemical and structural constraints. Write a Python program using "
        "the docplex.cp module (IBM CP Optimizer) to solve this DNA sequence design problem. The input "
        "data should be read from a file named \"word.txt\" located in the data subdirectory. Each line "
        "represents a sequence as a comma-separated list of integers in brackets, where 0=A, 1=T, 2=C, "
        "3=G.",
    "Battery Pack Design":
        "This problem arises from advanced battery management systems in electric vehicles and energy "
        "storage applications, where dynamic reconfiguration of battery modules is essential for "
        "optimal performance. The system consists of multiple battery modules arranged in a matrix "
        "configuration with parallel and series connections. Each module must satisfy several "
        "fundamental constraints: the system's total output voltage must strictly remain within the "
        "preset safe power supply range to prevent damage to connected loads; the current flowing "
        "through any activated battery module must not exceed its maximum safety threshold to avoid "
        "overheating and degradation; at any given moment, the number of parallel-connected battery "
        "modules in all series branches must be strictly consistent to ensure balanced load "
        "distribution. The optimization objective is to minimize the total net energy loss of all "
        "activated modules, which encompasses both internal resistance losses during activation periods "
        "and electrochemical recovery effects during deactivation periods, thereby maximizing the "
        "overall energy efficiency of the battery pack system. Write a Python program using the "
        "docplex.cp module (IBM CP Optimizer) to solve this battery pack design optimization problem. "
        "The input data should be read from a file named \"battery_data.txt\" located in the data "
        "subdirectory. The file format is as follows: the first line contains five parameters (number "
        "of parallel modules, number of series columns, number of time periods, cell nominal voltage, "
        "and maximum current per module); subsequent lines contain three matrices (C, R, V) with "
        "dimensions [NUM_PERIODS \u00d7 NUM_PARALLEL_MODULES \u00d7 NUM_SERIES_COLUMNS], each flattened "
        "row by row.",
    "VRP":
        "This problem arises from logistics and supply chain management where a fleet of vehicles must "
        "efficiently deliver goods to multiple customer locations. The operational framework imposes "
        "several fundamental constraints: the total goods delivered by each vehicle must not exceed its "
        "maximum load capacity; each customer point must be visited exactly once by one vehicle to "
        "fulfill its entire demand; each vehicle must depart from the central depot, serve its assigned "
        "customers, and return to the depot. The generated set of delivery paths must be logically "
        "conflict-free and executable. The optimization objective is to minimize the total travel "
        "distance of the entire vehicle fleet. Write a Python program using the docplex.cp module (IBM "
        "CP Optimizer) to solve this vehicle routing problem with time windows. The input data should "
        "be read from a text file (e.g., \"25c101.txt\") in the data subdirectory. Each row contains: "
        "customer number, x-coord, y-coord, demand, ready time, due time, service time.",
    "Charging Station Location":
        "This problem arises from urban infrastructure planning where electric vehicle charging "
        "facilities must be strategically deployed to serve a growing population of EV users. In the "
        "facility location optimization framework, each defined demand area has a predicted charging "
        "demand that must be fully served to ensure adequate service coverage. Each candidate station "
        "location has a maximum service capacity constraint, limiting the number of users or charging "
        "events it can accommodate. The decision involves determining the optimal subset of candidate "
        "locations to activate as charging stations. The objective is to minimize the combined cost "
        "function, which includes both the sum of fixed construction costs for all activated stations "
        "and the total service cost for all users, thereby achieving an economically efficient charging "
        "network layout. Write a Python program using the docplex.cp module (IBM CP Optimizer) to solve "
        "this facility location problem for charging station placement. The input data should be read "
        "from a file named \"plant_location.data\" located in the data subdirectory. The file format "
        "is: first two integers indicate the number of demand areas and candidate locations; followed "
        "by a service-cost matrix [nbCustomer \u00d7 nbLocation]; then demand values; fixed construction "
        "costs; and finally capacity limits.",
}


# ── helper: capture pipeline prints into a string (for the console drawer) ───
class _Tee(io.StringIO):
    pass


def _run_capture(fn):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            out = fn()
        return out, buf.getvalue(), None
    except Exception:
        return None, buf.getvalue(), traceback.format_exc()


def _structured_view(result: dict) -> dict:
    """Shape the pipeline result for the frontend."""
    fo = result.get("final_output", {})
    st = fo.get("_structured", {})
    mm = fo.get("math_model", {})
    return {
        "problem_type":    fo.get("problem_type", ""),
        "problem_summary": fo.get("problem_summary", ""),
        "constraint_label":fo.get("constraint_label", ""),
        "validation_score":fo.get("validation_score", 0.0),
        "decision_variables": st.get("decision_variables", []),
        "constraints":     st.get("constraints", []),
        "objective":       st.get("objective", {}),
        "certain_implicit":  st.get("certain_implicit", []),
        "possible_implicit": st.get("possible_implicit", []),
        "matched_domain":  st.get("matched_domain", ""),
        "rag_concepts":    st.get("_rag_concepts", []),
        "formal_templates":st.get("formal_templates", []),
        "math_model":      mm,
        "cp_code":         fo.get("cp_code", ""),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scenes")
def api_scenes():
    return jsonify({"status": "success", "scenes": SCENES,
                    "samples": SCENE_SAMPLES})


@app.route("/api/kbinfo")
def api_kbinfo():
    s = kb().summary()
    return jsonify({"status": "success",
                    "records": s["total_records"],
                    "concepts": s["total_concepts"],
                    "attributes": s["total_attributes"]})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Step 1-3: run the full pipeline on the basic problem description."""
    body  = request.get_json(force=True)
    nl    = (body.get("description") or "").strip()
    scene = body.get("scene", "")
    if not nl:
        return jsonify({"status": "error", "message": "Empty description."}), 400

    from pipeline.main_pipeline import run
    def job():
        return run(nl, kb=kb(), verbose=True, auto_update_kb=False)
    result, logs, err = _run_capture(job)
    if err:
        return jsonify({"status": "error", "message": err, "logs": logs}), 500

    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"result": result, "scene": scene}
    return jsonify({"status": "success", "session_id": sid,
                    "data": _structured_view(result), "logs": logs})


# ════════════════════════════════════════════════════════════════════════════
#  Manual (step-by-step) mode — each pipeline stage exposed on its own endpoint.
#  The same pipeline functions used by /api/generate are simply called one step
#  at a time, so user edits to an earlier stage are passed into the next one.
# ════════════════════════════════════════════════════════════════════════════
# The six fields that ARE the "Formal Expression" — exactly what Stage-2 consumes.
_FORMAL_KEYS = ["problem_type", "problem_summary", "decision_variables",
                "parameters", "constraints", "objective"]


def _disambiguation_concepts(nl_text: str, matched_domain: str) -> list:
    """Web-display only: surface the RAG-FCA *semantic-disambiguation confidence*
    (the intent-similarity scores of the matched formal concepts, paper Eq. 15).

    These scores are produced by the framework's own retrieval — here we merely
    read them for display. We scope the retrieval to the domain Stage-1 already
    settled on (matched_domain) so the shown scores match that result. Purely
    structural (no LLM); pipeline logic is unchanged.
    """
    try:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            rag = kb().rag_fca_retrieve(
                nl_text, top_k=5,
                problem_type_filter=(matched_domain or None))
        return rag.get("matched_concepts", [])
    except Exception:
        return []


def _formal_view(structured: dict) -> dict:
    """Editable 6-field formal expression + read-only RAG-FCA context."""
    return {
        "formal_expr": {k: structured.get(k) for k in _FORMAL_KEYS},
        "aux": {
            "matched_domain":    structured.get("matched_domain", ""),
            "certain_implicit":  structured.get("certain_implicit", []),
            "possible_implicit": structured.get("possible_implicit", []),
            "formal_templates":  structured.get("formal_templates", []),
            # Semantic-disambiguation confidence (matched concepts + scores).
            "disambiguation":    structured.get("_disambiguation", []),
        },
    }


def _check_view(report: dict) -> dict:
    """Shape a validator report for the UI: the static/dynamic gates, the quality
    score, constraint coverage and the dynamic (solver) outcome, plus errors and
    suggestions. Used by both /api/check and the Stage-3 build so the same
    validation summary renders everywhere."""
    cov = report.get("constraint_coverage") or {}
    dyn = report.get("dynamic_result") or {}
    return {
        "score":          report.get("score", 0.0),
        "overall_valid":  report.get("overall_valid"),
        "syntax_ok":      report.get("syntax_ok"),
        "static_ok":      report.get("static_ok"),
        "dynamic_ok":     report.get("dynamic_ok"),
        "solver_status":  dyn.get("status"),
        "objective":      dyn.get("objective"),
        "coverage_ratio": cov.get("coverage_ratio"),
        "errors":         report.get("all_errors", []),
        "suggestions":    report.get("all_suggestions", []),
    }


@app.route("/api/formalize", methods=["POST"])
def api_formalize():
    """Manual step 1: NL → structured Formal Expression (Stage-1 only)."""
    body  = request.get_json(force=True)
    nl    = (body.get("description") or "").strip()
    scene = body.get("scene", "")
    sid   = body.get("session_id", "")
    if not nl:
        return jsonify({"status": "error", "message": "Empty description."}), 400

    from pipeline.nl_structuring import run as stage1
    def job():
        return stage1(nl, kb(), verbose=True)
    structured, logs, err = _run_capture(job)
    if err:
        return jsonify({"status": "error", "message": err, "logs": logs}), 500

    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex[:12]
    # Surface the semantic-disambiguation confidence for the Formal-Expression view.
    structured["_disambiguation"] = _disambiguation_concepts(
        nl, structured.get("matched_domain", ""))
    SESSIONS[sid] = {"scene": scene, "nl": nl, "structured": structured,
                     "math_model": None, "result": None}
    return jsonify({"status": "success", "session_id": sid,
                    "data": _formal_view(structured), "logs": logs})


@app.route("/api/build_model", methods=["POST"])
def api_build_model():
    """Manual step 2: (edited) Formal Expression → Mathematical Model (Stage-2)."""
    body = request.get_json(force=True)
    sid  = body.get("session_id", "")
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({"status": "error", "message": "Session not found."}), 404

    structured = dict(sess.get("structured") or {})
    edited = body.get("formal_expr")
    if isinstance(edited, dict):                 # merge user edits into structured
        for k in _FORMAL_KEYS:
            if k in edited:
                structured[k] = edited[k]
        sess["structured"] = structured          # persist edits for downstream

    from pipeline.math_model import run as stage2
    def job():
        return stage2(structured, kb(), verbose=True)
    math_model, logs, err = _run_capture(job)
    if err:
        return jsonify({"status": "error", "message": err, "logs": logs}), 500

    sess["math_model"] = math_model
    sess["result"] = None                        # code is now stale
    return jsonify({"status": "success", "data": math_model, "logs": logs})


@app.route("/api/build_code", methods=["POST"])
def api_build_code():
    """Manual step 3: (edited) Math Model + structured → CP code (Stage-3 loop)."""
    body = request.get_json(force=True)
    sid  = body.get("session_id", "")
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({"status": "error", "message": "Session not found."}), 404

    structured = dict(sess.get("structured") or {})
    math_model = body.get("math_model")
    if not isinstance(math_model, dict) or not math_model:
        math_model = sess.get("math_model") or {}
    sess["math_model"] = math_model              # persist edits for downstream

    # First-pass code generation only. Validation and any repair are driven
    # MANUALLY from the UI (Check button -> if issues, Fix button), so the user
    # controls the verification/iteration loop instead of it running silently.
    from pipeline.cp_code_gen import run as gen_code
    from pipeline.main_pipeline import _build_formal_language, _constraint_label
    def job():
        return gen_code(structured, math_model, kb(), verbose=True)
    final_code, logs, err = _run_capture(job)
    if err:
        return jsonify({"status": "error", "message": err, "logs": logs}), 500

    try:
        formal_lang = _build_formal_language(structured, math_model)
    except Exception as e:
        formal_lang = f"(formal language generation failed: {e})"
    label = structured.get("constraint_label") or _constraint_label(structured)

    # Assemble a complete result so Check / Fix / Run / Save AND dynamic
    # add_constraint all work on a manual session exactly as in auto mode.
    sess["result"] = {
        "task_id": "manual_" + sid,
        "input": {"natural_language": sess.get("nl", "")},
        "dynamic_updates": [],
        "pipeline_stages": {},
        "metadata": {},
        "final_output": {
            "problem_type":       structured.get("problem_type", "Unknown"),
            "problem_summary":    structured.get("problem_summary", ""),
            "cp_natural_language":sess.get("nl", ""),
            "cp_formal_language": formal_lang,
            "math_model":         math_model,
            "cp_code":            final_code,
            "validation_score":   None,          # set when the user runs Check / Fix
            "constraint_label":   label,
            "_structured":        structured,
        },
    }
    return jsonify({"status": "success",
                    "code": final_code,
                    "logs": logs})


@app.route("/api/add_constraint", methods=["POST"])
def api_add_constraint():
    body = request.get_json(force=True)
    sid  = body.get("session_id", "")
    nc   = (body.get("constraint") or "").strip()
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({"status": "error", "message": "Session not found."}), 404
    if not nc:
        return jsonify({"status": "error", "message": "Empty constraint."}), 400

    from pipeline.main_pipeline import add_constraint
    def job():
        return add_constraint(nc, sess["result"], kb=kb(),
                              verbose=True, auto_update_kb=False)
    result, logs, err = _run_capture(job)
    if err:
        return jsonify({"status": "error", "message": err, "logs": logs}), 500
    sess["result"] = result
    sess["structured"] = result["final_output"].get("_structured", {})
    sess["math_model"] = result["final_output"].get("math_model", {})
    fo = result["final_output"]
    # Keep the Formal-Expression disambiguation confidence in sync after the merge.
    sess["structured"]["_disambiguation"] = _disambiguation_concepts(
        fo.get("cp_natural_language") or sess.get("nl", ""),
        sess["structured"].get("matched_domain", ""))
    return jsonify({"status": "success",
                    "data": _structured_view(result),                 # auto-mode view
                    "formal_view": _formal_view(sess["structured"]),   # manual step 1 view
                    "math_model":  fo.get("math_model", {}),           # manual step 2 view
                    "code":        fo.get("cp_code", ""),              # manual step 3 view
                    "validation_score": fo.get("validation_score", 0.0),
                    "constraint_label": fo.get("constraint_label", ""),
                    "logs": logs})


@app.route("/api/generate_code", methods=["POST"])
def api_generate_code():
    """Return the current code (already produced by the pipeline)."""
    sid  = request.get_json(force=True).get("session_id", "")
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({"status": "error", "message": "Session not found."}), 404
    fo = sess["result"].get("final_output", {})
    return jsonify({"status": "success", "code": fo.get("cp_code", "")})


@app.route("/api/check", methods=["POST"])
def api_check():
    body = request.get_json(force=True)
    sid  = body.get("session_id", "")
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({"status": "error", "message": "Session not found."}), 404

    from pipeline.validator import validate
    fo = sess["result"]["final_output"]
    # Manual mode may edit the code in-place — honour the latest edited version.
    if isinstance(body.get("code"), str) and body["code"].strip():
        fo["cp_code"] = body["code"]
    def job():
        return validate(fo["cp_code"], fo["math_model"], fo["_structured"], verbose=True)
    report, logs, err = _run_capture(job)
    if err:
        return jsonify({"status": "error", "message": err, "logs": logs}), 500
    fo["validation_score"] = report.get("score", 0.0)
    return jsonify({"status": "success", "logs": logs, "report": _check_view(report)})


@app.route("/api/fix", methods=["POST"])
def api_fix():
    """Manual repair: validate the current code, build the feedback message from
    its errors (plus a runtime/solver error if dynamic execution is available),
    ask the fix pass to repair it, and return the new code + validation report.
    ONE round per click — the user drives the iteration (Check -> Fix -> Check)."""
    body = request.get_json(force=True)
    sid  = body.get("session_id", "")
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({"status": "error", "message": "Session not found."}), 404

    fo = sess["result"]["final_output"]
    if isinstance(body.get("code"), str) and body["code"].strip():
        fo["cp_code"] = body["code"]
    structured = fo["_structured"]
    math_model = fo["math_model"]

    from pipeline.validator import validate
    from pipeline.feedback import _feedback_msg, _run_dynamic
    from pipeline.cp_code_gen import run as gen_code

    def _revalidate(c):
        rep = validate(c, math_model, structured, verbose=True)
        dyn = None
        if rep.get("syntax_ok") and rep.get("static_ok"):
            dyn = _run_dynamic(c, structured, verbose=True)
        rep["dynamic_result"] = dyn
        rep["dynamic_ok"]     = (dyn or {}).get("dynamic_ok") if dyn else None
        return rep, dyn

    def job():
        code = fo["cp_code"]
        report, dyn = _revalidate(code)
        extra = (dyn or {}).get("error", "") if dyn and dyn.get("kind") == "runtime_error" else ""
        feedback = _feedback_msg(report, code, kb().get_possible_constraints(), extra)
        new_code = gen_code(structured, math_model, kb(), verbose=True,
                            feedback=feedback, current_code=code)
        new_report, _ = _revalidate(new_code)
        return new_code, new_report
    out, logs, err = _run_capture(job)
    if err:
        return jsonify({"status": "error", "message": err, "logs": logs}), 500
    new_code, new_report = out
    fo["cp_code"]         = new_code
    fo["validation_score"] = new_report.get("score", 0.0)
    return jsonify({"status": "success", "code": new_code,
                    "report": _check_view(new_report), "logs": logs})


@app.route("/api/solve", methods=["POST"])
def api_solve():
    body = request.get_json(force=True)
    sid  = body.get("session_id", "")
    tl   = int(body.get("time_limit", 60))
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({"status": "error", "message": "Session not found."}), 404

    from pipeline.solver import execute_code, check_data_files
    fo = sess["result"]["final_output"]
    if isinstance(body.get("code"), str) and body["code"].strip():
        fo["cp_code"] = body["code"]
    ptype = fo.get("problem_type", "")
    def job():
        ok, files = check_data_files(ptype, DATA_DIR)
        print(f"[Solve] data dir: {DATA_DIR}")
        print(f"[Solve] required data files present: {ok} ({files})")
        return execute_code(fo["cp_code"], DATA_DIR, problem_type=ptype, time_limit=tl)
    res, logs, err = _run_capture(job)
    if err:
        return jsonify({"status": "error", "message": err, "logs": logs}), 500
    return jsonify({"status": "success", "logs": logs, "result": {
        "status": res.get("status"),
        "objective": res.get("objective"),
        "solve_time": res.get("solve_time"),
        "solution_text": res.get("solution_text", ""),
        "error_msg": res.get("error_msg", "") or res.get("stderr", "")[:2000],
    }})


@app.route("/api/save_kb", methods=["POST"])
def api_save_kb():
    body = request.get_json(force=True)
    sid  = body.get("session_id", "")
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({"status": "error", "message": "Session not found."}), 404
    fo = sess["result"]["final_output"]
    if isinstance(body.get("code"), str) and body["code"].strip():
        fo["cp_code"] = body["code"]
    rec = {
        "problem_type":    fo.get("problem_type", "Unknown"),
        "nl_text":         fo.get("cp_natural_language", ""),
        "cp_code":         fo.get("cp_code", ""),
        "constraint":      fo.get("constraint_label", ""),
        "constraint_desc": fo.get("cp_formal_language", ""),
        "math_model":      fo.get("math_model", {}),
    }
    def job():
        return kb().add_record_interactive(rec)
    out, logs, err = _run_capture(job)
    if err:
        return jsonify({"status": "error", "message": err, "logs": logs}), 500
    added, reason = out
    info = kb().summary()
    return jsonify({"status": "success", "added": added, "reason": reason,
                    "records": info["total_records"],
                    "concepts": info["total_concepts"], "logs": logs})


if __name__ == "__main__":
    print("Loading knowledge base...")
    kb()
    print("CPAM-LLM web app ready → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)