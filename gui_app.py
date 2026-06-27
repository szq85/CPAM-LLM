from __future__ import annotations
import os, sys, queue, threading, traceback, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ── pipeline imports (done lazily inside worker to show errors in-GUI) ───────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ════════════════════════════════════════════════════════════════════════════
#  Console redirection: capture print() from the pipeline into the GUI
# ════════════════════════════════════════════════════════════════════════════
class _QueueWriter:
    def __init__(self, q): self.q = q
    def write(self, s):
        if s:
            self.q.put(("log", s))
    def flush(self): pass


# ════════════════════════════════════════════════════════════════════════════
#  Main application
# ════════════════════════════════════════════════════════════════════════════
class CpamApp:
    SAMPLES = {
        "Aircraft Skin Processing":
            "We have a set of jobs that must be processed in a specific sequence by a set of "
            "robots. Each operation can be assigned to different machines with varying processing "
            "times. Each machine processes one operation at a time. There are temporal constraints "
            "between operations: tight (<=15) and loose (>=30). Minimize the maximum completion time. "
            "Read input from data file 'fjsp_location_newdataset_1.fjs' in the data subdirectory.",
        "DNA Sequence Design":
            "Design a set of synthetic DNA nanostructure sequences. Each must satisfy Watson-Crick "
            "base pairing, GC content of 50%, sequence diversity (Hamming distance >= 4), and reverse "
            "complementarity. Obtain six distinct sequences. It is a feasibility problem.",
        "VRP":
            "We have a fleet of unmanned vehicles serving customers from a central depot with limited "
            "capacity. Minimize total distance while each customer is visited exactly once and vehicle "
            "capacities are not exceeded.",
        "Battery Pack Design":
            "We have a reconfigurable photovoltaic energy storage system of parallel battery modules "
            "and series columns. Choose the activated modules per period to minimize total net energy "
            "loss while meeting parallel activation consistency, voltage range, and max module current.",
        "Charging Station Location":
            "Select which charging stations to open from candidate sites and allocate demand areas to "
            "them, minimizing total construction plus travel cost, ensuring each area is covered and no "
            "station exceeds its capacity.",
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CPAM-LLM  —  Constraint-Programming Auto Modeling")
        self.root.geometry("1180x780")

        self.kb = None
        self.current = None            # latest pipeline result dict
        self.q: "queue.Queue" = queue.Queue()
        self.busy = False

        self._build_ui()
        self._poll_queue()
        self._async(self._load_kb, "Loading knowledge base...")

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style()
        try: style.theme_use("clam")
        except tk.TclError: pass

        # Top: input area
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Problem description:",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)

        self.input_txt = scrolledtext.ScrolledText(top, height=6, wrap=tk.WORD,
                                                    font=("Consolas", 10))
        self.input_txt.pack(fill=tk.X, pady=(2, 6))

        row = ttk.Frame(top); row.pack(fill=tk.X)
        ttk.Label(row, text="Sample:").pack(side=tk.LEFT)
        self.sample_var = tk.StringVar()
        cb = ttk.Combobox(row, textvariable=self.sample_var, state="readonly",
                          values=list(self.SAMPLES.keys()), width=28)
        cb.pack(side=tk.LEFT, padx=4)
        cb.bind("<<ComboboxSelected>>", self._load_sample)

        self.btn_run = ttk.Button(row, text="▶  Generate", command=self.on_generate)
        self.btn_run.pack(side=tk.LEFT, padx=4)
        self.btn_solve = ttk.Button(row, text="⚙  Solve", command=self.on_solve, state=tk.DISABLED)
        self.btn_solve.pack(side=tk.LEFT, padx=4)
        self.btn_check = ttk.Button(row, text="✓  Check code", command=self.on_check, state=tk.DISABLED)
        self.btn_check.pack(side=tk.LEFT, padx=4)
        self.btn_save = ttk.Button(row, text="💾  Save to KB", command=self.on_save, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=4)

        # Add-constraint row
        row2 = ttk.Frame(top); row2.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(row2, text="New constraint:").pack(side=tk.LEFT)
        self.constraint_var = tk.StringVar()
        self.constraint_entry = ttk.Entry(row2, textvariable=self.constraint_var)
        self.constraint_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.btn_add = ttk.Button(row2, text="＋  Add constraint",
                                  command=self.on_add_constraint, state=tk.DISABLED)
        self.btn_add.pack(side=tk.LEFT)

        # Middle: notebook with result tabs
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.tab_struct = self._make_text_tab("1 · Structured")
        self.tab_fca    = self._make_text_tab("2 · RAG-FCA")
        self.tab_model  = self._make_text_tab("3 · Math model")
        self.tab_code   = self._make_text_tab("4 · CP code")
        self.tab_solve  = self._make_text_tab("Solve result")
        self.tab_log    = self._make_text_tab("Console")

        # Bottom: status bar
        self.status = tk.StringVar(value="Initializing...")
        bar = ttk.Frame(self.root, padding=(8, 2)); bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bar, textvariable=self.status).pack(side=tk.LEFT)
        self.pb = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.pb.pack(side=tk.RIGHT)

    def _make_text_tab(self, title: str) -> scrolledtext.ScrolledText:
        frame = ttk.Frame(self.nb)
        txt = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 10))
        txt.pack(fill=tk.BOTH, expand=True)
        self.nb.add(frame, text=title)
        txt._nb_frame = frame          # remember the notebook child frame
        return txt

    # ── small helpers ─────────────────────────────────────────────────────────
    def _set(self, widget: scrolledtext.ScrolledText, text: str):
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)

    def _log(self, text: str):
        self.tab_log.insert(tk.END, text)
        self.tab_log.see(tk.END)

    def _busy(self, on: bool, msg: str = ""):
        self.busy = on
        state = tk.DISABLED if on else tk.NORMAL
        self.btn_run.config(state=state)
        has = self.current is not None
        for b in (self.btn_solve, self.btn_check, self.btn_save, self.btn_add):
            b.config(state=(tk.DISABLED if on or not has else tk.NORMAL))
        if on:
            self.pb.start(12); self.status.set(msg or "Working...")
        else:
            self.pb.stop(); self.status.set(msg or "Ready.")

    def _async(self, fn, busy_msg: str):
        """Run fn() on a background thread; capture its prints into the console."""
        if self.busy:
            messagebox.showinfo("Busy", "Please wait for the current task to finish.")
            return
        self._busy(True, busy_msg)

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = _QueueWriter(self.q)
            try:
                fn()
            except Exception:
                self.q.put(("log", "\n[ERROR]\n" + traceback.format_exc()))
                self.q.put(("error", "Task failed — see Console tab."))
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                self.q.put(("done", None))
        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "render":
                    self._render(payload)
                elif kind == "render_solve":
                    self._set(self.tab_solve, payload)
                    self.nb.select(self.tab_solve._nb_frame)
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "error":
                    messagebox.showerror("CPAM-LLM", payload)
                elif kind == "info":
                    messagebox.showinfo("CPAM-LLM", payload)
                elif kind == "done":
                    self._busy(False)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    # ── KB loading ──────────────────────────────────────────────────────────
    def _load_kb(self):
        from rag_fca.knowledge_base import get_knowledge_base
        self.kb = get_knowledge_base()
        s = self.kb.summary()
        self.q.put(("status",
            f"KB ready: {s['total_records']} records · "
            f"{s['total_concepts']} concepts · {s['total_attributes']} attributes"))

    # ── actions ───────────────────────────────────────────────────────────────
    def _load_sample(self, _evt=None):
        key = self.sample_var.get()
        if key in self.SAMPLES:
            self._set(self.input_txt, self.SAMPLES[key])

    def on_generate(self):
        nl = self.input_txt.get("1.0", tk.END).strip()
        if not nl:
            messagebox.showwarning("CPAM-LLM", "Please enter a problem description.")
            return
        self.tab_log.delete("1.0", tk.END)

        def job():
            from pipeline.main_pipeline import run
            res = run(nl, kb=self.kb, verbose=True, auto_update_kb=False)
            self.current = res
            self.q.put(("render", res))
            self.q.put(("status", "Generation complete."))
        self._async(job, "Running pipeline (NL → model → code)...")

    def on_add_constraint(self):
        nc = self.constraint_var.get().strip()
        if not nc:
            messagebox.showwarning("CPAM-LLM", "Type the new constraint first.")
            return
        if not self.current:
            return

        def job():
            from pipeline.main_pipeline import add_constraint
            res = add_constraint(nc, self.current, kb=self.kb,
                                 verbose=True, auto_update_kb=False)
            self.current = res
            self.q.put(("render", res))
            self.q.put(("status", "Constraint added."))
        self._async(job, "Adding constraint...")

    def on_check(self):
        if not self.current:
            return

        def job():
            from pipeline.validator import validate
            fo = self.current["final_output"]
            report = validate(fo["cp_code"], fo["math_model"],
                              fo["_structured"], verbose=True)
            lines = [
                "Validation report",
                "=" * 60,
                f"Score          : {report.get('score', 0):.2f} / 1.00",
                f"Overall valid  : {report.get('overall_valid')}",
                f"Syntax OK      : {report.get('syntax_ok')}",
                f"Static OK      : {report.get('static_ok')}",
                "",
                "Issues:",
            ]
            errs = report.get("all_errors", []) or ["(none)"]
            lines += [f"  - {e}" for e in errs]
            sugg = report.get("all_suggestions", [])
            if sugg:
                lines += ["", "Suggestions:"] + [f"  → {s}" for s in sugg]
            self.q.put(("render_solve", "\n".join(lines)))
            self.q.put(("status", f"Check done — score {report.get('score',0):.2f}"))
        # render check output into the Solve-result tab
        self._async(self._wrap_solve_render(job), "Re-validating code...")

    def on_solve(self):
        if not self.current:
            return

        def job():
            from pipeline.solver import execute_code, check_data_files
            fo = self.current["final_output"]
            ptype = fo.get("problem_type", "")
            ok, files = check_data_files(ptype, DATA_DIR)
            print(f"[Solve] data dir: {DATA_DIR}")
            print(f"[Solve] required data files present: {ok}  ({files})")
            res = execute_code(fo["cp_code"], DATA_DIR, problem_type=ptype,
                               time_limit=60)
            out = [
                "Solve result",
                "=" * 60,
                f"Status     : {res.get('status')}",
                f"Objective  : {res.get('objective')}",
                f"Solve time : {res.get('solve_time')}s",
                "",
                "Solution:",
                res.get("solution_text", "") or "(none)",
            ]
            if res.get("status") == "error":
                out += ["", "Error:", res.get("error_msg", "") or res.get("stderr", "")[:1500]]
            self.q.put(("render_solve", "\n".join(out)))
            self.q.put(("status", f"Solve: {res.get('status')}"))
        self._async(self._wrap_solve_render(job), "Running CP solver...")

    def _wrap_solve_render(self, job):
        # job pushes ('render_solve', text); translate to tab write here
        def inner():
            job()
        return inner

    def on_save(self):
        if not self.current:
            return
        fo = self.current["final_output"]
        ptype = fo.get("problem_type", "Unknown")
        label = fo.get("constraint_label", "")
        score = fo.get("validation_score", 0.0)
        if not messagebox.askyesno(
            "Save to knowledge base",
            f"Save this result to the knowledge base?\n\n"
            f"Problem type : {ptype}\n"
            f"Constraints  : {label or '(base)'}\n"
            f"Valid score  : {score:.2f}\n\n"
            f"Only a genuinely new constraint set will be added; "
            f"exact duplicates are skipped."):
            self.q.put(("status", "Not saved — KB unchanged."))
            return

        def job():
            rec = {
                "problem_type":    ptype,
                "nl_text":         fo.get("cp_natural_language", ""),
                "cp_code":         fo.get("cp_code", ""),
                "constraint":      label,
                "constraint_desc": fo.get("cp_formal_language", ""),
                "math_model":      fo.get("math_model", {}),
            }
            added, reason = self.kb.add_record_interactive(rec)
            if added:
                msg = (f"Saved. {reason}\nKB now: {len(self.kb.records)} records, "
                       f"{len(self.kb.lattice.concepts)} concepts.")
                self.q.put(("info", msg))
            else:
                self.q.put(("info", f"Not saved: {reason}"))
            self.q.put(("status", "Save action complete."))
        self._async(job, "Saving to knowledge base...")

    # ── rendering ──────────────────────────────────────────────────────────────
    def _render(self, res: dict):
        fo = res.get("final_output", {})
        struct = fo.get("_structured", {})

        # 1 — structured
        lines = [f"Problem type : {fo.get('problem_type')}",
                 f"Summary      : {fo.get('problem_summary')}",
                 f"Constraint   : {fo.get('constraint_label')}",
                 f"Valid score  : {fo.get('validation_score', 0):.2f}",
                 "", "Decision variables:"]
        for v in struct.get("decision_variables", []):
            lines.append(f"  • {v.get('name')} [{v.get('type')}] — {v.get('description','')}")
        lines += ["", "Constraints:"]
        for c in struct.get("constraints", []):
            lines.append(f"  [{c.get('id')}] {c.get('type')} — {c.get('description','')}")
        obj = struct.get("objective", {})
        lines += ["", f"Objective    : {obj.get('direction','').upper()} {obj.get('expression','')}"]
        if struct.get("certain_implicit"):
            lines += ["", "Certain implicit (RAG-FCA): " + ", ".join(struct["certain_implicit"])]
        if struct.get("possible_implicit"):
            lines += ["Possible suggestions     : " + ", ".join(struct["possible_implicit"])]
        self._set(self.tab_struct, "\n".join(lines))

        # 2 — RAG-FCA concept retrieval / formal templates
        f = ["RAG-FCA structural retrieval (concept lattice)", "=" * 60]
        f.append(f"Matched domain : {struct.get('matched_domain','')}")
        for c in struct.get("_rag_concepts", []):
            f.append(f"  concept · type={c.get('problem_type')}  "
                     f"constraints={{{', '.join(c.get('constraints', [])[:8])}}}")
        f += ["", "Formal constraint templates (name · math · cp):"]
        for t in struct.get("formal_templates", []):
            f.append(f"  • {t.get('name')}")
            f.append(f"      desc: {t.get('desc','')}")
            f.append(f"      math: {t.get('math','')}")
            f.append(f"      cp  : {t.get('cp','')}")
        self._set(self.tab_fca, "\n".join(f))

        # 3 — math model
        mm = fo.get("math_model", {})
        m = [f"Model: {mm.get('model_title','')}", f"Type : {mm.get('model_type','')}", ""]
        m.append("Parameters:")
        for p in mm.get("parameters_section", []):
            m.append(f"  {p.get('symbol')} — {p.get('description','')}")
        m += ["", "Variables:"]
        for v in mm.get("variables_section", []):
            m.append(f"  {v.get('symbol')} [{v.get('var_type')}] — {v.get('description','')}")
        m += ["", "Constraints:"]
        for c in mm.get("constraints_section", []):
            m.append(f"  [{c.get('id')}] {c.get('name')}: {c.get('math_expression','')}")
        o = mm.get("objective_section", {})
        m += ["", f"Objective: {o.get('direction','').upper()} {o.get('math_expression','')}"]
        self._set(self.tab_model, "\n".join(m))

        # 4 — code
        self._set(self.tab_code, fo.get("cp_code", ""))

        # enable action buttons
        for b in (self.btn_solve, self.btn_check, self.btn_save, self.btn_add):
            b.config(state=tk.NORMAL)
        self.nb.select(0)


# ════════════════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    CpamApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
