from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import Counter

from config import (
    DATA_PATH, DATASET_SHEET, PROBLEM_TYPE_ATTRIBUTES,
    KB_UPDATE_POLICY, KB_UPDATE_SCORE_STRICT, KB_UPDATE_SCORE_RELAXED,
    KB_DEDUP_THRESHOLD, KB_REBUILD_BATCH,
    KB_JSON_PATH, KB_STORE_DIR,
)
from rag_fca.fca import (
    FCABuilder, FormalContext, ConceptLattice,
    extract_query_attributes, weighted_similarity,
    discover_implicit_constraints,
    learn_attribute_weights,
    FuzzyFormalContext, fuzzy_similarity, extract_query_memberships,
    Implication, FormalConcept,
)

_MAX_QUERY_LOG = 500
_WEIGHT_RELEARN_PERIOD = 20


class KnowledgeBase:
    def __init__(self):
        self.records:  List[Dict] = []
        self.context:  Optional[FormalContext] = None
        self.lattice:  Optional[ConceptLattice] = None
        self._builder  = FCABuilder()
        self._pending_since_rebuild = 0   # records added since last full rebuild

        self._attr_weights: Dict[str, float] = {}
        self._query_log: List[Tuple[frozenset, bool]] = []
        self._successful_retrievals: int = 0

        # One gold Formal Expression per base problem type.
        self.formal_expression_refs: Dict[str, Dict] = {}

    def load_formal_expression_refs(self, path: str = "") -> None:
        """
        Load the 5 canonical Formal-Expression references (one per base problem
        type) from data/formal_expression_refs.json into the knowledge base.

        Safe to call repeatedly; missing file just leaves the refs empty.
        """
        import json
        if not path:
            path = os.path.join(os.path.dirname(DATA_PATH), "formal_expression_refs.json")
        try:
            with open(path, encoding="utf-8") as f:
                self.formal_expression_refs = json.load(f)
            print(f"[KB] Loaded {len(self.formal_expression_refs)} canonical "
                  f"formal-expression references from {os.path.basename(path)}.")
        except Exception as e:
            print(f"[KB] ⚠ Could not load formal-expression references ({e}).")
            self.formal_expression_refs = {}

    def formal_expression_ref(self, problem_type: str) -> Dict:
        """Return the canonical Formal Expression for one problem type ({} if none)."""
        return self.formal_expression_refs.get(problem_type, {})

    def all_formal_expression_refs(self) -> Dict[str, Dict]:
        """Return all canonical Formal-Expression references (keyed by problem type)."""
        return self.formal_expression_refs

    def load_from_xlsx(self, path: str = DATA_PATH) -> None:
        df = pd.read_excel(path, sheet_name=DATASET_SHEET).fillna("")
        self.records = []
        for _, row in df.iterrows():
            self.records.append({
                "id":              str(row.get("Number", len(self.records))),
                "problem_type":    str(row.get("Problem types", "")).strip(),
                "nl_text":         str(row.get("CP natural language", "")).strip(),
                "cp_code":         str(row.get("CP code", "")).strip(),
                "constraint":      str(row.get("Constraint", "")).strip(),
                "constraint_desc": str(row.get("CP formal language", "")).strip(),
                "source":          "dataset",
            })
        self._full_rebuild()
        self.load_formal_expression_refs()
        print(f"[KB] Loaded {len(self.records)} records, "
              f"{len(self.lattice.concepts)} concepts, "
              f"{len(self.context.attributes)} attributes.")

    def _full_rebuild(self) -> None:
        """
        Rebuild formal context K=(G,M,I) and concept lattice L(K) from scratch.
        (Ganter & Wille 1999; Ganter 1984 NextClosure; Guigues & Duquenne 1986)

        Steps:
          1. Build crisp formal context K from KB records
          2. Enumerate ALL formal concepts via NextClosure (lectic order)
             so every C=(A,B) satisfies A'=B and B'=A exactly
          3. Compute canonical implication base (Duquenne-Guigues base)
             for query augmentation: Q_augmented = Q ∪ certain_implications(Q)
          4. Build Hasse diagram edges (direct cover relations)
        Capped at MAX_CONCEPTS to handle large KBs efficiently
        (Poelmans et al. 2013 §5.5 — scalability via concept stability / iceberg).
        """
        MAX_CONCEPTS = 2000   # scalability cap (Poelmans et al. 2013 §5.5)
        COMPUTE_IMPL = True   # always compute canonical implication base

        self.context = self._builder.build_context(self.records)
        self.lattice = self._builder.build_lattice(
            self.context,
            max_concepts=MAX_CONCEPTS,
            compute_implications=COMPUTE_IMPL,
            build_hasse=(len(self.records) <= 150),
        )
        self._pending_since_rebuild = 0
        self._attr_weights = {a: 1.0 for a in self.context.attributes}
        print(f"  [KB] {self.lattice.summary()}")

    # Store only fields the pipeline needs; load restores safe defaults.
    @staticmethod
    def _slim_record(r: Dict) -> Dict:
        keep = ("id", "problem_type", "nl_text", "cp_code", "constraint",
                "constraint_desc", "source")
        out = {}
        for k in keep:
            v = r.get(k, "")
            if k == "constraint_desc" and not v:      # always empty in dataset
                continue
            if k == "source" and v in ("", "dataset"):  # constant → drop
                continue
            if k == "cp_code" and not v:               # only base rows carry code
                continue
            out[k] = v
        for k, v in r.items():
            if k not in keep and k not in out:
                out[k] = v
        return out

    @staticmethod
    def _restore_record(r: Dict) -> Dict:
        r = dict(r)
        r.setdefault("nl_text", "")
        r.setdefault("cp_code", "")
        r.setdefault("constraint", "")
        r.setdefault("constraint_desc", "")
        r.setdefault("source", "dataset")
        return r

    def save_to_json(self, path: str = KB_JSON_PATH, source_xlsx: str = "") -> str:
        """
        Serialize the entire knowledge base — records + formal context K=(G,M,I)
        + concept lattice L(K) (concepts, canonical implication base, Hasse edges)
        + learned attribute weights — to a single human-readable JSON file.

        This file is the permanent KB: once built, main.py loads it directly and
        never needs the source xlsx again. The format is plain JSON so anyone can
        open it to inspect the concept lattice.
        """
        import os, json
        import numpy as np
        from datetime import datetime
        from rag_fca.formalizations import FORMAL as _FORMAL

        if not self.context or not self.lattice:
            raise RuntimeError("Nothing to save — build the lattice first "
                               "(load_from_xlsx).")

        ctx, lat = self.context, self.lattice

        concepts_json = [
            {
                "extent": sorted(c.extent),
                "intent": sorted(c.intent),
                "label":  c.label,
                "level":  int(c.level),
            }
            for c in lat.concepts
        ]
        implications_json = [
            {"premise": sorted(im.premise), "conclusion": sorted(im.conclusion)}
            for im in lat.implications
        ]
        hasse_json = [[int(a), int(b)] for (a, b) in lat.hasse_edges]

        store = {
            "meta": {
                "format_version": "1.0",
                "description": (
                    "CPAM-LLM permanent knowledge base. Contains the formal "
                    "context K=(G,M,I) and the concept lattice L(K) built via FCA "
                    "(NextClosure). Built once by build_kb.py and loaded by main.py."
                ),
                "created_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_xlsx":  os.path.basename(source_xlsx) if source_xlsx else "",
                "n_records":    len(self.records),
                "n_objects":    len(ctx.objects),
                "n_attributes": len(ctx.attributes),
                "n_concepts":   len(lat.concepts),
                "n_implications": len(lat.implications),
                "n_hasse_edges":  len(lat.hasse_edges),
            },
            "records": [self._slim_record(r) for r in self.records],
            "formal_context": {
                "objects":    list(ctx.objects),
                "attributes": list(ctx.attributes),
                # |G| x |M| incidence matrix, stored as 0/1 ints for readability
                "relation":   ctx.relation.astype(int).tolist(),
            },
            "attribute_weights": {k: float(v) for k, v in self._attr_weights.items()},
            # Formal representation of every attribute (constraint TYPE / objective):
            # name -> {desc, math, cp}. Stored so the concept-lattice knowledge
            # base is self-contained and FORMAL — the lattice intents are labels,
            # this is what makes the formal-model → code stage possible.
            "attribute_formalizations": {
                a: _FORMAL[a] for a in ctx.attributes if a in _FORMAL
            },
            "concept_lattice": {
                "concepts":     concepts_json,
                "implications": implications_json,
                "hasse_edges":  hasse_json,
            },
        }

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
        print(f"[KB] Saved knowledge base → {path}  "
              f"({len(self.records)} records, {len(lat.concepts)} concepts, "
              f"{len(ctx.attributes)} attributes)")
        return path

    def load_from_json(self, path: str = KB_JSON_PATH) -> None:
        """
        Reconstruct the knowledge base from the permanent JSON store WITHOUT
        recomputing the lattice. main.py uses this so it can start instantly and
        without the source xlsx present.
        """
        import os, json
        import numpy as np

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Knowledge-base JSON not found:\n    {path}\n"
                f"Build it first:  python build_kb.py"
            )

        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)

        self.records = [self._restore_record(r) for r in store["records"]]

        fc = store["formal_context"]
        relation = np.array(fc["relation"], dtype=bool) if fc["objects"] else \
            np.zeros((0, len(fc["attributes"])), dtype=bool)
        self.context = FormalContext(
            objects=list(fc["objects"]),
            attributes=list(fc["attributes"]),
            relation=relation,
            object_data={str(r["id"]): r for r in self.records},
        )

        cl = store["concept_lattice"]
        concepts = [
            FormalConcept(
                extent=frozenset(c["extent"]),
                intent=frozenset(c["intent"]),
                label=c.get("label", ""),
                level=int(c.get("level", 0)),
            )
            for c in cl["concepts"]
        ]
        implications = [
            Implication(
                premise=frozenset(im["premise"]),
                conclusion=frozenset(im["conclusion"]),
            )
            for im in cl["implications"]
        ]
        hasse_edges = [tuple(e) for e in cl["hasse_edges"]]
        self.lattice = ConceptLattice(
            concepts=concepts,
            implications=implications,
            hasse_edges=hasse_edges,
            context=self.context,
        )

        self._attr_weights = {k: float(v)
                              for k, v in store.get("attribute_weights", {}).items()}
        if not self._attr_weights:
            self._attr_weights = {a: 1.0 for a in self.context.attributes}
        self._pending_since_rebuild = 0

        self.load_formal_expression_refs()

        meta = store.get("meta", {})
        print(f"[KB] Loaded knowledge base from JSON → {path}")
        print(f"  [KB] {len(self.records)} records, "
              f"{len(self.lattice.concepts)} concepts, "
              f"{len(self.context.attributes)} attributes  "
              f"(built {meta.get('created_at', '?')})")

    def build_and_save(
        self,
        xlsx_path: str = DATA_PATH,
        json_path: str = KB_JSON_PATH,
        export_review: bool = True,
    ) -> str:
        """
        One-shot KB construction used by build_kb.py:
          1. Load records from the (clean) source xlsx.
          2. Build the formal context + concept lattice via FCA.
          3. Persist everything to the permanent JSON store.
          4. Optionally also write the human-readable FCA review files
             (kb_formal_concepts.json, kb_implication_base.json, ...).
        """
        self.load_from_xlsx(xlsx_path)
        out = self.save_to_json(json_path, source_xlsx=xlsx_path)
        if export_review:
            review_dir = os.path.join(KB_STORE_DIR, "review")
            self.export_to_file(output_dir=review_dir)
        return out

    # ── RAG-FCA Retrieval ─────────────────────────────────────
    def retrieve(
        self,
        query_text: str,
        top_k: int = 3,
        threshold: float = 0.10,
        problem_type_filter: Optional[str] = None,
        return_suggestions: bool = False,
    ) -> List[Dict]:
        """
        RAG-FCA retrieval (Paper Eq. 15):
          1. Extract attribute set Q' from query text.
          2. Augment with certain implicit constraints via rough-set approximation (Eq. 22).
          3. Score each record by weighted Jaccard similarity (Eq. 15).
          4. Return top-k above threshold.

        Possible constraints (Eq. 21) are stored in self._last_possible_constraints
        for optional surfacing to the user.

        Args:
            return_suggestions: If True, also log possible constraints for display.
        """
        q_attrs = extract_query_attributes(query_text, self.context.attributes)

        # Query augmentation via implication base (Guigues & Duquenne 1986):
        # If implications U→V are known and U ⊆ q_attrs, add V to the query.
        # This is LinClosure(q_attrs, implications), matching Eq.22 semantics.
        q_impl_aug = set(q_attrs)
        if self.lattice.implications:
            changed = True
            while changed:
                changed = False
                for impl in self.lattice.implications:
                    if impl.premise <= q_impl_aug and not impl.conclusion <= q_impl_aug:
                        q_impl_aug |= impl.conclusion
                        changed = True
        q_attrs = frozenset(q_impl_aug)

        # Rough-set discovery (Eq. 19-22) runs on the EXPLICIT seed (no base
        # injection); otherwise the lower approximation (certain) is empty by
        # construction. Retrieval/augmentation still use the full q_attrs above.
        q_seed = frozenset(extract_query_attributes(
            query_text, self.context.attributes, include_base=False))
        certain, possible = discover_implicit_constraints(q_seed, self.lattice, self.context)
        q_aug = q_attrs | certain
        # Cache possible constraints for surfacing to user (Eq. 21)
        self._last_possible_constraints = sorted(possible)
        self._last_certain_constraints  = sorted(certain)
        # Cache the full augmented constraint set (explicit ∪ certain) so the
        # model/code stages can be handed FORMAL templates for each constraint.
        self._last_augmented_constraints = sorted(q_aug)

        scored = []
        for rec in self.records:
            if problem_type_filter and rec["problem_type"] != problem_type_filter:
                continue
            rec_id = rec["id"]
            if rec_id not in self.context.objects:
                continue
            obj_i    = self.context.objects.index(rec_id)
            rec_attrs = frozenset(
                a for a, v in zip(self.context.attributes, self.context.relation[obj_i]) if v
            )
            sim = weighted_similarity(q_aug, rec_attrs, weights=self._attr_weights)
            if sim >= threshold:
                scored.append((sim, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [r for _, r in scored[:top_k]]

        print(f"[RAG-FCA] Query attrs: {sorted(q_attrs)[:6]}... "
              f"| Certain implicit: {sorted(certain)[:4]}... "
              f"| Retrieved: {len(results)}")
        if possible:
            print(f"[RAG-FCA] Possible constraints (suggestions): {sorted(possible)[:6]}")

        # Log query for weight learning
        if results:
            self._query_log.append((q_aug, True))   # mark success tentatively
            self._successful_retrievals += 1
            if self._successful_retrievals % _WEIGHT_RELEARN_PERIOD == 0:
                self._relearn_weights()
        # Prune log
        if len(self._query_log) > _MAX_QUERY_LOG:
            self._query_log = self._query_log[-_MAX_QUERY_LOG:]

        return results

    def get_possible_constraints(self) -> List[str]:
        """
        Return the possible constraints from the last retrieve() call.
        These are candidate implicit constraints not yet confirmed (Eq. 21),
        presented to the user as interactive suggestions for query refinement.
        """
        return getattr(self, "_last_possible_constraints", [])

    def get_certain_constraints(self) -> List[str]:
        """Return the certain implicit constraints added to the last augmented query (Eq. 22)."""
        return getattr(self, "_last_certain_constraints", [])

    # ── Formal representations (concept-lattice attributes → CP templates) ────
    def formalize(self, attrs: List[str]) -> List[Dict]:
        """Map attribute names to their formal {name,desc,math,cp} templates."""
        from rag_fca import formalizations as _fm
        return _fm.formalize(attrs)

    def get_constraint_templates(self) -> List[Dict]:
        """Formal templates for the FULL augmented constraint set of the last
        retrieve() (explicit + certain implicit). This is what the second stage
        (formal model → CP code) consumes."""
        return self.formalize(getattr(self, "_last_augmented_constraints", []))

    def get_certain_constraint_templates(self) -> List[Dict]:
        """Formal templates for just the certain implicit constraints (Eq. 22)."""
        return self.formalize(getattr(self, "_last_certain_constraints", []))

    def templates_prompt_block(self, header: str = "Formal constraint templates") -> str:
        """Render the augmented constraint set as a formal prompt block."""
        from rag_fca import formalizations as _fm
        return _fm.as_prompt_block(
            getattr(self, "_last_augmented_constraints", []), header=header)

    def domain_knowledge_base(self, problem_type: str) -> Dict:
        """
        Return the FORMALIZED per-domain knowledge base for one scenario — the
        same structure exported to kb_store/review/kb_domain_lattices.json, built
        live from the loaded concept lattice. This is the per-scenario view that
        matches the paper's Fig 2b–6b: anchor, objective, base + additional
        constraints (each with desc/math/cp), the domain's concepts (intents
        split and formalized), and the domain-internal implications.
        """
        from rag_fca import constraints as _con
        from rag_fca import formalizations as _fm
        spec = _con.CATALOG.get(problem_type)
        if not spec:
            return {}
        anchor = spec["anchor"]

        def _detail(name):
            f = _fm.FORMAL.get(name, {})
            return {"name": name, "desc": f.get("desc", ""),
                    "math": f.get("math", ""), "cp": f.get("cp", "")}

        domain_attrs = ({anchor, spec["objective"]} | set(spec["base"]) |
                        {o["name"] for o in spec["optional"]})

        concepts = []
        for idx, c in enumerate(self.lattice.concepts):
            if anchor not in c.intent:
                continue
            concepts.append({
                "concept_index": idx,
                "extent_size":   len(c.extent),
                "objective":     [_detail(spec["objective"])] if spec["objective"] in c.intent else [],
                "base_constraints":      [_detail(a) for a in spec["base"] if a in c.intent],
                "additional_constraints":[_detail(o["name"]) for o in spec["optional"]
                                          if o["name"] in c.intent],
            })
        concepts.sort(key=lambda x: -x["extent_size"])

        impls = []
        for im in self.lattice.implications:
            names = set(im.premise) | set(im.conclusion)
            if names <= domain_attrs and (anchor in names or
               names & set(spec["base"]) or
               names & {o["name"] for o in spec["optional"]}):
                impls.append({"premise": sorted(im.premise),
                              "conclusion": sorted(im.conclusion)})

        return {
            "scenario":  problem_type,
            "anchor":    _detail(anchor),
            "objective": _detail(spec["objective"]),
            "base_constraints":       [_detail(a) for a in spec["base"]],
            "additional_constraints": [_detail(o["name"]) for o in spec["optional"]],
            "concepts":     concepts,
            "implications": impls,
        }

    def base_template(self, problem_type: str) -> str:
        """
        Return the base-scenario CP code skeleton (the seed model) for a problem
        type. Used by the code stage purely as a docplex API reference, NOT as a
        'retrieved example paragraph' to copy.
        """
        for r in self.records:
            if (r.get("problem_type") == problem_type and
                    str(r.get("constraint", "")).strip().lower() == "base" and
                    r.get("cp_code")):
                return r["cp_code"]
        for r in self.records:               # fallback: any coded record of type
            if r.get("problem_type") == problem_type and r.get("cp_code"):
                return r["cp_code"]
        return ""

    def rag_fca_retrieve(self, query_text: str, top_k: int = 3,
                         problem_type_filter: Optional[str] = None) -> Dict:
        """
        Concept-lattice RAG-FCA retrieval (Paper Eq. 15-22, Fig 1b).

        This is the *structural* retrieval the paper describes — and the reason
        FCA is used instead of plain semantic search:

          1. Extract the query's constraint attributes Q' and close them under
             the implication base (Eq. 22 LinClosure).
          2. Score every concept by Eq. 15  score(Q,B)=Σ_{m∈Q∩B}w / Σ_{m∈Q∪B}w
             against its INTENT B (a constraint set) — not against raw text.
          3. Take the concept (A*,B*) with maximal overlap (Eq. 17-18) and
             augment the query with its full constraint set.
          4. Add rough-set certain / possible constraints (Eq. 19-22).
          5. Resolve every resulting constraint to its FORMAL template.

        Returns the formalized constraint structure
            Q_aug = Q' ∪ B* ∪ Constraints(Q')
        plus formal templates and (only as references) the example problem IDs in
        the matched concept's extent. It deliberately returns NO natural-language
        paragraphs or code blobs.
        """
        from collections import Counter
        from rag_fca import formalizations as _fm
        from rag_fca.fca import detect_scenario_strong

        effective_filter = problem_type_filter
        if not effective_filter:
            _strong = detect_scenario_strong(query_text)
            if _strong:
                effective_filter = _strong

        q_attrs = set(extract_query_attributes(
            query_text, self.context.attributes,
            force_scenario=effective_filter or None))
        changed = True
        while changed and self.lattice.implications:
            changed = False
            for impl in self.lattice.implications:
                if impl.premise <= q_attrs and not impl.conclusion <= q_attrs:
                    q_attrs |= impl.conclusion
                    changed = True
        q_attrs = frozenset(q_attrs)

        # Rough-set implicit-constraint discovery (paper Alg., Eq. 19-22) must run
        # on the *explicitly stated* query attributes — NOT on the base-injected /
        # implication-closed q_attrs used for retrieval. Feeding it q_attrs makes
        # the lower approximation (certain constraints) empty by construction,
        # because the scenario's base constraints — exactly what the lower
        # approximation is meant to discover — are already in q_attrs. We therefore
        # build an explicit seed Q' (anchor + objective + matched optional
        # signatures, WITHOUT the base set) and discover certain/possible from it.
        q_seed = frozenset(extract_query_attributes(
            query_text, self.context.attributes,
            force_scenario=effective_filter or None, include_base=False))
        certain, possible = discover_implicit_constraints(
            q_seed, self.lattice, self.context)

        n_obj = len(self.context.objects)
        scored = []
        for c in self.lattice.concepts:
            if not c.intent or len(c.extent) == 0 or len(c.extent) == n_obj:
                continue
            s = weighted_similarity(q_attrs, c.intent, weights=self._attr_weights)
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])

        best_B = frozenset()
        if scored:
            best_c = max(scored, key=lambda sc: len(q_attrs & sc[1].intent))[1]
            best_B = best_c.intent

        # Q_augmented = Q' ∪ certain (paper Eq. 22); base is already in q_attrs, so
        # the augmented set is unchanged by surfacing certain explicitly.
        q_aug = set(q_attrs) | set(best_B) | set(certain)

        self._last_possible_constraints  = sorted(possible)
        self._last_certain_constraints   = sorted(certain)
        self._last_augmented_constraints = sorted(q_aug)

        def _dom(extent):
            d = Counter(self.context.object_data.get(g, {}).get("problem_type", "")
                        for g in extent)
            return d.most_common(1)[0][0] if d else ""

        matched = [{
            "score":        round(float(s), 3),
            "problem_type": _dom(c.extent),
            "constraints":  sorted(c.intent),
            "extent_size":  len(c.extent),
            "example_ids":  sorted(c.extent, key=lambda x: int(x) if str(x).isdigit() else 0)[:5],
        } for s, c in scored[:top_k]]

        if effective_filter:
            _filtered = [m for m in matched
                         if m["problem_type"] == effective_filter]
            if not _filtered:
                for s, c in scored:
                    if _dom(c.extent) == effective_filter:
                        _filtered.append({
                            "score":        round(float(s), 3),
                            "problem_type": effective_filter,
                            "constraints":  sorted(c.intent),
                            "extent_size":  len(c.extent),
                            "example_ids":  sorted(c.extent,
                                key=lambda x: int(x) if str(x).isdigit() else 0)[:5],
                        })
                        if len(_filtered) >= top_k:
                            break
            if _filtered:
                matched = _filtered

        dom = effective_filter or (matched[0]["problem_type"] if matched else "")

        result = {
            "query_constraints":     sorted(q_attrs),
            "matched_concepts":      matched,
            "certain_constraints":   sorted(certain),
            "possible_constraints":  sorted(possible),
            "augmented_constraints": sorted(q_aug),
            "formal_templates":      _fm.formalize(sorted(q_aug)),
            "matched_domain":        dom,
            "domain_knowledge_base": self.domain_knowledge_base(dom) if dom else {},
            "base_template":         self.base_template(dom) if dom else "",
        }
        print(f"[RAG-FCA] Q'={sorted(q_attrs)[:5]} | concept B*={sorted(best_B)[:5]} "
              f"| +certain={sorted(certain)[:3]} | formal templates={len(result['formal_templates'])}")
        return result

    def _relearn_weights(self) -> None:
        """
        Periodically relearn attribute importance weights from query log.
        (Paper §RAG-FCA: w(m) learned from historical query logs)
        """
        if not self._query_log or not self.context:
            return
        new_weights = learn_attribute_weights(
            self._query_log,
            self.context.attributes,
            prior_weight=1.0,
            positive_boost=0.15,
            negative_decay=0.05,
        )
        self._attr_weights = new_weights
        print(f"[KB] Attribute weights relearned from {len(self._query_log)} query log entries.")

    def mark_retrieval_failed(self, query_text: str) -> None:
        """
        Mark the most recent retrieval as failed (used for weight learning).
        Call this when a retrieval led to validation failure.
        """
        q_attrs = extract_query_attributes(query_text, self.context.attributes)
        certain, _ = discover_implicit_constraints(q_attrs, self.lattice, self.context)
        q_aug = q_attrs | certain
        if self._query_log:
            # Replace last entry with a failure marker
            self._query_log[-1] = (q_aug, False)

    # ── Update policy check ───────────────────────────────────
    def should_update(self, val_report: Dict) -> Tuple[bool, str]:
        """
        Returns (should_add: bool, reason: str) based on KB_UPDATE_POLICY.
        val_report is the dict returned by pipeline/validator.validate().
        """
        policy = KB_UPDATE_POLICY
        score  = val_report.get("score", 0.0)
        syntax = val_report.get("syntax_ok", False)
        static = val_report.get("static_ok", False)

        if policy == "manual":
            return False, "manual policy — skipping auto-update"

        if policy == "strict":
            if score >= KB_UPDATE_SCORE_STRICT:
                return True, f"strict policy passed (score {score:.2f} ≥ {KB_UPDATE_SCORE_STRICT})"
            return False, f"strict policy — score {score:.2f} < {KB_UPDATE_SCORE_STRICT}"

        if policy == "relaxed":
            if syntax and static and score >= KB_UPDATE_SCORE_RELAXED:
                return True, f"relaxed policy passed (score {score:.2f} ≥ {KB_UPDATE_SCORE_RELAXED})"
            return False, (f"relaxed policy — syntax={syntax} static={static} "
                           f"score={score:.2f} < {KB_UPDATE_SCORE_RELAXED}")

        return False, f"unknown policy '{policy}'"

    # ── Incremental Update (Eq. 23-24) ───────────────────────
    def add_record(self, record: Dict, save_path: str = KB_JSON_PATH) -> str:
        """
        Add a new verified record to the KB.
        Implements incremental formal context update (Eq. 23-24):
          K_{t+1} = (G_t ∪ {g_new}, M_t ∪ M_new, I_t ∪ I_new)

        Lattice rebuild strategy:
          • If KB_REBUILD_BATCH == 0 → full rebuild every time (safest, slowest).
          • If pending count < KB_REBUILD_BATCH → fast incremental attribute merge.
          • When pending count reaches KB_REBUILD_BATCH → trigger full rebuild.
        """
        existing_ids = {int(r["id"]) for r in self.records if r["id"].isdigit()}
        # Non-digit IDs (e.g. "task_20260528_...") are excluded from the numeric
        # pool; fall back to record count when no numeric IDs exist so we never
        # produce a spurious gap in the ID sequence.
        if existing_ids:
            new_id = str(max(existing_ids) + 1)
        else:
            new_id = str(len(self.records))  # 0-based count is the new index
        record["id"]     = new_id
        record["source"] = record.get("source", "generated")
        self.records.append(record)

        self._pending_since_rebuild += 1
        if KB_REBUILD_BATCH == 0 or self._pending_since_rebuild >= KB_REBUILD_BATCH:
            # Full rebuild: guarantees lattice correctness (Eq. 23)
            self._full_rebuild()
            print(f"[KB] Full lattice rebuild — "
                  f"{len(self.lattice.concepts)} concepts, "
                  f"{len(self.context.attributes)} attributes.")
        else:
            # Fast path: incremental context extension without full lattice rebuild
            self._incremental_extend(record, new_id)
            print(f"[KB] Incremental update (Eq. 24) — "
                  f"next full rebuild in "
                  f"{KB_REBUILD_BATCH - self._pending_since_rebuild} record(s).")

        # Persist the updated KB back to the permanent JSON store.
        try:
            self.save_to_json(save_path)
            print(f"[KB] Record ID={new_id} persisted → {save_path}")
        except Exception as e:
            print(f"[KB] ⚠  Persist warning: {e}")

        return new_id

    def _incremental_extend(self, record: Dict, obj_id: str) -> None:
        """
        Fast-path incremental update (Eq. 24):
          ΔG = {g_new}, ΔI = new incidence pairs.
        The attribute set M is fixed (the named-constraint catalog), so a new
        record's incidence is computed by constraints.attributes_for() — using
        its `Constraint` label when present, else NL signature extraction.
        """
        import numpy as np
        from rag_fca import constraints as _con

        new_attrs = _con.attributes_for(record)
        new_row_bool = [attr in new_attrs for attr in self.context.attributes]
        new_row_arr = np.array([new_row_bool], dtype=bool)
        self.context.relation = np.vstack([self.context.relation, new_row_arr])
        self.context.objects.append(obj_id)
        self.context.object_data[obj_id] = record

        for attr in new_attrs:
            if attr not in self._attr_weights:
                self._attr_weights[attr] = 1.0

    def add_record_interactive(self, record: Dict,
                               save_path: str = KB_JSON_PATH) -> Tuple[bool, str]:
        """
        User-initiated save (menu option [5]). Unlike try_auto_update this does
        NOT apply the validation-score gate — the user has explicitly chosen to
        keep this result — but it still skips exact/near duplicates so the KB
        does not accumulate identical problems. On success the new problem is
        inserted into the formal context and the concept lattice is updated
        (incremental, Eq. 23-24).
        """
        nl_input = record.get("nl_text", "")
        # For a user-initiated save, only skip an EXACT duplicate: same problem
        # type AND identical constraint attribute set (the formal incidence).
        # Fuzzy similarity would wrongly reject genuinely new variants that share
        # the same base constraints, so it is not used here.
        from rag_fca import constraints as _con
        new_attrs = _con.attributes_for(record)
        for r in self.records:
            if r.get("problem_type") != record.get("problem_type"):
                continue
            if _con.attributes_for(r) == new_attrs:
                return False, (f"an identical constraint set already exists "
                               f"(ID={r['id']}, same type & constraints)")
        new_id = self.add_record(record, save_path=save_path)
        return True, f"added as record ID={new_id}"

    def try_auto_update(
        self,
        record: Dict,
        val_report: Dict,
        nl_input: str,
        save_path: str = KB_JSON_PATH,
    ) -> Tuple[bool, str]:
        """
        Called by the pipeline after each successful generation.
        Returns (was_added: bool, reason: str).

        Steps:
          1. Check policy gate (should_update).
          2. Check for duplicates (similarity ≥ KB_DEDUP_THRESHOLD → skip).
          3. add_record() if all checks pass.
          4. Mark query as success/failure for weight learning.
        """
        ok, reason = self.should_update(val_report)
        if not ok:
            print(f"[KB] Skip auto-update: {reason}")
            # Mark last query as failed for weight learning
            self.mark_retrieval_failed(nl_input)
            return False, reason

        # Duplicate check
        similar = self.retrieve(nl_input, top_k=1, threshold=KB_DEDUP_THRESHOLD)
        if similar:
            msg = f"duplicate exists (ID={similar[0]['id']}, sim ≥ {KB_DEDUP_THRESHOLD})"
            print(f"[KB] Skip auto-update: {msg}")
            return False, msg

        new_id = self.add_record(record, save_path=save_path)
        print(f"[KB] ✓ New record added ID={new_id} | {reason}")
        return True, f"added as ID={new_id}"

    def summary(self) -> Dict:
        return {
            "total_records":    len(self.records),
            "problem_types":    dict(Counter(r["problem_type"] for r in self.records)),
            "total_concepts":   len(self.lattice.concepts)   if self.lattice  else 0,
            "total_attributes": len(self.context.attributes) if self.context  else 0,
            "pending_since_rebuild": self._pending_since_rebuild,
            "update_policy":    KB_UPDATE_POLICY,
            "query_log_size":   len(self._query_log),
            "top_attributes":   self._top_weighted_attributes(5),
        }

    def _top_weighted_attributes(self, n: int = 5) -> List[Dict]:
        """Return top-n attributes by learned importance weight."""
        if not self._attr_weights:
            return []
        sorted_attrs = sorted(self._attr_weights.items(), key=lambda x: x[1], reverse=True)
        return [{"attribute": a, "weight": round(w, 4)} for a, w in sorted_attrs[:n]]

    def export_sft_dataset(
        self,
        output_path: str,
        n_iterations: int = 1,
        use_llm_reframe: bool = False,
    ) -> int:
        """
        Convenience method: export SFT dataset from current KB records.
        See augmentation.chaos_map.export_sft_dataset for full documentation.
        """
        from augmentation.chaos_map import export_sft_dataset
        return export_sft_dataset(
            self.records,
            output_path=output_path,
            include_augmented=True,
            n_aug_iterations=n_iterations,
            use_llm_reframe=use_llm_reframe,
        )


    # ── Knowledge Base Export ─────────────────────────────────────────────────

    def export_to_file(self, output_dir: str = ".", formats: list = None) -> dict:
        """
        Export the knowledge base as structured, human-readable files.

        Produces files that reviewers can inspect to understand the FCA structure:
          - <output_dir>/kb_formal_context.csv    : cross-table K=(G,M,I)
          - <output_dir>/kb_formal_concepts.json  : all formal concepts (A,B)
          - <output_dir>/kb_implication_base.json : canonical implication base
          - <output_dir>/kb_hasse_edges.json      : Hasse diagram edges
          - <output_dir>/kb_summary.txt           : human-readable text summary

        Structure follows Ganter & Wille (1999) and Poelmans et al. (2013).
        """
        import os, json, csv
        from datetime import datetime

        if formats is None:
            formats = ["csv", "json", "txt"]

        os.makedirs(output_dir, exist_ok=True)
        outputs = {}

        if not self.context or not self.lattice:
            print("  [KB Export] No lattice built yet — call load_from_xlsx() first.")
            return outputs

        ctx     = self.context
        lattice = self.lattice

        # ── 1. Formal context cross-table K=(G,M,I)  [Ganter & Wille 1999 Table 1] ──
        if "csv" in formats:
            csv_path = os.path.join(output_dir, "kb_formal_context.csv")
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                # Header: object ID | problem_type | constraint | M attributes...
                header = ["object_id", "problem_type", "constraint"] + ctx.attributes
                writer.writerow(header)
                for i, g in enumerate(ctx.objects):
                    rec   = ctx.object_data.get(g, {})
                    ptype = rec.get("problem_type", "")
                    cst   = rec.get("constraint", "")
                    row   = [g, ptype, cst] + ["x" if ctx.relation[i, j] else ""
                             for j in range(len(ctx.attributes))]
                    writer.writerow(row)
            outputs["formal_context_csv"] = csv_path
            print(f"  [KB Export] Formal context → {csv_path}  "
                  f"({len(ctx.objects)} objects × {len(ctx.attributes)} attributes)")

        # ── 2. All formal concepts (A,B) ─────────────────────────────────────────────
        if "json" in formats:
            concepts_data = []
            for idx, c in enumerate(lattice.concepts):
                # For each concept, record extent (object IDs + their types),
                # intent (attribute names), and the label
                extent_info = []
                for g in sorted(c.extent):
                    rec = ctx.object_data.get(g, {})
                    extent_info.append({
                        "id":           g,
                        "problem_type": rec.get("problem_type", ""),
                        "constraint":   rec.get("constraint", ""),
                    })
                concepts_data.append({
                    "concept_index": idx,
                    "label":         c.label,
                    "level":         c.level,
                    "extent_size":   len(c.extent),
                    "intent_size":   len(c.intent),
                    "extent":        extent_info,
                    "intent":        sorted(c.intent),
                    "note": ("TOP concept" if len(c.extent) == len(ctx.objects)
                             else "BOTTOM concept" if len(c.extent) == 0
                             else ""),
                })
            # Sort: TOP first, then by descending extent size
            concepts_data.sort(key=lambda x: -x["extent_size"])

            concepts_out = {
                "description": (
                    "All formal concepts C=(A,B) of the CPAM-LLM knowledge base. "
                    "Each concept satisfies A'=B (A=object prime) and B'=A (B=attribute prime), "
                    "forming the concept lattice L(K). "
                    "Enumerated via NextClosure algorithm (Ganter 1984)."
                ),
                "formal_context_info": {
                    "objects_G":    len(ctx.objects),
                    "attributes_M": len(ctx.attributes),
                    "lattice_size": len(lattice.concepts),
                },
                "concepts": concepts_data,
            }
            concepts_path = os.path.join(output_dir, "kb_formal_concepts.json")
            with open(concepts_path, "w", encoding="utf-8") as f:
                json.dump(concepts_out, f, ensure_ascii=False, indent=2)
            outputs["formal_concepts_json"] = concepts_path
            print(f"  [KB Export] Formal concepts → {concepts_path}  "
                  f"({len(lattice.concepts)} concepts)")

            # ── 3. Canonical implication base (Duquenne-Guigues) ─────────────────
            impl_data = []
            for impl in lattice.implications:
                impl_data.append({
                    "premise":    sorted(impl.premise),
                    "conclusion": sorted(impl.conclusion),
                    "reads_as":   (f"{{{', '.join(sorted(impl.premise))}}} → "
                                   f"{{{', '.join(sorted(impl.conclusion))}}}"),
                })
            impl_out = {
                "description": (
                    "Canonical implication base (Duquenne-Guigues base) of K. "
                    "An implication U→V holds in K if every object having all attrs "
                    "in U also has all attrs in V. "
                    "This base is cardinality-minimal (Guigues & Duquenne 1986). "
                    "Computed via NextClosure / pseudo-closed set enumeration (Ganter 1984)."
                ),
                "implication_count": len(impl_data),
                "implications": impl_data,
            }
            impl_path = os.path.join(output_dir, "kb_implication_base.json")
            with open(impl_path, "w", encoding="utf-8") as f:
                json.dump(impl_out, f, ensure_ascii=False, indent=2)
            outputs["implication_base_json"] = impl_path
            print(f"  [KB Export] Implication base → {impl_path}  "
                  f"({len(impl_data)} implications)")

            # ── 4. Hasse diagram edges ────────────────────────────────────────────
            if lattice.hasse_edges:
                hasse_data = []
                for child_idx, parent_idx in lattice.hasse_edges:
                    child  = lattice.concepts[child_idx]
                    parent = lattice.concepts[parent_idx]
                    hasse_data.append({
                        "child_index":  child_idx,
                        "parent_index": parent_idx,
                        "child_label":  child.label,
                        "parent_label": parent.label,
                        "child_extent_size":  len(child.extent),
                        "parent_extent_size": len(parent.extent),
                        "meaning": (f"concept[{child_idx}] ≤ concept[{parent_idx}]  "
                                    f"(A_{child_idx} ⊂ A_{parent_idx})"),
                    })
                hasse_out = {
                    "description": (
                        "Hasse diagram of the concept lattice L(K): direct cover relations. "
                        "(A1,B1) ≤ (A2,B2) iff A1 ⊆ A2 iff B2 ⊆ B1 (Ganter & Wille 1999 Eq.14). "
                        "Only direct covers are listed (no transitive edges)."
                    ),
                    "edge_count": len(hasse_data),
                    "edges": hasse_data,
                }
                hasse_path = os.path.join(output_dir, "kb_hasse_edges.json")
                with open(hasse_path, "w", encoding="utf-8") as f:
                    json.dump(hasse_out, f, ensure_ascii=False, indent=2)
                outputs["hasse_edges_json"] = hasse_path
                print(f"  [KB Export] Hasse edges → {hasse_path}  "
                      f"({len(hasse_data)} edges)")

            # ── 4b. Per-domain concept lattices (paper Figs 2b–6b) ───────────────
            try:
                from rag_fca import constraints as _con
            except Exception:
                _con = None
            if _con is not None:
                from rag_fca import formalizations as _fm

                def _detail(name):
                    """Constraint name → full formal record {name,desc,math,cp}."""
                    f = _fm.FORMAL.get(name, {})
                    return {
                        "name": name,
                        "desc": f.get("desc", ""),
                        "math": f.get("math", ""),
                        "cp":   f.get("cp",   ""),
                    }

                # precompute concept index → its domain anchors
                domains_out = {}
                for ptype, spec in _con.CATALOG.items():
                    anchor = spec["anchor"]

                    # concepts belonging to this domain, richly described
                    dom_concepts = []
                    dom_concept_idx = set()
                    for idx, c in enumerate(lattice.concepts):
                        if anchor not in c.intent:
                            continue
                        dom_concept_idx.add(idx)
                        # split this concept's intent into the meaningful groups
                        base_here = [a for a in spec["base"] if a in c.intent]
                        opt_here  = [o["name"] for o in spec["optional"]
                                     if o["name"] in c.intent]
                        obj_here  = [spec["objective"]] if spec["objective"] in c.intent else []
                        example_ids = sorted(
                            c.extent, key=lambda x: int(x) if str(x).isdigit() else 0)[:6]
                        dom_concepts.append({
                            "concept_index":   idx,
                            "extent_size":     len(c.extent),
                            "example_problem_ids": example_ids,
                            "objective":       [_detail(a) for a in obj_here],
                            "base_constraints":     [_detail(a) for a in base_here],
                            "additional_constraints":[_detail(a) for a in opt_here],
                            # full formal intent (everything, in one list)
                            "all_constraints": [_detail(a) for a in sorted(c.intent)],
                        })
                    dom_concepts.sort(key=lambda x: -x["extent_size"])

                    # implications whose premise+conclusion live in this domain
                    dom_impls = []
                    for im in lattice.implications:
                        names = set(im.premise) | set(im.conclusion)
                        if anchor in names or names & set(spec["base"]) or \
                           names & {o["name"] for o in spec["optional"]}:
                            # keep only implications fully inside this domain
                            domain_attrs = ({anchor, spec["objective"]} |
                                            set(spec["base"]) |
                                            {o["name"] for o in spec["optional"]})
                            if names <= domain_attrs:
                                dom_impls.append({
                                    "premise":    sorted(im.premise),
                                    "conclusion": sorted(im.conclusion),
                                    "reading": (
                                        "a problem with {"
                                        + ", ".join(sorted(im.premise))
                                        + "} must also have {"
                                        + ", ".join(sorted(im.conclusion)) + "}"),
                                })

                    # Hasse cover relations internal to this domain
                    dom_edges = []
                    for child_idx, parent_idx in (lattice.hasse_edges or []):
                        if child_idx in dom_concept_idx and parent_idx in dom_concept_idx:
                            dom_edges.append({"child_index": child_idx,
                                              "parent_index": parent_idx})

                    domains_out[ptype] = {
                        "scenario":  ptype,
                        "anchor":    _detail(anchor),
                        "objective": _detail(spec["objective"]),
                        "base_constraints": [_detail(a) for a in spec["base"]],
                        "additional_constraints": [
                            {**_detail(o["name"]),
                             "dataset_constraint_id": o.get("id"),
                             "relaxes": o.get("relax", "")}
                            for o in spec["optional"]],
                        "n_concepts":   len(dom_concepts),
                        "n_implications": len(dom_impls),
                        "concepts":     dom_concepts,
                        "implications": dom_impls,
                        "hasse_edges":  dom_edges,
                    }

                dom_out = {
                    "description": (
                        "Per-domain concept-lattice knowledge bases — one per scenario, "
                        "matching CPAM-LLM Figs 2b–6b but FULLER than the simplified "
                        "figures. For every concept and every constraint we give the "
                        "formal representation: a one-line description, the constraint in "
                        "mathematical notation, and a docplex.cp code template. This is "
                        "the structural knowledge the RAG-FCA stage hands to the LLM "
                        "(it resolves a query's constraints to these formal templates), "
                        "so the second stage can build the formal model and CP code."
                    ),
                    "field_guide": {
                        "anchor": "problem-type tag every problem of the scenario carries",
                        "objective": "objective function (formalized)",
                        "base_constraints": "constraints present in EVERY problem of the scenario",
                        "additional_constraints": "optional constraint types; dataset_constraint_id "
                            "is the numbered id used in the source `Constraint` column; "
                            "relaxes (if set) means this variant REMOVES that base constraint",
                        "concepts": "formal concepts of this domain's sub-lattice; each lists its "
                            "constraint intent split into objective/base/additional, all formalized",
                        "implications": "Duquenne-Guigues implications internal to this domain",
                        "hasse_edges": "cover relations between this domain's concepts",
                    },
                    "domains": domains_out,
                }
                dom_path = os.path.join(output_dir, "kb_domain_lattices.json")
                with open(dom_path, "w", encoding="utf-8") as f:
                    json.dump(dom_out, f, ensure_ascii=False, indent=2)
                outputs["domain_lattices_json"] = dom_path
                print(f"  [KB Export] Per-domain lattices (formalized) → {dom_path}  "
                      f"({len(domains_out)} domains)")

        # ── 5. Human-readable text summary ────────────────────────────────────────
        if "txt" in formats:
            txt_path = os.path.join(output_dir, "kb_summary.txt")
            lines = []
            W = 72

            lines += [
                "="*W,
                "  CPAM-LLM Knowledge Base — FCA Concept Lattice Summary",
                "  References:",
                "    Ganter & Wille (1999): Formal Concept Analysis",
                "    Guigues & Duquenne (1986): Canonical implication base",
                "    Poelmans et al. (2013): FCA in knowledge processing",
                "    NextClosure algorithm: Ganter (1984)",
                f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "="*W,
                "",
                f"1. FORMAL CONTEXT  K = (G, M, I)",
                f"   Objects  |G| = {len(ctx.objects):5d}  (KB problem instances)",
                f"   Attributes |M| = {len(ctx.attributes):5d}  (CP domain keywords)",
                f"   Incidence entries: {int(ctx.relation.sum())} / {len(ctx.objects)*len(ctx.attributes)}",
                f"   Density: {ctx.relation.mean()*100:.1f}%",
                "",
                "   Attribute set M (complete list):",
            ]
            for i, attr in enumerate(ctx.attributes):
                lines.append(f"     m{i+1:03d}: {attr}")

            lines += [
                "",
                f"2. CONCEPT LATTICE  L(K)",
                f"   |L(K)| = {len(lattice.concepts)} formal concepts",
                f"   Each concept C=(A,B) satisfies  A' = B  AND  B' = A",
                f"   Enumerated via NextClosure (Ganter 1984, lectic order)",
                "",
                "   Formal concepts ordered by |A| descending:",
                f"   {'Idx':>4}  {'|A|':>5}  {'|B|':>5}  {'Label':<35}  Sample intent (first 4 attrs)",
                "   " + "-"*(W-3),
            ]
            for idx, c in enumerate(sorted(lattice.concepts,
                                           key=lambda x: -len(x.extent))):
                sample_intent = ", ".join(sorted(c.intent)[:4])
                if len(c.intent) > 4:
                    sample_intent += f", ... (+{len(c.intent)-4})"
                lines.append(
                    f"   {idx:>4}  {len(c.extent):>5}  {len(c.intent):>5}  "
                    f"{c.label:<35}  {sample_intent}"
                )

            if lattice.implications:
                lines += [
                    "",
                    f"3. CANONICAL IMPLICATION BASE  (Duquenne-Guigues base)",
                    f"   {len(lattice.implications)} implications — cardinality-minimal complete base",
                    f"   An implication U→V holds iff every object with attrs U has attrs V",
                    "",
                ]
                for impl in lattice.implications:
                    lines.append(f"   {{{', '.join(sorted(impl.premise))}}} → "
                                 f"{{{', '.join(sorted(impl.conclusion))}}}")
            else:
                lines += [
                    "",
                    "3. CANONICAL IMPLICATION BASE",
                    "   (Not computed — KB too large or trivial implication base)",
                ]

            if lattice.hasse_edges:
                lines += [
                    "",
                    f"4. HASSE DIAGRAM  (cover relations of L(K))",
                    f"   {len(lattice.hasse_edges)} direct cover edges",
                    f"   (A1,B1) covers (A2,B2) iff A1 ⊂ A2, no intermediate concept",
                    "",
                ]
                for ci, pi in lattice.hasse_edges[:20]:
                    cc, pc = lattice.concepts[ci], lattice.concepts[pi]
                    lines.append(f"   concept[{ci}] ({cc.label[:20]}) "
                                 f"< concept[{pi}] ({pc.label[:20]})")
                if len(lattice.hasse_edges) > 20:
                    lines.append(f"   ... ({len(lattice.hasse_edges)-20} more edges)")

            lines += ["", "="*W]

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            outputs["summary_txt"] = txt_path
            print(f"  [KB Export] Text summary → {txt_path}")

        return outputs


_kb_instance: Optional[KnowledgeBase] = None

def get_knowledge_base() -> KnowledgeBase:
    """
    Return the singleton knowledge base, loaded from the permanent JSON store.

    The KB must be built first (once) with:  python build_kb.py
    main.py and the pipeline then load the JSON directly — fast, and with no
    dependency on the source xlsx.
    """
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
        if not os.path.exists(KB_JSON_PATH):
            raise FileNotFoundError(
                "\n  Knowledge base has not been built yet.\n"
                f"  Expected JSON store at: {KB_JSON_PATH}\n"
                "  Build it once with:      python build_kb.py\n"
            )
        _kb_instance.load_from_json(KB_JSON_PATH)
    return _kb_instance
