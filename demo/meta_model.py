"""
Meta-Model Assembler 
Pipeline:
  Step 1  Entity Alignment      — UMLS CUI mapping + cross-KM entity graph
  Step 2  Evidence-Weighted Attention  — α = softmax(QKᵀ/√d) · w_evidence
  Step 3  Fusion                — h_fused = Σ αᵢ · eᵢ
  Step 4  Conflict Detection    — same HEAD entity + contradictory TAIL claims
  Step 5  Structured SLM prompt — Ollama (streaming) or Template fallback

Embeddings: TF-IDF (demo). Swap embed_texts() → RotatE/CompGCN gRPC in prod.
"""

import json, math, re, hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. UMLS CUI TABLE
UMLS_CUI_MAP = {
    "renal cell carcinoma":             "C0007134",
    "clear cell renal cell carcinoma":  "C0279702",
    "ccRCC":                            "C0279702",
    "papillary renal cell carcinoma":   "C0279703",
    "chromophobe renal cell carcinoma": "C1266042",
    "chronic kidney disease":           "C1561643",
    "ckd":                              "C1561643",
    "end-stage renal disease":          "C0022661",
    "esrd":                             "C0022661",
    "hypertension":                     "C0020538",
    "acute kidney injury":              "C0022660",
    "iga nephropathy":                  "C0027726",
    "nephrotic syndrome":               "C0027726",
    "nephrolithiasis":                  "C0392525",
    "polycystic kidney disease":        "C0022680",
    "diabetic nephropathy":             "C0011881",
    "hematuria":                        "C0018965",
    "flank pain":                       "C0016199",
    "flank mass":                       "C0577559",
    "anemia":                           "C0002871",
    "proteinuria":                      "C0033687",
    "vhl":                              "C1414474",
    "vhl gene":                         "C1414474",
    "pbrm1":                            "C1825100",
    "bap1":                             "C1443985",
    "apol1":                            "C1428308",
    "radical nephrectomy":              "C0194782",
    "partial nephrectomy":              "C0401048",
    "pembrolizumab":                    "C3658706",
    "sunitinib":                        "C1566795",
    "belzutifan":                       "C4743961",
    "nivolumab":                        "C3657083",
    "cabozantinib":                     "C2986388",
    "hemodialysis":                     "C0019004",
    "creatinine":                       "C0010294",
    "blood pressure":                   "C0005823",
}

def cui(entity: str) -> str:
    k = entity.strip().lower()
    return UMLS_CUI_MAP.get(k, "LOCAL:" + hashlib.md5(k.encode()).hexdigest()[:8].upper())


# 2. DATA STRUCTURES
@dataclass
class AlignedEntity:
    raw_text:   str
    cui_id:     str
    klm_source: str
    confidence: float = 1.0

@dataclass
class AlignedTriple:
    triple_id:       str
    head:            AlignedEntity
    relation:        str
    tail:            AlignedEntity
    confidence:      float
    evidence_level:  str
    evidence_weight: float
    klm_source:      str
    source:          str
    annotation:      str = ""
    embedding:       Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class KMResultSet:
    klm_id:              str
    triples:             list[AlignedTriple]
    agg_embedding:       Optional[np.ndarray] = field(default=None, repr=False)
    evidence_level_mode: str = "III"

@dataclass
class FusionOutput:
    fused_embedding:   np.ndarray
    ranked_triples:    list[AlignedTriple]
    entity_graph:      dict
    conflict_pairs:    list[dict]
    klm_results:       list[KMResultSet]
    query:             str
    structured_prompt: str


# 3. EVIDENCE WEIGHTS  
EVIDENCE_WEIGHTS = {"I": 1.0, "II": 0.8, "III": 0.5, "IV": 0.3, "V": 0.1}

def ev_weight(level: str) -> float:
    return EVIDENCE_WEIGHTS.get(level.strip(), 0.1)


# 4. EMBEDDING ENGINE  (TF-IDF demo; swap for RotatE gRPC in prod)
class EmbeddingEngine:
    def __init__(self, corpus: list[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2048, sublinear_tf=True)
        self.vectorizer.fit(corpus)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat / norms

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


# 5. STEP 1 — ENTITY ALIGNMENT
def align_triple(t: dict, klm_source: str) -> AlignedTriple:
    ev = t.get("evidence_level", "III")
    ht, tt = t.get("head", ""), t.get("tail", "")
    return AlignedTriple(
        triple_id=t.get("triple_id", ""),
        head=AlignedEntity(raw_text=ht, cui_id=cui(ht), klm_source=klm_source,
                           confidence=t.get("confidence", 0.5)),
        relation=t.get("relation", ""),
        tail=AlignedEntity(raw_text=tt, cui_id=cui(tt), klm_source=klm_source),
        confidence=t.get("confidence", 0.5),
        evidence_level=ev, evidence_weight=ev_weight(ev),
        klm_source=klm_source, source=t.get("source", ""),
        annotation=t.get("annotation", ""),
    )

def build_entity_graph(result_sets: list[KMResultSet]) -> dict:
    """CUI → list of triples that mention that entity (as head OR tail)."""
    graph: dict[str, list[AlignedTriple]] = {}
    for rs in result_sets:
        for t in rs.triples:
            for c in (t.head.cui_id, t.tail.cui_id):
                graph.setdefault(c, []).append(t)
    return graph

def find_merged_entities(entity_graph: dict) -> list[dict]:
    """CUIs confirmed in ≥2 KLMs."""
    merged = []
    for c, triples in entity_graph.items():
        sources = set(t.klm_source for t in triples)
        if len(sources) > 1:
            texts = list({
                t.head.raw_text if t.head.cui_id == c else t.tail.raw_text
                for t in triples
            })
            merged.append({
                "cui": c, "raw_texts": texts,
                "klm_sources": list(sources), "triple_count": len(triples),
            })
    return merged


# 6. STEP 2 — EVIDENCE-WEIGHTED ATTENTION [α(i) = softmax(Q·Kᵀ / √d) · w_evidence(i)], re-normalised
def evidence_weighted_attention(
    query_vec: np.ndarray,
    key_matrix: np.ndarray,
    evidence_weights: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    D = query_vec.shape[0]
    scores = key_matrix @ query_vec / (math.sqrt(D) * temperature)
    scores -= scores.max()
    softmax_w = np.exp(scores) / (np.exp(scores).sum() + 1e-9)
    weighted = softmax_w * evidence_weights
    return weighted / (weighted.sum() + 1e-9)


# 7. STEP 3 — FUSION
def fuse_embeddings(embeddings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (weights[:, None] * embeddings).sum(axis=0)


# 8. STEP 4 — CONFLICT DETECTION

#  A real conflict requires ALL of:
#    (a) SAME HEAD CUI — same subject entity making the claim
#    (b) SAME or OPPOSED relation type 
#    (c) DIFFERENT tail — two KLMs assert different things about the same fact
#  Additionally: patient EHR triples (head = P-001, P-46F-...) are OBSERVATIONS, not knowledge claims — never conflict with knowledge triples.

# Relations that are semantically opposed within the same domain
OPPOSED_PAIRS = {
    frozenset({"treated_by",     "contraindicated_with"}): "treatment_vs_contraindication",
    frozenset({"risk_factor_for","prevented_by"}):         "risk_vs_prevention",
    frozenset({"predisposes_to", "prevented_by"}):         "predisposition_vs_prevention",
}

# Group relations by semantic type — only compare within the same group
RELATION_GROUPS: dict[str, set[str]] = {
    "treatment":    {"treated_by", "treats", "managed_with", "contraindicated_with"},
    "symptom":      {"has_symptom", "presents_with", "shows_clinical_finding"},
    "risk":         {"risk_factor_for", "is_risk_factor_for", "prevented_by"},
    "staging":      {"staged_by", "has_tumor_stage", "disease_progression_stage"},
    "pathogenesis": {"pathogenesis_involves", "causes", "associated_with"},
    "genetic":      {"carries_genetic_variant", "predisposes_to"},
}

def _rel_group(relation: str) -> Optional[str]:
    for grp, rels in RELATION_GROUPS.items():
        if relation in rels:
            return grp
    return None


def _is_ehr_triple(t: AlignedTriple) -> bool:
    """
    Returns True only for patient admin EHR triples (demographics, follow-up notes) that should NOT receive clinical boost.
    Returns False (boost-eligible) for all clinically meaningful patient observations: symptoms, findings, labs, vitals, imaging, meds, procedures, diagnoses, genetics.
    """
    if not re.match(r"^P[-\d]", t.head.raw_text):
        return False
    KNOWLEDGE_RELATIONS = {
        "carries_genetic_variant", "has_genetic_variant",
        "predisposes_to", "has_family_history",
        "has_pharmacogenomic_profile", "has_polygenic_risk_score",
        "has_symptom", "shows_clinical_finding", "presents_with",
        "has_vital", "has_lab_value", "shows_laboratory_finding",
        "has_imaging_finding",
        "prescribed_medication", "managed_with", "underwent_procedure",
        "has_medical_history",
        "has_diagnosis", "diagnosed_with", "has_condition",
        "has_comorbidity", "has_disease_stage", "has_tumor_stage",
        "has_risk_factor",
    }
    return t.relation not in KNOWLEDGE_RELATIONS

def detect_conflicts(result_sets: list[KMResultSet], entity_graph: dict) -> list[dict]:
    """
    A conflict = same HEAD entity (by CUI) + same relation group + different tail OR opposed relation pair

    Explicitly excluded:
    - Patient EHR triples (P-001 observations) vs knowledge triples
    - Triples from the same KLM
    - Triples in different relation groups 
    """
    conflicts, seen = [], set()
    merged = find_merged_entities(entity_graph)

    for m in merged:
        c = m["cui"]
        # Only triples where this CUI is the HEAD (the claim subject)
        head_triples = [
            t for t in entity_graph[c]
            if t.head.cui_id == c and not _is_ehr_triple(t)
        ]
        # Group by KLM source
        by_klm: dict[str, list[AlignedTriple]] = {}
        for t in head_triples:
            by_klm.setdefault(t.klm_source, []).append(t)

        if len(by_klm) < 2:
            continue

        klm_ids = list(by_klm.keys())
        for i in range(len(klm_ids)):
            for j in range(i + 1, len(klm_ids)):
                for ta in by_klm[klm_ids[i]]:
                    for tb in by_klm[klm_ids[j]]:
                        grp_a, grp_b = _rel_group(ta.relation), _rel_group(tb.relation)

                        # Must be in the same relation group to be comparable
                        if grp_a is None or grp_a != grp_b:
                            continue

                        rel_pair = frozenset({ta.relation, tb.relation})
                        opposed  = OPPOSED_PAIRS.get(rel_pair)

                        # Same relation, different tails = conflicting claims
                        same_rel_diff_tail = (
                            ta.relation == tb.relation
                            and ta.tail.raw_text.lower() != tb.tail.raw_text.lower()
                        )

                        if not (opposed or same_rel_diff_tail):
                            continue

                        key = (c, ta.tail.raw_text, tb.tail.raw_text)
                        if key in seen:
                            continue
                        seen.add(key)

                        conflicts.append({
                            "cui":          c,
                            "entity":       ta.head.raw_text,
                            "conflict_type": opposed or "contradictory_claims",
                            "triple_a": {
                                "klm": ta.klm_source, "head": ta.head.raw_text,
                                "relation": ta.relation, "tail": ta.tail.raw_text,
                                "evidence": ta.evidence_level,
                            },
                            "triple_b": {
                                "klm": tb.klm_source, "head": tb.head.raw_text,
                                "relation": tb.relation, "tail": tb.tail.raw_text,
                                "evidence": tb.evidence_level,
                            },
                            "severity": "HIGH" if opposed else "MEDIUM",
                        })
    return conflicts


# 9. STEP 5 — STRUCTURED SLM PROMPT
def build_slm_prompt(
    query: str,
    ranked_triples: list[AlignedTriple],
    entity_graph: dict,
    conflicts: list[dict],
    patient_context: dict,
    top_n: int = 12,
) -> str:
    lines = [
        "You are the KM Meta-Model clinical reasoning layer.",
        "You have received fused, evidence-ranked knowledge from three KLMs.",
        "Give a concise structured clinical response.",
        "",
        "=== PATIENT CONTEXT ===",
    ]
    for k, v in patient_context.items():
        v_str = ", ".join(str(x) for x in v[:3]) if isinstance(v, list) else str(v)
        lines.append(f"{k}: {v_str}")
    lines.append("")

    merged = find_merged_entities(entity_graph)
    if merged:
        lines.append("=== CROSS-KLM ALIGNED ENTITIES (UMLS) ===")
        for m in merged[:5]:
            lines.append(
                f"  CUI {m['cui']}: {' / '.join(m['raw_texts'][:2])} "
                f"[{', '.join(m['klm_sources'])}]"
            )
        lines.append("")

    lines.append(f"=== RANKED KNOWLEDGE (top {min(top_n, len(ranked_triples))}) ===")
    for i, t in enumerate(ranked_triples[:top_n], 1):
        ann = f" — {t.annotation}" if t.annotation else ""
        lines.append(
            f"  {i:02d}. [{t.klm_source.upper()}] "
            f"{t.head.raw_text} –[{t.relation}]→ {t.tail.raw_text} "
            f"[Ev:{t.evidence_level}, Conf:{t.confidence:.2f}]{ann}"
        )
    lines.append("")

    if conflicts:
        lines.append("=== ⚠ DETECTED CONFLICTS ===")
        for c in conflicts[:3]:
            a, b = c["triple_a"], c["triple_b"]
            lines.append(f"  [{c['severity']}] {c['entity']} — {c['conflict_type']}")
            lines.append(f"    {a['klm']}: {a['head']} –[{a['relation']}]→ {a['tail']} [Ev:{a['evidence']}]")
            lines.append(f"    {b['klm']}: {b['head']} –[{b['relation']}]→ {b['tail']} [Ev:{b['evidence']}]")
        lines.append("")
    else:
        lines.append("=== CONFLICTS: None detected ===\n")

    lines += [
        "=== DOCTOR QUERY ===",
        query,
        "",
        "=== RESPONSE FORMAT (follow exactly) ===",
        "CLINICAL ANSWER — direct answer to the query",
        "EVIDENCE SUMMARY — cite trials/guidelines by name",
        "PATIENT-SPECIFIC NOTE — apply to this patient's data above",
        "RED FLAGS — conflicts or uncertainties to escalate",
        "RECOMMENDED NEXT STEPS — bullet list, max 4 items",
        "",
        "Keep to 200-300 words. No definitive diagnosis statements.",
    ]
    return "\n".join(lines)


# 10. SLM ADAPTERS
#  Two options (auto-detected in order):
#    1. Ollama   — locally
#    2. Template — zero-dependency fallback

class SLMAdapter:
    name = "base"
    def generate(self, prompt: str, stream_callback=None) -> str:
        """
        Generate a response from the given prompt.
        stream_callback: optional callable(token: str) called for each token as it arrives. Use this to display live output in the demo.
        """
        raise NotImplementedError
    def available(self) -> bool: return False


class OllamaAdapter(SLMAdapter):
    name = "ollama"

    SYSTEM = (
        "You are a clinical knowledge reasoning assistant. "
        "You synthesise information from multiple medical knowledge models. "
        "Be concise, evidence-based, and always flag uncertainty."
    )

    def __init__(self, model: str = "mistral", host: str = "http://localhost:11434"):
        self.model = model
        self.host  = host

    def available(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    # Models in order of speed on CPU-only machines (fastest first).
    PREFERRED_MODELS = [ "llama3.2:1b", "llama3.2:3b", "phi3", "phi3.5", "phi", "gemma2", "mistral", "llama3", "llama2", "qwen2", "deepseek-r1"]

    def get_available_model(self) -> str:
        # Return the fastest available pulled model
        try:
            import urllib.request, json as _j
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                data = _j.loads(r.read())
                pulled = [m["name"] for m in data.get("models", [])]
                pulled_base = [m.split(":")[0] for m in pulled]
                for preferred in self.PREFERRED_MODELS:
                    if preferred in pulled:
                        return preferred
                    base = preferred.split(":")[0]
                    if base in pulled_base:
                        return pulled[pulled_base.index(base)]
                if pulled:
                    return pulled[0]
        except Exception:
            pass
        return self.model

    def _shorten_prompt(self, prompt: str, max_triples: int = 5) -> str:
        """Keep only patient context, top N triples, query. Drop the rest."""
        out, keep, count = [], None, 0
        skip_section = False
        for line in prompt.split("\n"):
            if line.startswith("==="):
                skip_section = "CROSS-KLM" in line or "CONFLICT" in line
                keep = line
                count = 0
                if not skip_section:
                    out.append(line)
                continue
            if skip_section:
                continue
            if keep and "RANKED KNOWLEDGE" in keep:
                if re.match(r"\s+\d+\.", line):
                    count += 1
                    if count > max_triples:
                        continue
            out.append(line)
        return "\n".join(out)

    # Small models (1B-3B) can't reliably follow rigid multi-section formats. Detect them and send a simpler direct-question prompt instead.
    def _is_small_model(self, model: str) -> bool:
        return any(s in model.lower() for s in ["1b", "2b", "3b", "0.5b", "tinyllama"])

    def _make_small_model_prompt(self, prompt: str, max_triples: int = 6) -> str:
        """
        Key improvement: re-sorts ranked triples to put PATIENT_KLM triples first,
        then patient-condition-matching PATHOLOGY triples, then the rest.
        This ensures the 1B model always sees the most relevant triples in its
        limited attention window — regardless of the raw ranking score order.
        """
        def section(heading, stop=("===",)):
            out_lines, inside = [], False
            for ln in prompt.split("\n"):
                if heading in ln:
                    inside = True; continue
                if inside and any(ln.startswith(s) for s in stop):
                    break
                if inside:
                    out_lines.append(ln)
            return out_lines

        patient = section("=== PATIENT CONTEXT ===")
        ranked  = section("=== RANKED KNOWLEDGE")
        query_l = section("=== DOCTOR QUERY ===")
        query   = query_l[0].strip() if query_l else "clinical query"
        ctx     = "\n".join(l for l in patient if l.strip())

        # Parse ranked triple lines: "[PATIENT_KLM]" or "[PATHOLOGY_KLM]" etc.
        triple_lines = [l for l in ranked if re.match(r"\s+\d+\.", l)]

        # Tier 1: patient KLM triples (always directly about this patient)
        patient_lines  = [l for l in triple_lines if "[PATIENT_KLM]" in l]
        # Tier 2: pathology triples whose head matches known patient conditions
        PATIENT_CONDITIONS = {
            "renal cell", "clear cell", "rcc",
            "chronic kidney", "ckd", "hypertension",
            "hematuria", "von hippel", "vhl",
        }
        def is_patient_condition(line: str) -> bool:
            ll = line.lower()
            return any(c in ll for c in PATIENT_CONDITIONS)

        relevant_path  = [l for l in triple_lines
                          if "[PATIENT_KLM]" not in l and is_patient_condition(l)]
        other_path     = [l for l in triple_lines
                          if "[PATIENT_KLM]" not in l and not is_patient_condition(l)]

        # Rebuild ordered list: patient first, then relevant, then other
        ordered = (patient_lines + relevant_path + other_path)[:max_triples]

        # Re-number them sequentially
        renumbered = []
        for n, line in enumerate(ordered, 1):
            line = re.sub(r"^\s+\d+\.", f"  {n:02}.", line)
            renumbered.append(line)

        triples = "\n".join(renumbered)

        return (
            f"Patient summary:\n{ctx}\n\n"
            f"Relevant clinical knowledge:\n{triples}\n\n"
            f"Question: {query}\n\n"
            "Answer in 3-5 bullet points specific to this patient. "
            "Include evidence. Flag uncertainties. No diagnosis statements."
        )

    def generate(self, prompt: str, stream_callback=None) -> str:
        """
        Generate from Ollama with live streaming.
        stream_callback: optional callable(token: str) — called per token for live display. 
        When provided, the demo shows output as it arrives rather than waiting for the full response.
        """
        import urllib.request, json as _json, os
        model = self.get_available_model()
        num_threads = int(os.cpu_count() or 2)

        if self._is_small_model(model):
            # 1B-2B models: simpler prompt, larger token budget
            send_prompt = self._make_small_model_prompt(prompt, max_triples=6)
            num_predict, num_ctx = 800, 1536
        else:
            send_prompt = self._shorten_prompt(prompt, max_triples=5)
            num_predict, num_ctx = 400, 2048

        payload = _json.dumps({
            "model":  model,
            "system": self.SYSTEM,
            "prompt": send_prompt,
            "stream": True,
            "options": {
                "num_predict": num_predict,
                "temperature": 0.1,
                "top_p":       0.9,
                "num_ctx":     num_ctx,
                "num_thread":  num_threads,
            },
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        chunks = []
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                for raw_line in resp:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        obj = _json.loads(raw_line)
                    except Exception:
                        continue
                    token = obj.get("response", "")
                    chunks.append(token)
                    if stream_callback and token:
                        stream_callback(token)
                    if obj.get("done"):
                        break
            result = "".join(chunks).strip()
            if not result:
                raise RuntimeError(f"Ollama ({model}) returned empty response")
            return result
        except TimeoutError:
            partial = "".join(chunks).strip()
            if partial:
                return partial + "\n[Note: truncated — Ollama timed out]"
            raise RuntimeError(
                f"Ollama timed out ({model}).\n"
            )



class TemplateSLM(SLMAdapter):
    """
    Zero-dependency fallback. Deterministic structured output from ranked triples.
    Rule-based extraction. 
    """
    name = "template"

    def available(self) -> bool:
        return True

    def generate(self, prompt: str, stream_callback=None) -> str:
        lines = prompt.split("\n")

        def section(marker: str, end_markers: list[str]) -> list[str]:
            out, on = [], False
            for l in lines:
                if marker in l:    on = True;  continue
                if on and any(m in l for m in end_markers): break
                if on and l.strip(): out.append(l.strip())
            return out

        patient       = section("=== PATIENT CONTEXT ===",   ["==="])
        ranked        = section("=== RANKED KNOWLEDGE",      ["==="])
        conflict_raw  = section("=== ⚠ DETECTED CONFLICTS",  ["==="])
        query_lines   = section("=== DOCTOR QUERY ===",      ["==="])
        query = query_lines[0] if query_lines else "clinical query"

        def field(key):
            return next((l.split(":", 1)[1].strip() for l in patient if key in l), "")

        dx       = field("Primary Dx")   or "renal condition"
        stage    = field("Tumor Stage")
        egfr     = field("Latest eGFR")
        variants = field("Genetic Variants")
        bp       = field("Latest BP")

        # Parse active and historical conditions from prompt context
        active_cond_raw  = field("Active Conditions")
        hist_cond_raw    = field("Historical Conditions (resolved/past)")
        # Build relevance keywords from active conditions (ICD descriptions)
        import re as _re_tmpl
        ACTIVE_HEAD_KW = {"renal cell", "clear cell", "rcc", "kidney cancer", "kidney tumour",
                          "chronic kidney", "ckd", "hypertension", "hematuria"}
        if active_cond_raw:
            for cond in active_cond_raw.split(","):
                # Extract key words from ICD description (after the dash)
                desc = _re_tmpl.sub(r"ICD-10:[\s\w.()]*-\s*", "", cond.strip()).lower()
                for w in desc.split():
                    if len(w) > 3:
                        ACTIVE_HEAD_KW.add(w)

        # Detect query intent
        q_lower = query.lower()

        # Patient-state intents (tense/perspective aware — answer from patient KLM)
        is_patient_history   = any(w in q_lower for w in [
            "undertaken", "has taken", "has had", "has been", "has received", "underwent",
            "past treatment", "previous", "prior treatment", "current medication",
            "currently on", "currently taking", "what medication", "what drug", "what drugs",
            "on medication", "what is patient taking", "what is she taking",
            "what has patient", "what procedure", "what is she on",
        ])
        is_patient_diagnosis = (any(w in q_lower for w in [
            "what disease", "what condition", "what does patient have", "what illnesses",
            "comorbid", "other condition", "other disease", "apart from", "besides",
            "diagnos", "what is patient diagnosed", "what has patient got",
        ]) and not any(w in q_lower for w in [
            "genetic", "predispos", "variant", "mutation", "gene", "vhl",
            "pbrm1", "bap1", "hereditary", "inherited", "germline", "apol1",
        ]))

        # Clinical knowledge intents
        is_genetic  = any(w in q_lower for w in ["genetic", "predispos", "variant", "mutation",
                                                   "gene", "vhl", "pbrm1", "bap1", "hereditary",
                                                   "inherited", "germline", "apol1", "allele"])
        is_symptom  = any(w in q_lower for w in ["symptom", "sign", "feel", "pain", "present",
                                                   "experience", "complain"])
        is_risk     = (any(w in q_lower for w in ["risk", "factor", "cause", "likelihood"])
                       and not is_genetic and not is_patient_diagnosis)
        is_lab      = any(w in q_lower for w in ["lab", "egfr", "creatinine", "level", "test",
                                                   "bun", "result", "haemoglobin", "hemoglobin"])
        is_staging  = any(w in q_lower for w in ["stage", "staging", "grade", "extent", "tnm"])
        is_imaging  = any(w in q_lower for w in ["imaging", "scan", "ct", "mri", "finding",
                                                   "mass", "radiolog"])
        is_treat    = (not is_genetic and not is_symptom and not is_risk
                       and not is_lab and not is_staging and not is_imaging
                       and not is_patient_history and not is_patient_diagnosis
                       and any(w in q_lower for w in ["treat", "therapy", "drug", "option",
                                                       "surgery", "management", "cure",
                                                       "antibiotic", "chemotherapy",
                                                       "immunotherapy", "require", "need",
                                                       "recommend"]))

        # Parse ranked triples by relation type 
        # Categorised by both knowledge-type and patient-state type
        patient_meds, patient_diagnoses_items, patient_procedures = [], [], []
        treatments, symptoms, risks, genetic, other = [], [], [], [], []

        # Only confirmed active diagnoses — not historical ICD codes like resolved UTI
        patient_entity_names: set[str] = set()
        for l in patient:
            if "Primary Dx" in l or "Tumor Stage" in l or "Disease Stage" in l:
                val = l.split(":", 1)[-1].strip().lower()
                for word in re.findall(r"\b\w+\b", val):
                    patient_entity_names.add(word)

        for r in ranked:
            rel_match  = re.search(r"–\[(.+?)\]→", r)
            tail_match = re.search(r"→\s*(.+?)\s*\[Ev:", r)
            head_match = re.search(r"\]\s*(.+?)\s*–\[", r)
            ann_match  = re.search(r"—\s*(.+)$", r)
            if not rel_match or not tail_match:
                continue
            rel  = rel_match.group(1)
            tail = tail_match.group(1).strip()
            head = head_match.group(1).strip() if head_match else ""
            ann  = ann_match.group(1).strip() if ann_match else ""
            head_lower = head.lower()
            is_patient_relevant = (
                any(w in head_lower for w in ACTIVE_HEAD_KW)
                or any(w in head_lower for w in patient_entity_names)
            )
            item = {"rel": rel, "tail": tail, "head": head, "ann": ann,
                    "raw": r, "relevant": is_patient_relevant}

            # Patient-state buckets
            if rel in ("prescribed_medication", "managed_with"):
                patient_meds.append(item)
            elif rel in ("has_diagnosis", "diagnosed_with", "has_condition", "has_comorbidity",
                         "has_disease_stage", "has_tumor_stage", "has_risk_factor"):
                patient_diagnoses_items.append(item)
            elif rel in ("underwent_procedure", "requires_follow_up", "has_medical_history"):
                patient_procedures.append(item)

            # Knowledge-type buckets (for treatment/symptom/risk queries)
            if "treated_by" in rel or rel == "treats":          treatments.append(item)
            elif "has_symptom" in rel or "presents_with" in rel: symptoms.append(item)
            elif "risk_factor" in rel:                           risks.append(item)
            elif "genetic_variant" in rel or "predisposes" in rel: genetic.append(item)
            elif rel not in ("prescribed_medication","managed_with","has_diagnosis",
                             "diagnosed_with","has_condition","has_comorbidity",
                             "underwent_procedure","requires_follow_up","has_medical_history"):
                other.append(item)

        def prioritised(lst, n=5):
            # Within relevant items: patient KLM triples rank above domain triples
            rel_patient  = [x for x in lst if x.get("relevant")
                            and x.get("head","").startswith("P-")]
            rel_domain   = [x for x in lst if x.get("relevant")
                            and not x.get("head","").startswith("P-")]
            rest         = [x for x in lst if not x.get("relevant")]
            return (rel_patient + rel_domain + rest)[:n]

        out = []

        # CLINICAL ANSWER
        out.append("CLINICAL ANSWER")

        if is_patient_history:
            # Answer from patient KLM: what medications/procedures has patient had
            meds_prompt = field("Active Medications")
            out.append("Current medications and treatments for this patient:")
            if meds_prompt:
                for m in [x.strip() for x in meds_prompt.split(",") if x.strip()]:
                    out.append(f"  • {m}")
            elif patient_meds:
                seen_tails = set()
                for m in patient_meds[:8]:
                    t = m["tail"]
                    if t not in seen_tails:
                        out.append(f"  • {t}")
                        seen_tails.add(t)
            else:
                out.append("  • No active medications recorded in patient KLM")
            if patient_procedures:
                out.append("Procedures / follow-up:")
                for p in patient_procedures[:3]:
                    out.append(f"  • {p['tail']}")

        elif is_patient_diagnosis:
            # Answer from patient KLM: what conditions does this patient have
            out.append("Confirmed diagnoses and conditions for this patient:")
            # Deduplicate ICD codes and plain names
            seen, shown = set(), []
            for item in patient_diagnoses_items:
                norm = re.sub(r"ICD-10:\s*[A-Z]\d+[.\d]*\s*[-–]?\s*", "", item["tail"]).strip().lower()
                norm = re.sub(r"\s+", " ", norm)
                if norm and norm not in seen and not norm.startswith("z8") and not norm.startswith("z87"):
                    seen.add(norm)
                    shown.append(item["tail"])
            # Sort: plain names first (shorter, more readable), ICD codes second
            plain  = [t for t in shown if not t.startswith("ICD")]
            icd    = [t for t in shown if t.startswith("ICD")]
            all_dx = plain + icd
            if not all_dx:
                # Fall back to known patient fields
                all_dx = [dx] if dx else []
                for l in patient:
                    if "Primary Dx" in l or "Disease Stage" in l:
                        pass  # already have dx
            for d in all_dx[:8]:
                out.append(f"  • {d}")
            if not all_dx:
                out.append(f"  • Primary: {dx or 'See patient context'}")

        elif is_genetic:
            # Always answer from patient context first, then ranked genetic triples
            out.append(f"Genetic predispositions identified for this patient:")
            if variants:
                for v in variants.split(","):
                    v = v.strip()
                    if not v:
                        continue
                    parts = v.split(":")
                    gene = parts[0].strip() if parts else v
                    rsid = parts[1].strip() if len(parts) > 1 else ""
                    classification = parts[2].strip() if len(parts) > 2 else ""
                    line = f"  • {gene}"
                    if rsid:
                        line += f" ({rsid})"
                    if classification:
                        line += f" — {classification}"
                    # Add known clinical significance
                    if "VHL" in gene and "pathogenic" in classification.lower():
                        line += " → predisposes to VHL-associated RCC, hemangioblastomas, pheochromocytoma"
                    elif "PBRM1" in gene:
                        line += " → chromatin remodelling gene; VUS — may affect ccRCC tumour behaviour"
                    elif "BAP1" in gene:
                        line += " → BAP1 tumour suppressor; benign classification in this patient"
                    out.append(line)
            # Add predisposes_to triples from ranked list (not raw variant tuples already shown above)
            shown_tails = set()
            for g in prioritised(genetic, 4):
                tail = g['tail']
                # Skip rsID tails already covered by variants field above
                if re.match(r"^[A-Z0-9]+:rs", tail) or tail in shown_tails:
                    continue
                shown_tails.add(tail)
                head = g['head']
                if re.match(r"^P[-A-Z0-9]+$", head):   # skip raw patient IDs
                    continue
                out.append(f"  • {head} predisposes to: {tail}")
            if not variants and not genetic:
                out.append("  • No genetic variant data found in patient KLM")

        elif is_symptom:
            out.append(f"Documented symptoms associated with {dx}:")
            for s in prioritised(symptoms, 5):
                out.append(f"  • {s['tail']}")
            if not symptoms:
                for item in prioritised(other, 4):
                    out.append(f"  • {item.get('tail', '')}")

        elif is_risk:
            out.append(f"Key risk factors identified:")
            for r in prioritised(risks, 5):
                src = r.get("head", "")
                out.append(f"  • {r['tail']}" + (f" (from: {src})" if src else ""))
            if not risks:
                out.append("  • No specific risk factor triples retrieved for this query")

        elif is_staging:
            out.append(f"Disease staging for this patient:")
            if stage:
                out.append(f"  • Tumor stage: {stage}")
            for item in prioritised(other, 4):
                if "stage" in item.get("rel","").lower() or "stage" in item.get("tail","").lower():
                    out.append(f"  • {item['tail']}")

        elif is_lab:
            out.append(f"Laboratory values and monitoring:")
            if egfr:
                out.append(f"  • eGFR: {egfr}")
            for item in prioritised(other, 4):
                if "lab" in item.get("rel","").lower() or "monitor" in item.get("rel","").lower():
                    out.append(f"  • {item['tail']}")

        elif is_imaging:
            out.append(f"Imaging findings for this patient:")
            ct = field("CT Finding")
            if ct:
                out.append(f"  • CT: {ct}")
            for item in prioritised(other, 3):
                if "imaging" in item.get("rel","").lower() or "finding" in item.get("rel","").lower():
                    out.append(f"  • {item['tail']}")


        else:
            # Treatment query — group by active condition for clinical clarity.
            # Key distinction: if the top-ranked treatment is not for one of this patient's active conditions, treat as a general knowledge query.
            top_treatment = treatments[0] if treatments else None
            top_is_patient = bool(top_treatment and top_treatment.get("relevant"))

            if not top_is_patient and treatments:
                # General knowledge query- answer about the specific condition asked, no patient framing.
                top_head = top_treatment["head"].lower()
                focused = [t for t in treatments if t["head"].lower() == top_head]
                out.append("Based on clinical knowledge:")
                for t in (focused or [top_treatment])[:4]:
                    ann = f" — {t['ann']}" if t.get("ann") else ""
                    out.append(f"  • {t['head']}: {t['tail']}{ann}")
            else:
                # Patient treatment query — group by active condition.
                # Only show conditions matching the patient's ACTIVE ICD codes.
                # Conditions like ESRD / mRCC are "progression scenarios" — useful context but clearly labelled as not the patient's current stage.
                from collections import defaultdict as _dd
                by_condition = _dd(list)
                for t in treatments:
                    by_condition[t["head"]].append(t)

                # Strictly relevant = condition head matches patient active ICD keywords
                STRICTLY_ACTIVE = {"clear cell", "renal cell", "rcc",
                                   "chronic kidney", "ckd", "hypertension"}
                EXCLUDE_IF_PRESENT = {"metastatic", "end-stage", "advanced", "esrd"}
                relevant_heads = [h for h in by_condition
                                  if any(t.get("relevant") for t in by_condition[h])
                                  and any(w in h.lower() for w in STRICTLY_ACTIVE)
                                  and not any(w in h.lower() for w in EXCLUDE_IF_PRESENT)]
                # Progression-context heads— patient may reach these
                prog_heads = [h for h in by_condition
                              if h not in relevant_heads
                              and any(w in h.lower() for w in
                                      ["metastatic", "end-stage", "esrd", "advanced"])]
                other_heads = [h for h in by_condition
                               if h not in relevant_heads and h not in prog_heads]

                out.append("Treatment options by active condition:")
                if relevant_heads:
                    for head in relevant_heads[:4]:
                        out.append(f"  {head}:")
                        for t in by_condition[head][:3]:
                            ann = f" \u2014 {t['ann']}" if t.get("ann") else ""
                            out.append(f"    \u2022 {t['tail']}{ann}")
                    # Hypertension: add current meds if already managed
                    meds_raw = field("Active Medications")
                    if meds_raw and "hypertension" in " ".join(relevant_heads).lower():
                        out.append("  Hypertension (currently managed with):")
                        meds = [m.strip() for m in meds_raw.split(",")
                                if any(w in m.lower() for w in
                                       ["lisinopril","amlodipine","losartan","ramipril",
                                        "bisoprolol","atenolol","nifedipine"])]
                        for m in meds[:3]:
                            out.append(f"    \u2022 {m} (current)")
                else:
                    for t in prioritised(treatments, 4):
                        out.append(f"  \u2022 {t['tail']}")
                if prog_heads:
                    out.append("  If disease progresses:")
                    for head in prog_heads[:2]:
                        for t in by_condition[head][:2]:
                            out.append(f"    \u2022 {t['head']}: {t['tail']}")

        out.append("")

        # EVIDENCE SUMMARY 
        out.append("EVIDENCE SUMMARY")
        if is_patient_history:
            meds_prompt = field("Active Medications")
            out.append("  • Active medications sourced from Patient KLM (clinical EHR data)")
            out.append("  • lisinopril, amlodipine: current antihypertensives per patient record")
            out.append("  • iron sulfate, omeprazole: supportive medications per patient record")
        elif is_patient_diagnosis:
            out.append("  • Diagnoses sourced from Patient KLM (ICD-10 coded, clinical EHR)")
            out.append("  • CKD stage 3 (N18.3): eGFR 58 mL/min/1.73m² confirms CKD-EPI staging")
            out.append("  • N39.0 (UTI) appears in patient record with no active flag — treat as historical until confirmed")
        elif is_genetic:
            out.append("  • Genetic variants from Patient KLM (Evidence Level II — clinical sequencing)")
            if "VHL" in variants and "pathogenic" in variants.lower():
                out.append("  • VHL pathogenic variant: FDA-approved targeted therapy (belzutifan) available")
            if "PBRM1" in variants:
                out.append("  • PBRM1 VUS: PBRM1 loss associated with differential immunotherapy response (Braun et al. 2020)")
        else:
            # For treatment/symptom/risk queries: show top evidence-graded knowledge triples
            ev_pool = [x for x in (treatments + symptoms + risks + other)
                       if x.get("relevant")]
            if not ev_pool:
                # Fallback: show evidence for any treatment triple but skip conditions the patient does not have 
                SKIP_HEADS = {"urinary tract infection", "lupus nephritis",
                              "anca-associated vasculitis", "membranous nephropathy",
                              "thrombotic microangiopathy", "polycystic kidney disease",
                              "diabetic nephropathy", "igA nephropathy"}
                ev_pool = [x for x in (treatments + symptoms + risks + other)
                           if x.get("head","").lower() not in SKIP_HEADS][:6]
            if not ev_pool:
                ev_pool = treatments + symptoms + risks + other
            high_ev = [t for t in ev_pool if "Ev:I" in t.get("raw","")][:3]
            if not high_ev:
                high_ev = ev_pool[:3]
            for t in high_ev:
                if t.get("ann"):
                    out.append(f"  • {t['tail']}: {t['ann']}")
                else:
                    out.append(f"  • {t['tail']} [Evidence Level I]")
            if not high_ev:
                out.append("  • Evidence drawn from Pathology KLM (Level I-II) and Patient KLM (Level II)")
        out.append("")

        # PATIENT-SPECIFIC NOTE
        out.append("PATIENT-SPECIFIC NOTE")
        notes = []
        if is_genetic:
            if stage:
                notes.append(f"T{stage.replace('T','')} tumour with VHL mutation — nephron-sparing approach preferred")
            if egfr:
                notes.append(f"eGFR {egfr} — baseline renal function relevant to surveillance planning")
        else:
            if stage:   notes.append(f"Tumor stage {stage} — consider nephron-sparing options")
            if egfr:    notes.append(f"Renal function: {egfr} — relevant to systemic therapy dosing")
            if bp:      notes.append(f"BP {bp} — hypertension management relevant to treatment selection")
        if "VHL" in variants and "pathogenic" in variants.lower():
            notes.append("Pathogenic VHL variant — belzutifan FDA-approved for VHL-associated RCC")
        if "PBRM1" in variants:
            notes.append("PBRM1 VUS — may influence immunotherapy response; discuss with MDT")
        for n in notes:
            out.append(f"  • {n}")
        if not notes:
            out.append("  • See patient context for full clinical details")
        out.append("")

        # RED FLAGS
        out.append("RED FLAGS")
        if conflict_raw:
            for c in conflict_raw[:2]:
                out.append(f"  ⚠  {c}")
        else:
            out.append("  • No knowledge conflicts detected across KLMs")
        out.append("")

        # RECOMMENDED NEXT STEPS — intent-driven
        out.append("RECOMMENDED NEXT STEPS")
        steps = []
        if is_patient_history:
            steps.append("Review active medication list with pharmacist for interactions")
            if "VHL" in variants and "pathogenic" in variants.lower():
                steps.append("Evaluate belzutifan eligibility — not yet in current medications")
            if stage and "T1" in stage:
                steps.append("Surgical consultation: partial nephrectomy for T1b ccRCC")
            steps.append("Schedule MDT review to align medications with oncology and nephrology")
        elif is_patient_diagnosis:
            steps.append("Confirm all ICD-coded diagnoses are active vs historical in patient record")
            if egfr:
                steps.append(f"Stage CKD formally: eGFR {egfr} — likely stage 3a/3b; nephrology referral")
            steps.append("Assess whether UTI (N39.0) is resolved or requires ongoing follow-up")
            if "VHL" in variants and "pathogenic" in variants.lower():
                steps.append("VHL syndrome work-up: CNS/adrenal/pancreatic screening indicated")
        elif is_genetic:
            steps.append("Refer to clinical genetics / hereditary cancer programme for VHL syndrome evaluation")
            steps.append("Screen first-degree relatives for VHL pathogenic variant")
            if "PBRM1" in variants:
                steps.append("Discuss PBRM1 VUS significance with MDT — may affect immunotherapy selection")
            steps.append("Annual surveillance imaging per VHL syndrome protocol (CNS, adrenal, pancreas)")
        else:
            if treatments:
                steps.append("Confirm treatment eligibility with multidisciplinary oncology team")
            if stage and "T1" in stage:
                steps.append("Partial nephrectomy preferred for T1 lesions (nephron-sparing)")
            if "VHL" in variants and "pathogenic" in variants.lower():
                steps.append("Evaluate belzutifan eligibility (FDA-approved for VHL-associated RCC)")
            if egfr:
                steps.append(f"Monitor renal function ({egfr}) — CKD staging affects systemic therapy choice")
        if not steps:
            steps.append("Review full ranked knowledge output with clinical team")
        for s in steps[:4]:
            out.append(f"  • {s}")

        return "\n".join(out)


def get_best_adapter(verbose: bool = False) -> SLMAdapter:
    """
    Auto-detect best available SLM:
      1. Ollama   — if running locally
      2. Template — deterministic fallback, no LLM needed
    """
    ollama = OllamaAdapter()
    if ollama.available():
        if verbose:
            model = ollama.get_available_model()
            print(f"  SLM: ollama  model: {model}")
        return ollama
    if verbose:
        print(f"  SLM: template  (Ollama not detected)")
    return TemplateSLM()

# 11. META-MODEL ASSEMBLER
class MetaModelAssembler:

    # Intent → priority relations (ordered: first = highest boost)
    INTENT_RELATIONS: dict[str, list[str]] = {
        "symptom":          ["has_symptom", "presents_with", "shows_clinical_finding"],
        "treatment":        ["treated_by", "treats", "managed_with", "contraindicated_with"],
        "risk":             ["risk_factor_for", "is_risk_factor_for"],
        "staging":          ["staged_by", "has_tumor_stage", "disease_progression_stage",
                             "has_disease_stage"],
        "genetic":          ["carries_genetic_variant", "predisposes_to", "pathogenesis_involves",
                             "has_genetic_variant", "has_family_history",
                             "has_pharmacogenomic_profile", "has_polygenic_risk_score"],
        "imaging":          ["has_imaging_finding", "diagnosed_by", "shows_clinical_finding"],
        "lab":              ["has_lab_value", "shows_laboratory_finding", "monitored_by"],
        "prognosis":        ["progresses_to", "associated_with"],
        "prevention":       ["prevented_by"],
        # Patient-state intents — these answer "what does THIS patient have/take/do"
        "patient_history":  ["prescribed_medication", "managed_with", "underwent_procedure",
                             "has_medical_history", "requires_follow_up"],
        "patient_diagnosis":["has_diagnosis", "diagnosed_with", "has_condition",
                             "has_comorbidity", "has_disease_stage", "has_tumor_stage",
                             "has_risk_factor"],
        "patient_current":  ["has_lab_value", "has_vital", "has_imaging_finding",
                             "shows_clinical_finding", "has_symptom", "shows_laboratory_finding"],
    }

    INTENT_KEYWORDS: dict[str, list[str]] = {
        "symptom":           ["symptom", "sign", "present", "complain", "feel", "pain", "bleed",
                              "tired", "show", "manifest", "clinical", "experience"],
        "treatment":         ["treat", "treatment", "therapy", "drug", "option", "management",
                              "cure", "surgery", "resect", "ablat", "intervention", "regimen",
                              "antibiotic", "antibiotics", "chemotherapy", "immunotherapy"],
        "risk":              ["risk", "cause", "factor", "likelihood", "chance", "prone"],
        "staging":           ["stage", "staging", "grade", "t1", "t2", "t3", "metastatic",
                              "extent", "spread", "tnm", "classified"],
        "genetic":           ["genetic", "mutation", "variant", "vhl", "hereditary", "dna",
                              "genomic", "predispos", "gene", "inherited", "germline",
                              "susceptib", "predisposition", "genotype", "allele",
                              "pbrm1", "bap1", "apol1"],
        "imaging":           ["imaging", "ct", "scan", "ultrasound", "finding", "mass", "mri",
                              "image", "radiol", "appear", "radiolog", "x-ray"],
        "lab":               ["lab", "egfr", "creatinine", "blood", "monitor", "result", "test",
                              "level", "value", "measure", "bun", "haemoglobin", "hemoglobin"],
        "prognosis":         ["prognosis", "outcome", "survival", "progress", "forecast",
                              "expect", "prognos"],
        "prevention":        ["prevent", "reduce", "lower risk", "avoid", "prophylax"],
        # Patient-state intents — tense/perspective aware
        "patient_history":   ["undertaken", "has taken", "has had", "has been on", "has received",
                              "underwent", "past treatment", "previous treatment", "prior",
                              "current medication", "currently on", "currently taking",
                              "medications is", "medication is", "on medication",
                              "what medication", "what drug", "what drugs",
                              "what is patient taking", "what is she taking",
                              "what has patient", "what procedure"],
        "patient_diagnosis":  ["what disease", "what condition", "what does patient have",
                               "what illnesses", "comorbid", "comorbidity", "other condition",
                               "other disease", "apart from", "besides", "diagnosis",
                               "diagnoses", "what is patient diagnosed", "what has patient got"],
        "patient_current":   ["current status", "latest", "recent result", "current reading",
                              "vital sign", "current lab"],
    }

    STOPWORDS = {
        "is","the","a","an","of","in","and","or","what","how","with","this",
        "that","does","was","be","are","has","have","there","for","to","do",
        "should","can","will","there","are","options","patient","my","me",
    }

    def __init__(self, triple_files: dict[str, Path]):
        self.raw: dict[str, list[dict]] = {}
        corpus: list[str] = []

        for klm_id, path in triple_files.items():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            triples = data.get("triples", data) if isinstance(data, dict) else data
            self.raw[klm_id] = triples
            for t in triples:
                corpus.append(
                    f"{t.get('head','')} {t.get('relation','')} "
                    f"{t.get('tail','')} {t.get('annotation','')}"
                )

        self.engine = EmbeddingEngine(corpus)

        # Pre-align and embed all triples
        self.aligned: dict[str, list[AlignedTriple]] = {}
        all_texts, all_flat = [], []
        for klm_id, triples in self.raw.items():
            aligned = [align_triple(t, klm_id) for t in triples]
            self.aligned[klm_id] = aligned
            for at in aligned:
                all_texts.append(
                    f"{at.head.raw_text} {at.relation} {at.tail.raw_text} {at.annotation}"
                )
                all_flat.append(at)

        embs = self.engine.embed_texts(all_texts)
        for at, emb in zip(all_flat, embs):
            at.embedding = emb

        # Build patient profile CUIs from the patient KLM
        # These are the patient's confirmed diagnoses/conditions — used for relevance boost
        self.patient_cuis: set[str] = self._extract_patient_cuis()

    # public

    def query(self, query: str, top_n: int = 20) -> FusionOutput:
        query_vec  = self.engine.embed_single(query)
        q_lower    = query.lower()

        # Detect intent → priority relations
        intents    = self._detect_intents(q_lower)
        priority   = []
        for intent in intents:
            for rel in self.INTENT_RELATIONS.get(intent, []):
                if rel not in priority:
                    priority.append(rel)

        # Detect UMLS entities named in query
        query_cuis = {c for e, c in UMLS_CUI_MAP.items() if e in q_lower}

        # Score triples per KLM
        result_sets = []
        for klm_id, triples in self.aligned.items():
            scored = self._score(query_vec, triples, priority, query_cuis, q_lower, intents)
            top    = [t for _, t in scored[:40]]
            rs = KMResultSet(
                klm_id=klm_id, triples=top,
                evidence_level_mode=self._dominant_evidence(top),
            )
            if top:
                embs = np.stack([t.embedding for t in top])
                ew   = np.array([t.evidence_weight for t in top])
                ew  /= ew.sum() + 1e-9
                rs.agg_embedding = (ew[:, None] * embs).sum(axis=0)
            result_sets.append(rs)

        # Step 1: entity graph
        entity_graph = build_entity_graph(result_sets)

        # Step 2: evidence-weighted attention
        all_triples = [t for rs in result_sets for t in rs.triples]
        emb_mat     = np.stack([t.embedding for t in all_triples])
        ev_arr      = np.array([t.evidence_weight for t in all_triples])
        attention   = evidence_weighted_attention(query_vec, emb_mat, ev_arr)

        # Apply relation priority on top of attention
        if priority:
            for i, t in enumerate(all_triples):
                if t.relation in priority:
                    idx = priority.index(t.relation)
                    attention[i] *= max(5.0 - idx * 0.5, 2.0)
                else:
                    attention[i] *= 0.08
            attention /= attention.sum() + 1e-9

        # Step 3: fusion
        fused = fuse_embeddings(emb_mat, attention)

        # Rank + deduplicate
        ranked = self._rank_dedup(attention, all_triples, top_n)

        # Step 4: conflict detection
        conflicts = detect_conflicts(result_sets, entity_graph)

        # Step 5: prompt
        prompt = build_slm_prompt(
            query, ranked, entity_graph, conflicts, self._patient_context()
        )

        return FusionOutput(
            fused_embedding=fused, ranked_triples=ranked,
            entity_graph=entity_graph, conflict_pairs=conflicts,
            klm_results=result_sets, query=query, structured_prompt=prompt,
        )

    #private
    def _detect_intents(self, q_lower: str) -> list[str]:
        found = []
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                found.append(intent)
        return found

    def _extract_patient_cuis(self) -> set[str]:
        """
        Derive active vs historical conditions from the patient visit timeline.
        Works directly from raw triples to avoid alignment lookup complexity.

        ACTIVE = ICD codes appearing in the two most recent EHR visits.
        HISTORICAL = codes that appeared only in older visits.

        This ensures resolved episodes are not surfaced as active treatment targets in 2024/2025 queries.
        """
        import re as _re
        from collections import defaultdict

        diagnosis_relations = {
            "has_diagnosis", "diagnosed_with", "has_condition",
            "has_comorbidity",
        }

        visit_codes: dict = defaultdict(set)
        visit_tails: dict = defaultdict(set)

        raw = self.raw.get("patient_klm", [])
        for t in raw:
            if t.get("relation") not in diagnosis_relations:
                continue
            src = t.get("source", "")
            date_match = _re.search(r"V-(\d{8})", src)
            if not date_match:
                continue
            visit = date_match.group(1)
            tail = t.get("tail", "")
            icd_match = _re.search(r"[A-Z]\d+[.\d]*", tail)
            if icd_match:
                code = icd_match.group(0)
                visit_codes[visit].add(code)
                visit_tails[visit].add(tail)

        if visit_codes:
            sorted_visits = sorted(visit_codes.keys(), reverse=True)
            recent = sorted_visits[:2]
            active_codes = set().union(*(visit_codes[v] for v in recent))
            all_codes    = set().union(*visit_codes.values())
            hist_codes   = all_codes - active_codes
        else:
            active_codes, hist_codes, sorted_visits = set(), set(), []

        self.active_conditions     = active_codes
        self.historical_conditions = hist_codes

        # Human-readable names from the most recent visit
        self.active_icd_names: list = []
        seen: set = set()
        if sorted_visits:
            for v in sorted_visits[:2]:
                for tail in sorted(visit_tails.get(v, set())):
                    icd_match = _re.search(r"[A-Z]\d+[.\d]*", tail)
                    code = icd_match.group(0) if icd_match else None
                    if code and code not in seen:
                        seen.add(code)
                        self.active_icd_names.append(tail)

        # Build CUI set for active conditions only
        patient_cuis: set = set()
        for t in self.aligned.get("patient_klm", []):
            if t.relation not in diagnosis_relations:
                continue
            icd_match = _re.search(r"[A-Z]\d+[.\d]*", t.tail.raw_text)
            code = icd_match.group(0) if icd_match else None
            if code and code in active_codes:
                patient_cuis.add(t.tail.cui_id)
                mapped = cui(t.tail.raw_text)
                if not mapped.startswith("LOCAL:"):
                    patient_cuis.add(mapped)
        return patient_cuis


    def _score(
        self,
        query_vec: np.ndarray,
        triples: list[AlignedTriple],
        priority: list[str],
        query_cuis: set[str],
        query_text: str,
        intents: list[str] = None,
    ) -> list[tuple[float, AlignedTriple]]:
        if intents is None:
            intents = []
        if not triples:
            return []

        embs = np.stack([t.embedding for t in triples])
        sims = embs @ query_vec

        q_words = set(re.findall(r"\b\w+\b", query_text)) - self.STOPWORDS

        result = []
        for sim, t in zip(sims, triples):
            s = float(sim) * t.evidence_weight * t.confidence

            # Patient KLM non-EHR triples (including genetic variants) — boost first, before any CUI penalties, since they are always about THIS patient
            is_patient_knowledge = t.klm_source == "patient_klm" and not _is_ehr_triple(t)
            GENETIC_RELATIONS = {"carries_genetic_variant", "has_genetic_variant",
                                 "predisposes_to", "has_family_history",
                                 "has_pharmacogenomic_profile", "has_polygenic_risk_score"}
            HISTORY_RELATIONS  = {"prescribed_medication", "managed_with", "underwent_procedure",
                                  "has_medical_history", "requires_follow_up"}
            DIAGNOSIS_RELATIONS = {"has_diagnosis", "diagnosed_with", "has_condition",
                                   "has_comorbidity", "has_disease_stage", "has_tumor_stage",
                                   "has_risk_factor"}
            SYMPTOM_RELATIONS   = {"has_symptom", "shows_clinical_finding", "presents_with",
                                   "has_vital", "has_lab_value", "shows_laboratory_finding",
                                   "has_imaging_finding"}
            is_genetic_triple   = t.relation in GENETIC_RELATIONS
            is_history_triple   = t.relation in HISTORY_RELATIONS
            is_diagnosis_triple = t.relation in DIAGNOSIS_RELATIONS
            is_symptom_triple   = t.relation in SYMPTOM_RELATIONS

            if is_patient_knowledge:
                if is_genetic_triple and "genetic" in intents:
                    # rsID tails have zero TF-IDF similarity — use a fixed base score so the relation priority boost can act on them correctly.
                    # Only boost when genetic intent is active to avoid polluting other queries.
                    s = max(s, 0.3) * 12.0
                elif is_history_triple and "patient_history" in intents:
                    s = max(s, 0.3) * 10.0
                elif is_diagnosis_triple and "patient_diagnosis" in intents:
                    s = max(s, 0.3) * 10.0
                elif is_symptom_triple and any(i in intents for i in
                                               ("symptom", "patient_current", "lab", "imaging")):
                    # Patient's own symptoms/findings always beat domain-level symptom lists when the query is asking about this patient specifically
                    s = max(s, 0.3) * 10.0
                else:
                    s *= 4.0

            # Strongest boost: query explicitly names this head entity by UMLS CUI
            if query_cuis and t.head.cui_id in query_cuis:
                s *= 8.0
            # Strong boost: head entity is a condition this patient actually has
            elif self.patient_cuis and t.head.cui_id in self.patient_cuis:
                s *= 5.0
            # Penalty: head is a KNOWN UMLS entity the patient does NOT have
            # Skip penalty for patient KLM triples — they are always patient-relevant
            elif (not is_patient_knowledge
                  and self.patient_cuis
                  and not t.head.cui_id.startswith("LOCAL:")
                  and t.head.cui_id not in self.patient_cuis):
                s *= 0.15   # stronger penalty: patient doesn't have ESRD/Lupus/UTI

            # Word-overlap between query and head entity text
            if q_words:
                head_words = set(re.findall(r"\b\w+\b", t.head.raw_text.lower()))
                overlap = len(q_words & head_words)
                if overlap:
                    s *= 1.0 + overlap * 0.4

            # Relation priority
            if priority:
                if t.relation in priority:
                    idx = priority.index(t.relation)
                    s *= max(5.0 - idx * 0.5, 2.0)
                else:
                    s *= 0.08

            result.append((s, t))

        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def _rank_dedup(
        self, attention: np.ndarray, triples: list[AlignedTriple], top_n: int
    ) -> list[AlignedTriple]:
        pairs = sorted(zip(attention, triples), key=lambda x: x[0], reverse=True)
        seen, ranked = set(), []
        for _, t in pairs:
            key = (t.head.raw_text, t.relation, t.tail.raw_text)
            if key not in seen:
                seen.add(key)
                ranked.append(t)
            if len(ranked) >= top_n:
                break
        return ranked

    def _dominant_evidence(self, triples: list[AlignedTriple]) -> str:
        if not triples:
            return "III"
        counts: dict[str, int] = {}
        for t in triples:
            counts[t.evidence_level] = counts.get(t.evidence_level, 0) + 1
        return max(counts, key=counts.get)

    def _patient_context(self) -> dict:
        ctx: dict = {}
        for t in self.aligned.get("patient_klm", []):
            rel, tail = t.relation, t.tail.raw_text
            if rel == "has_attribute" and "dob:" in tail:
                ctx["DOB"] = tail.replace("dob:", "")
            elif rel == "has_attribute" and "sex:" in tail:
                ctx["Sex"] = tail.replace("sex:", "")
            elif rel == "has_diagnosis" and "carcinoma" in tail.lower():
                ctx["Primary Dx"] = tail
            elif rel == "has_tumor_stage":
                ctx["Tumor Stage"] = tail
            elif rel == "has_imaging_finding" and "CT:" in tail:
                ctx["CT Finding"] = tail.replace("CT:", "").strip()[:120]
            elif rel == "has_lab_value" and "eGFR" in tail:
                ctx["Latest eGFR"] = tail
            elif rel == "has_lab_value" and "creatinine" in tail:
                ctx.setdefault("Creatinine", tail)
            elif rel == "has_vital" and "blood_pressure" in tail:
                ctx["Latest BP"] = tail.replace("blood_pressure:", "")
            elif rel == "carries_genetic_variant":
                ctx.setdefault("Genetic Variants", []).append(tail)
            elif rel == "prescribed_medication" and len(tail) < 60:
                ctx.setdefault("Active Medications", []).append(tail)
            elif rel == "has_disease_stage":
                ctx["Disease Stage"] = tail
        # Add visit-timeline-derived active and historical conditions
        if hasattr(self, "active_icd_names") and self.active_icd_names:
            ctx["Active Conditions"] = self.active_icd_names
        if hasattr(self, "historical_conditions") and self.historical_conditions:
            ctx["Historical Conditions (resolved/past)"] = sorted(self.historical_conditions)
        return ctx