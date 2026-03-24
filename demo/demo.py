# Meta-Model Demo

import os, sys, json, time, textwrap, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from meta_model import (
    MetaModelAssembler, find_merged_entities, get_best_adapter,
    _is_ehr_triple, UMLS_CUI_MAP,
)

# ANSI colours 
R="\033[0m"; B="\033[1m"; DIM="\033[2m"
RED="\033[91m"; GRN="\033[92m"; YLW="\033[93m"
BLU="\033[94m"; MAG="\033[95m"; CYN="\033[96m"; WHT="\033[97m"

KLM_CLR = {"patient_klm": CYN, "cardiology_klm": MAG, "nephrology_klm": YLW, "hypertension_klm": BLU} # for different klms
KLM_LBL = {"patient_klm": "PATIENT  ", "cardiology_klm": "CARDIO   ",
    "nephrology_klm": "NEPHRO   ",
    "hypertension_klm": "HTN      "}
EV_CLR  = {"I": GRN, "II": GRN, "III": YLW, "IV": YLW, "V": RED} # evidence level

BASE = Path(__file__).parent


def hr(ch="─", w=72, c=DIM):   print(f"{c}{ch*w}{R}")
def hdr(title, c=CYN):         print(f"\n{B}{c}▌ {title}{R}"); hr(c=c+DIM)
def wrap(s, ind=4, w=68):
    return textwrap.fill(s, w, initial_indent=" "*ind, subsequent_indent=" "*ind)

# DISPLAY
def show_step1(output, assembler):
    hdr("Step 1: Entity Alignment  [UMLS CUI]", CYN)
    merged = find_merged_entities(output.entity_graph)
    total  = len(output.entity_graph)
    print(f"  Unique CUIs in cross-KM graph : {B}{total}{R}")
    print(f"  Entities confirmed across KLMs: {B}{GRN}{len(merged)}{R}")
    if merged:
        print()
        for m in merged[:6]:
            names = " / ".join(m["raw_texts"][:2])
            srcs  = "  +  ".join(
                f"{KLM_CLR.get(s,WHT)}{KLM_LBL.get(s,s)}{R}"
                for s in sorted(m["klm_sources"])
            )
            print(f"  {DIM}CUI {m['cui']:20}{R}  {WHT}{names[:45]}{R}")
            print(f"    {DIM}confirmed in: {srcs}{R}")


def show_step2(output):
    """Show top-ranked triples, split by patient-relevant vs general."""
    hdr("Step 2: Evidence-Weighted Attention  [α = softmax(QKᵀ/√d) · w_ev]", BLU)

    patient_cuis = getattr(output, "_patient_cuis", set())

    # Split into patient-relevant (head CUI in patient profile or patient KLM triple) vs general knowledge
    relevant, general = [], []
    for t in output.ranked_triples[:16]:
        if t.klm_source == "patient_klm" and not _is_ehr_triple(t):
            relevant.append(t)
        elif t.head.cui_id in (output._patient_cuis if hasattr(output, "_patient_cuis") else set()):
            relevant.append(t)
        else:
            general.append(t)

    def print_triple_row(t, rank):
        c   = KLM_CLR.get(t.klm_source, WHT)
        lbl = KLM_LBL.get(t.klm_source, t.klm_source)
        ec  = EV_CLR.get(t.evidence_level, WHT)
        head = t.head.raw_text[:24]
        tail = t.tail.raw_text[:28]
        print(f"  {c}{lbl}{R}  {DIM}{t.relation[:22]:22}{R}  "
              f"{ec}{t.evidence_level:3}{R}"
              f"{B}{head}{R} → {tail}")

    if relevant:
        print(f"\n  {GRN}{B}Patient-relevant (by diagnosis profile):{R}")
        for i, t in enumerate(relevant[:8]):
            print_triple_row(t, i)

    if general:
        print(f"\n  {DIM}General domain knowledge (also in context):{R}")
        for i, t in enumerate(general[:4]):
            print_triple_row(t, i + len(relevant))


def show_step3(output):
    hdr("Step 3: Fusion  [h_fused = Σ αᵢ · eᵢ]", MAG)
    for rs in output.klm_results:
        c   = KLM_CLR.get(rs.klm_id, WHT)
        lbl = KLM_LBL.get(rs.klm_id, rs.klm_id)
        if rs.agg_embedding is not None:
            norm = float((rs.agg_embedding**2).sum()**0.5)
            print(f"  {c}{B}{lbl}{R}  n={rs.triples.__len__():3}  "
                  f"evidence level={rs.evidence_level_mode}  ‖emb‖={norm:.4f}") #||emb|- magnitude of embeddings
    fn = float((output.fused_embedding**2).sum()**0.5)
    print(f"\n  {GRN}{B}h_fused{R}  dim={output.fused_embedding.shape[0]}  ‖h‖={fn:.4f}  "
          f"{DIM}[weighted sum across all KLMs]{R}")


def show_step4(output):
    hdr("Step 4: Conflict Detection", RED)
    if not output.conflict_pairs:
        print(f"  {GRN}✔ No conflicts detected across KLMs.{R}")
        return

    print(f"  {RED}{B}{len(output.conflict_pairs)} genuine conflict(s){R} detected:\n")
    for c in output.conflict_pairs[:3]:
        sc = RED if c["severity"] == "HIGH" else YLW
        a, b = c["triple_a"], c["triple_b"]
        print(f"  {sc}{B}[{c['severity']}]{R} {WHT}{c['entity']}{R}  —  {c['conflict_type']}")
        print(f"    {KLM_CLR.get(a['klm'],WHT)}{KLM_LBL.get(a['klm'],a['klm'])}{R}: "
              f"{a['head']} –[{a['relation']}]→ {a['tail']} [{a['evidence']}]")
        print(f"    {KLM_CLR.get(b['klm'],WHT)}{KLM_LBL.get(b['klm'],b['klm'])}{R}: "
              f"{b['head']} –[{b['relation']}]→ {b['tail']} [{b['evidence']}]")
        print()


def show_step5(response: str, adapter_name: str):
    hdr(f"Step 5: SLM Output  [{adapter_name}]", GRN)

    SECTIONS = {
        "CLINICAL ANSWER":        (GRN,  "💊 Clinical Answer"),
        "EVIDENCE SUMMARY":       (BLU,  "📄 Evidence Summary"),
        "PATIENT-SPECIFIC NOTE":  (CYN,  "👤 Patient-Specific Note"),
        "RED FLAGS":              (RED,  "⚠  Red Flags"),
        "RECOMMENDED NEXT STEPS": (YLW,  "📋 Next Steps"),
    }

    cur_col, cur_lbl, buf = WHT, "", []

    def flush():
        if not buf or not cur_lbl:
            return
        print(f"\n  {B}{cur_col}{cur_lbl}{R}")
        for line in buf:
            line = line.rstrip()
            if not line:
                continue
            is_bullet = re.match(r"^(\s*[•\-]\s?|\s{2,}[•\-]\s?)", line)
            is_sub    = re.match(r"^\s{4,}", line) and not is_bullet
            if is_bullet:
                leader_m = re.match(r"^(\s*[•\-]\s?)", line)
                leader   = leader_m.group(1) if leader_m else "  • "
                body     = line[len(leader):]
                indent   = "    " + " " * len(leader)
                print(textwrap.fill(
                    leader + body,
                    width=72,
                    initial_indent="    ",
                    subsequent_indent=indent,
                ))
            elif is_sub:
                print(f"    {line.strip()}")
            else:
                print(textwrap.fill(
                    line.strip(), width=72,
                    initial_indent="    ",
                    subsequent_indent="    ",
                ))

    for line in response.split("\n"):
        matched = False
        for key, (col, lbl) in SECTIONS.items():
            if key in line.upper():
                flush()
                buf = []
                cur_col, cur_lbl = col, lbl
                rest = line.split("—", 1)[-1].strip() if "—" in line else ""
                if rest:
                    buf.append(rest)
                matched = True
                break
        if not matched:
            buf.append(line)

    flush()

# MAIN
def main():
    # Header
    hr("═", c=CYN+B)
    print(f"{B}{CYN} LOOP KM  │  META-MODEL ASSEMBLER {R}")
    hr("═", c=CYN+B)
    print()

    # Load
    print(f"  {DIM}Loading KLMs...{R}", end="", flush=True)
    t0 = time.time()
    assembler = MetaModelAssembler({
    "patient_klm":    BASE / "patient_triples.json",
    "cardiology_klm":    BASE / "cardiology_triples.json",
    "nephrology_klm":    BASE / "nephrology_triples.json",
    "hypertension_klm":  BASE / "hypertension_triples.json",
})

    # Auto-detect SLM
    slm = get_best_adapter(verbose=False)
    elapsed = time.time() - t0

    counts = {k: len(v) for k, v in assembler.aligned.items()}
    print(f"\r  {GRN}✔ Ready{R}  ({elapsed:.1f}s)  "
          f"{DIM}—{R}  "
          f"{CYN}Patient:{counts.get('patient_klm',0)}{R}  "
          f"{MAG}Cardio:{counts.get('cardiology_klm',0)}{R}  "
          f"{YLW}Nephro:{counts.get('nephrology_klm',0)}{R}  "
          f"{BLU}HTN:{counts.get('hypertension_klm',0)}{R}  "
          f"triples")

    # SLM status
    slm_color = GRN if slm.name != "template" else YLW
    print(f"  {slm_color}SLM:{R} {B}{slm.name}{R}", end="")
    if slm.name == "template":
        print(f"  {DIM}(no local LLM detected){R}")
    elif slm.name == "ollama":
        model = slm.get_available_model()
        print(f"  {DIM}model: {model}{R}")
    else:
        print()

    while True:
        try:
            query = input(f"{B}{BLU}  Doctor Query:  {R}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Session ended.{R}\n"); break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print(f"\n{DIM}Session ended.{R}\n"); break

        print()
        t0 = time.time()
        try:
            output = assembler.query(query)
        except Exception as e:
            print(f"  {RED}Pipeline error: {e}{R}"); continue

        # Attach patient CUIs to output for display
        output._patient_cuis = assembler.patient_cuis

        pipeline_ms = (time.time() - t0) * 1000

        # Display pipeline steps
        show_step1(output, assembler)
        show_step2(output)
        show_step3(output)
        show_step4(output)

        # SLM call
        hdr(f"Step 5: SLM Output [{slm.name}]", GRN)
        print(f"  {DIM}Ranked triples: {len(output.ranked_triples)}  "
              f"| Prompt: {len(output.structured_prompt)} chars  "
              f"| Pipeline: {pipeline_ms:.0f}ms{R}")

        active_slm = slm   # may switch to template on error
        if slm.name == "template":
            print(f"  {YLW}Using template synthesis...{R}\n")

        # For Ollama: stream tokens live 
        is_ollama = hasattr(slm, "generate") and "ollama" in slm.name.lower()
        streaming_started = False

        def on_token(tok: str):
            nonlocal streaming_started
            if not streaming_started:
                print(f"[streaming]", GRN)
                print(f"  {GRN}💊{R} ", end="", flush=True)
                streaming_started = True
            print(tok, end="", flush=True)

        t1 = time.time()
        try:
            if is_ollama:
                response = slm.generate(output.structured_prompt,
                                        stream_callback=on_token)
                slm_ms = (time.time() - t1) * 1000
                if streaming_started:
                    # End streaming block cleanly
                    print(f"\n\n  {DIM}({slm_ms:.0f}ms){R}")
                    hr()
                    merged = find_merged_entities(output.entity_graph)
                    print(f"  {DIM}Total: {pipeline_ms+slm_ms:.0f}ms  "
                          f"| Conflicts: {len(output.conflict_pairs)}  "
                          f"| Cross-KM entities: {len(merged)}{R}")
                    hr()
                    continue  # skip the normal show_step5 + footer below
            else:
                if slm.name != "template":
                    print(f"  {DIM}Generating...{R}", end="", flush=True)
                response = slm.generate(output.structured_prompt)
                slm_ms = (time.time() - t1) * 1000
                if slm.name != "template":
                    print(f"\r  {GRN}✔ Generated{R}  ({slm_ms:.0f}ms)\n")
        except RuntimeError as e:
            slm_ms = (time.time() - t1) * 1000
            if streaming_started:
                print()
            print(f"  {YLW}⚠  {e}{R}")
            print(f"  {DIM}Falling back to template output...{R}\n")
            from meta_model import TemplateSLM
            active_slm = TemplateSLM()
            response = active_slm.generate(output.structured_prompt)
        except Exception as e:
            slm_ms = (time.time() - t1) * 1000
            if streaming_started:
                print()
            print(f"  {RED}SLM error: {e}{R}")
            print(f"  {DIM}Falling back to template output...{R}\n")
            from meta_model import TemplateSLM
            active_slm = TemplateSLM()
            response = active_slm.generate(output.structured_prompt)

        show_step5(response, active_slm.name)

        hr()
        merged = find_merged_entities(output.entity_graph)
        print(f"  {DIM}Total: {pipeline_ms+slm_ms:.0f}ms  "
              f"| Conflicts: {len(output.conflict_pairs)}  "
              f"| Cross-KM entities: {len(merged)}{R}")

        # Optional: show raw prompt
        try:
            show = input(f"\n  {DIM}Show raw SLM prompt? (Y/N): {R}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break
        if show == "y":
            print(f"\n{DIM}{'─'*72}")
            print(output.structured_prompt)
            print(f"{'─'*72}{R}")


if __name__ == "__main__":
    main()
