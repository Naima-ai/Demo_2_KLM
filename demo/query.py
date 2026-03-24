# -*- coding: utf-8 -*-
import sys
import json
import time
import argparse
from pathlib import Path

# locate KLM data files relative to this script
BASE = Path(__file__).parent

KLM_PATHS = {
    "patient_klm":    BASE / "patient_triples.json",
    "cardiology_klm":    BASE / "cardiology_triples.json",
    "nephrology_klm":    BASE / "nephrology_triples.json",
    "hypertension_klm":  BASE / "hypertension_triples.json"
}

sys.path.insert(0, str(BASE))
from meta_model import MetaModelAssembler, get_best_adapter, find_merged_entities


def _serialize_triple(i: int, t) -> dict:
    # Convert an AlignedTriple to a plain JSON-safe dict.
    return {
        "rank":       i + 1,
        "klm":        t.klm_source,
        "head":       t.head.raw_text,
        "relation":   t.relation,
        "tail":       t.tail.raw_text,
        "confidence": round(float(t.confidence), 4),
        "evidence":   t.evidence_level,
        "annotation": t.annotation or "",
    }


def run_query(query: str, top_n: int = 20) -> dict:
    
    t_start = time.time()

    print(f"Query             : {query!r}", file=sys.stderr)
    print(f"Loading KLMs...", file=sys.stderr)

    assembler   = MetaModelAssembler(KLM_PATHS)
    output      = assembler.query(query, top_n=top_n)
    pipeline_ms = (time.time() - t_start) * 1000

    merged = find_merged_entities(output.entity_graph)

    print(
        f"Pipeline          : {pipeline_ms:.0f}ms  "
        f"{len(output.ranked_triples)} triples ranked, "
        f"{len(merged)} cross-KLM entities, "
        f"{len(output.conflict_pairs)} conflicts",
        file=sys.stderr,
    )

    slm = get_best_adapter(verbose=False)  # always silent; we log to stderr ourselves

    print(f"SLM               : {slm.name}  generating...", file=sys.stderr)
    if hasattr(slm, "get_available_model"):
        print("Using Ollama model:", slm.get_available_model())

    t_slm    = time.time()
    answer   = slm.generate(output.structured_prompt)
    slm_ms   = (time.time() - t_slm) * 1000
    total_ms = (time.time() - t_start) * 1000

    print(f"Done              : {total_ms:.0f}ms total", file=sys.stderr)

    return {
        "query":  query,
        "answer": answer,
        "slm":    slm.name,
        "conflicts": [
            {
                "entity":        c["entity"],
                "severity":      c["severity"],
                "conflict_type": c["conflict_type"],
                "klm_a":         c["triple_a"]["klm"],
                "claim_a":       c["triple_a"]["tail"],
                "klm_b":         c["triple_b"]["klm"],
                "claim_b":       c["triple_b"]["tail"],
            }
            for c in output.conflict_pairs
        ],
        "cross_klm_entities": [
            {
                "cui":   e["cui"],
                "names": e["raw_texts"],
                "klms":  e["klm_sources"],
            }
            for e in merged
        ],
        "top_triples": [
            _serialize_triple(i, t)
            for i, t in enumerate(output.ranked_triples[:top_n])
        ],
        "pipeline_ms": round(pipeline_ms, 1),
        "slm_ms":      round(slm_ms, 1),
        "total_ms":    round(total_ms, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Loop KM Meta-Model  one-shot query"
    )
    parser.add_argument(
        "query",
        help='Clinical query e.g. "What treatment options are there for this patient?"',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the full result as JSON (default: plain text answer only)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="N",
        help="Number of ranked triples to include in the prompt (default: 20)",
    )
    args = parser.parse_args()

    try:
        result = run_query(args.query, top_n=args.top_n)
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["answer"])


if __name__ == "__main__":
    main()
