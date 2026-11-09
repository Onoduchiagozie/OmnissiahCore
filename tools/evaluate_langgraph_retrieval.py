"""Read-only old-vs-LangGraph selection benchmark on Oracle's existing assets.

NO generation calls, production code edits, service restarts, or indexing.
Requires the optional LangGraph dependency available to the isolated Python
test interpreter. It loads a SECOND FAISS/BM25/embedding/reranker process;
inspect available RAM before running it on the live Oracle VM.

Run from an isolated dev-branch checkout:
  python tools/evaluate_langgraph_retrieval.py \
    --runtime-root /opt/omnissiah/app --case horus

The production runtime root supplies ONLY configuration/data/model paths.
The staged experimental modules come from this checkout. Never overwrite
the deployed Core directory to run an experiment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

QUERIES = {
    "horus": "why did horus turn traitor",
    "loken-jubal": "loken saving the girl from jubal when possessed",
    "horus-invisibles": "horus kills the invisibles",
    "kakophoni": "kakaponi of the 3rd legion attacked the 5th legion using sonic weapons",
    "lorgar-magnus": "discussion of lorgar and magnus",
    "sea-of-souls": "lorgar speaking about the sea of souls like an ocean",
    "white-scars-daemon": "slaneshi demon facing the white scars as they race to terra entire fight",
    "sharrowkyn-lucius": "sharokin and lucius battle angel exterminatus",
}


def _load(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _hits(items):
    """Preserve useful source passages for manual judgement, no false pass label."""
    return [{
        "chunk_id": x.get("chunk_id"),
        "source": x.get("source"),
        "chapter": x.get("chapter"),
        "stitch_range": x.get("stitch_range"),
        "legacy_rerank_score": x.get("rerank_score"),
        "graph_raw_rerank_logit": x.get("raw_rerank_logit"),
        "text": x.get("text", "")[:2200],
    } for x in items]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--case", choices=[*QUERIES, "all"], default="horus")
    parser.add_argument("--out", type=Path, default=Path("/tmp/omnissiah-langgraph-ab.json"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()

    checkout = Path(__file__).resolve().parents[1]
    deployed = args.runtime_root.resolve()
    if not deployed.is_dir() or not (deployed / "config.json").is_file():
        parser.error("runtime root must contain the existing deployed config.json")
    os.environ.setdefault("OMNISSIAH_ACTIVE_PROFILE", "dell_query")
    sys.path.insert(0, str(deployed))

    # Runtime's Core package supplies config_loader (production data paths).
    # Extend its module search path with our isolated experimental source.
    import Core
    # Import production config BEFORE adding experimental modules to the package
    # path; otherwise the development checkout may select its own missing Db.
    from Core.config_loader import paths
    Core.__path__.insert(0, str(checkout / "Core"))
    cache = Path(paths["bm25_cache"])
    metadata = Path(paths["metadata"])
    if not cache.is_dir() or cache.stat().st_mtime < metadata.stat().st_mtime:
        parser.error("BM25 cache absent/stale: aborting to avoid a rebuild")
    from Core.graphs.retrieval_graph import RetrievalGraph

    experimental = _load("experimental_retriever", checkout / "Core" / "retriever.py")
    retriever = experimental.OmnissiahRetriever()
    graph = RetrievalGraph(retriever)
    cases = QUERIES if args.case == "all" else {args.case: QUERIES[args.case]}
    output = {}
    for name, question in cases.items():
        legacy = retriever._search_base(
            question, top_k=args.top_k, candidate_pool=args.pool,
            stitching_window=args.window,
        )
        graph_state = graph.run(
            question, top_k=args.top_k, candidate_pool=args.pool,
            stitching_window=args.window, max_passes=2,
        )
        output[name] = {
            "query": question,
            "baseline": _hits(legacy),
            "langgraph": _hits(graph_state.get("selected_evidence", [])),
            "graph_trace": graph_state.get("diagnostic", []),
            "graph_gaps": graph_state.get("evidence_gaps", []),
            "accuracy_status": "NOT_VERIFIED__MANUAL_EVIDENCE_REVIEW_REQUIRED",
        }
        print(f"{name}: baseline {len(legacy)} / graph "
              f"{len(graph_state.get('selected_evidence', []))} passages")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", args.out, "— assess passages, not source counts.")


if __name__ == "__main__":
    main()
