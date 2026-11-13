"""Read-only real-corpus comparison of legacy and experimental retrieval.

Run on a machine with FAISS, metadata, cached BGE-M3 and BM25 assets.
No LLM calls, production edits, service restarts or index rebuilding.

From an isolated checkout of this development branch on Oracle:
  /opt/omnissiah/app/.venv/bin/python tools/evaluate_retrieval_recovery.py \
    --runtime-root /opt/omnissiah/app --case horus
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

CASES = {
    "horus": "why did horus turn traitor",
    "loken-jubal": "loken saving the girl from jubal when possessed",
    "horus-invisibles": "horus kills the invisibles",
    "kakophoni": "kakaponi of the 3rd legion attacked the 5th legion using sonic weapons",
    "lorgar-magnus": "discussion of lorgar and magnus",
    "sea-of-souls": "lorgar speaking about the sea of souls like an ocean",
    "white-scars-daemon": "slaneshi demon facing the white scars as they race to terra entire fight",
    "sharrowkyn-lucius": "sharokin and lucius battle angel exterminatus",
}


def _load_from_file(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _summarize(hits):
    return [{
        "chunk_id": h.get("chunk_id"),
        "source": h.get("source", ""),
        "chapter": h.get("chapter", ""),
        "stitch_range": h.get("stitch_range", ""),
        "legacy_rerank_score": h.get("rerank_score"),
        "recovery_raw_rerank_score": h.get("recovery_raw_rerank_score"),
        "excerpt": h.get("text", "")[:700],
    } for h in hits]


def run_case(retriever, query, cfg, pool, top_k, window):
    before = retriever._search_base(
        query=query, top_k=top_k, candidate_pool=pool, stitching_window=window,
    )
    previous = cfg.get("adaptive_retrieval", False)
    cfg["adaptive_retrieval"] = True
    try:
        after = retriever.search(
            query=query, top_k=top_k, candidate_pool=pool, stitching_window=window,
        )
    finally:
        cfg["adaptive_retrieval"] = previous
    return {"query": query, "baseline": _summarize(before), "adaptive": _summarize(after)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--case", choices=[*CASES, "all"], default="horus")
    parser.add_argument("--out", type=Path, default=Path("/tmp/omnissiah-retrieval-ab.json"))
    parser.add_argument("--pool", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()
    stage_root = Path(__file__).resolve().parents[1]
    runtime_root = args.runtime_root.resolve()
    sys.path.insert(0, str(runtime_root))
    os.environ.setdefault("OMNISSIAH_ACTIVE_PROFILE", "dell_query")
    from Core.config_loader import retrieval_cfg
    _load_from_file("Core.retrieval_recovery", stage_root / "Core" / "retrieval_recovery.py")
    staged = _load_from_file("staged_retriever_ab", stage_root / "Core" / "retriever.py")
    # Standalone process: production service and all deployed source remain untouched.
    retriever = staged.OmnissiahRetriever()
    names = CASES.keys() if args.case == "all" else (args.case,)
    results = {
        name: run_case(retriever, CASES[name], retrieval_cfg, args.pool, args.top_k, args.window)
        for name in names
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, item in results.items():
        print(f"{name}: legacy={len(item['baseline'])} adaptive={len(item['adaptive'])}")
        print("   legacy:", [x["source"] for x in item["baseline"]])
        print("   adaptive:", [x["source"] for x in item["adaptive"]])
    print("Read-only report saved to", args.out)
    print("WARNING: counts and book titles are NOT evidence-accuracy passes.")


if __name__ == "__main__":
    main()
