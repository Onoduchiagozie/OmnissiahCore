"""Read-only post-deploy retrieval smoke test for Oracle. Does not call the LLM.

Usage:
    python tools/oracle_horus_smoke.py --base-url http://127.0.0.1:8000
"""
import argparse
import json
import sys
import urllib.request


def post_json(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    endpoint = args.base_url.rstrip("/") + "/query/inspect"
    cases = [
        {
            "query": "Why did Horus turn traitor?",
            "top_k": 6,
            "candidate_pool": 30,
            "stitching_window": 1,
        },
        {
            "query": "What caused Horus to reject the Emperor after being wounded on Davin?",
            "book_filter": "False Gods",
            "top_k": 5,
            "candidate_pool": 50,
            "stitching_window": 1,
        },
    ]
    failures = []
    for case in cases:
        result = post_json(endpoint, case)
        data = result.get("inspection", {})
        faiss = data.get("faiss_hits", [])
        bm25 = data.get("bm25_hits", [])
        hits = data.get("stitched_hits", [])
        size = sum(len(item.get("text", "")) for item in hits)
        sources = sorted(set(item.get("source", "?") for item in hits))
        print(json.dumps({
            "query": case["query"],
            "causal_mode": data.get("causal_mode"),
            "faiss_count": len(faiss),
            "bm25_count": len(bm25),
            "stitched_count": len(hits),
            "stitched_chars": size,
            "sources": sources,
        }, indent=2))
        if not faiss or not bm25:
            failures.append("FAISS/BM25 returned no diagnostic hits")
        if not hits:
            failures.append("No stitched evidence")
        if size > 18000:
            failures.append("Causal context exceeds 18000-character budget")
        if case.get("book_filter") and not all("false gods" in s.lower() for s in sources):
            failures.append("Book filter was not respected")
    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("PASS: retrieval health checks. Human review of causal accuracy is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
