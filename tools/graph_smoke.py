"""Opt-in smoke test against the local FastAPI app; no third-party services."""
from __future__ import annotations
import argparse
import json
import urllib.error
import urllib.request

CASES = [
    "Why did Horus turn traitor?",
    "Ferrus Manus vs Fulgrim",
    "Who was Kharn's closest friend?",
    "Why did Fulgrim take up the Laer blade?",
    "What happened at Isstvan V?",
    "Describe the Anathame blade",
    "What is the battle of Prospero?",
]


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--no-generation", action="store_true", help="Only call retrieval-only /query/inspect")
    options = parser.parse_args()
    endpoint = "/query/inspect" if options.no_generation else "/query/graph"
    failed = 0
    for query in CASES:
        try:
            result = post_json(options.api.rstrip("/") + endpoint,
                               {"query": query, "top_k": 6, "candidate_pool": 20})
            if options.no_generation:
                inspection = result.get("inspection", {})
                hits = inspection.get("stitched_hits", [])
                print(f"INSPECT | hits={len(hits)} | {query}")
                for chunk in hits[:3]:
                    print("   SOURCE", chunk.get("source", "?"), "CHAPTER", chunk.get("chapter", "?"))
            else:
                print(f"{result.get('status')} | attempts={result.get('attempts')} | "
                      f"sources={result.get('chunks_used')} | {query}")
                print("   ANSWER", result.get("response", "")[:300])
                print("   SOURCES", [x.get("source") for x in result.get("sources", [])[:3]])
                if result.get("status") != "ok":
                    failed += 1
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            failed += 1
            print("FAILED", query, type(exc).__name__, str(exc)[:200])
    print(f"Finished: cases={len(CASES)}, non-ok={failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
