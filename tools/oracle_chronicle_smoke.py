"""Live Oracle acceptance test for FINAL narrated evidence without LLM generation.

After deployment:
    .venv/bin/python tools/oracle_chronicle_smoke.py

This tests /query/evidence (including the cross-encoder), NOT /query/inspect.
Exit 1 means more retrieval work is needed, not that the API is down.
"""
import json
import sys
import urllib.request


QUESTION = (
    "Tell the complete story of Horus falling to Chaos, from his growing "
    "doubts and the events on Davin through his eventual betrayal."
)


def main():
    payload = {
        "query": QUESTION,
        "top_k": 8,
        "candidate_pool": 60,
        "stitching_window": 1,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8000/query/evidence",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.load(response)

    hits = data.get("hits", [])
    print("=== FINAL EVIDENCE / NO GEMINI CALL ===")
    print("Chronicle mode:", data.get("chronicle_mode"))
    print("Facets:", data.get("facet_queries"))
    print("Selected passages:", len(hits))
    for i, hit in enumerate(hits, 1):
        body = hit.get("text", "")
        lower = body.lower()
        print(f"\n[{i}] {hit.get('source')} | chunk {hit.get('chunk_id')}"
              f" | rerank {hit.get('rerank_score')}")
        print("Coverage:", {term: term in lower for term in
                            ("horus", "davin", "doubt", "betray")})
        print("Preview:", " ".join(body[:400].split()))

    selected_texts = [(h.get("text") or "").lower() for h in hits]
    has_davin_event = any("davin" in x and "horus" in x for x in selected_texts)
    has_primary_book = any("false gods" in (h.get("source") or "").lower() for h in hits)
    chronicle_enabled = data.get("chronicle_mode") is True
    if not (chronicle_enabled and has_davin_event and has_primary_book):
        print("\nFAIL: Missing Horus/Davin anchor or primary source in FINAL evidence.")
        print("This is a quality failure; do not claim the RAG problem is fixed.")
        return 1

    print("\nPASS: final evidence includes Horus on Davin and False Gods.")
    print("Manual review is still required for chronology, causality, and narration.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
