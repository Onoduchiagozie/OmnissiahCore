"""Read-only 15-topic live retrieval quality audit: NO Gemini calls.

Collects both source titles and actual passage-anchor coverage. This is
a recall proxy, not a factuality or chronological-coherence certificate.
"""
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://emerald-nativity-guacamole.ngrok-free.dev"
CASES = [
    ("Prospero duel", "Reconstruct the duel between Leman Russ and Magnus on Prospero.", ["russ", "magnus"]),
    ("Isstvan duel", "Narrate Fulgrim's duel with Ferrus Manus on Isstvan V.", ["fulgrim", "ferrus"]),
    ("Calth", "Explain the Ultramarines' battle with the Word Bearers at Calth.", ["calth", "ultramarines"]),
    ("Angron", "How did Angron become a daemon primarch on Nuceria?", ["angron", "nuceria"]),
    ("Yarrick", "Why did Ghazghkull respect Commissar Yarrick?", ["ghazghkull", "yarrick"]),
    ("Baal", "What happened when Tyranids invaded Baal?", ["baal", "tyranid"]),
    ("Iyanden", "Describe the Tyranid invasion of Iyanden.", ["iyanden", "tyranid"]),
    ("Rubric", "Explain why Ahriman performed the Rubric of the Thousand Sons.", ["ahriman", "rubric"]),
    ("Necrons", "Explain the awakening of the Necron tomb worlds.", ["necron", "tomb"]),
    ("Raven Guard", "Reconstruct the Raven Guard escape from Isstvan V.", ["raven guard", "isstvan"]),
    ("Omnissiah", "Explain the Mechanicus teachings about the Omnissiah.", ["mechanicus", "omnissiah"]),
    ("Night Lords", "Describe the Night Lords use of terror.", ["night lords", "terror"]),
    ("Eisenhorn", "Explain Eisenhorn's dealings with daemonhosts.", ["eisenhorn", "daemonhost"]),
    ("Helsreach", "Tell the story of Black Templars at Helsreach.", ["hel sreach", "black templars"]),
    ("Laer blade", "Explain the Laer blade and Fulgrim's corruption.", ["laer", "fulgrim"]),
]


def post(route, query):
    req = urllib.request.Request(
        BASE + route,
        method="POST",
        data=json.dumps({
            "query": query, "top_k": 8,
            "candidate_pool": 35, "stitching_window": 1,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "ngrok-skip-browser-warning": "true"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def audit():
    results = []
    for name, query, anchors in CASES:
        data = post("/query/inspect", query)
        hits = data.get("inspection", {}).get("stitched_hits", [])
        passages = []
        for hit in hits:
            text = (hit.get("text") or "").casefold()
            matches = {
                a: (a in text if a != "hel sreach" else "helsreach" in text)
                for a in anchors
            }
            passages.append({
                "source": hit.get("source"),
                "chunk_id": hit.get("chunk_id"),
                "matched_anchors": matches,
                "all_anchor_matches": all(matches.values()),
                "preview": " ".join((hit.get("text") or "")[:155].split()),
            })
        anchored = sum(row["all_anchor_matches"] for row in passages)
        result = {
            "topic": name, "query": query, "anchors": anchors,
            "retrieved": len(hits), "anchored_passages": anchored,
            "source_names": list(dict.fromkeys(p["source"] for p in passages)),
            "passages": passages,
        }
        results.append(result)
        state = "ANCHORED" if anchored else "GAP"
        print(f"{state:8} | {name:16} | {anchored}/{len(hits)} passages matching all anchors")
        print("         ", result["source_names"][:4], flush=True)

    covered = sum(x["anchored_passages"] > 0 for x in results)
    out = Path("/tmp/nonhorus_retrieval_audit.json")
    out.write_text(json.dumps({
        "tested": len(results), "anchor_covered_topics": covered, "cases": results,
        "note": "Read-only /query/inspect pre-final-rerank; lexical co-occurrence is only a weak relevance proxy.",
    }, indent=2), encoding="utf-8")
    print(f"\nSUBJECT+EVENT COVERAGE: {covered}/{len(results)} queries")
    print("This result is PRE-RERANK, not final evidence and not answer accuracy.")
    print("Report:", out)
    return 0 if len(results) == len(CASES) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(audit())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
