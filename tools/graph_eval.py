"""Baseline vs experimental graph retrieval audit; sends NO generation requests.

Does not invent answer-accuracy metrics. Source overlap is a descriptive measure,
not a measure of whether a source is genuinely relevant to the question.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

CASES = [
    "Why did Horus turn traitor?",
    "Ferrus Manus vs Fulgrim",
    "Who was Kharn's closest friend?",
    "Why did Fulgrim take up the Laer blade?",
    "What happened at Isstvan V?",
    "Describe the Anathame blade",
    "What happened at the battle of Prospero?",
]


def request_json(url: str, question: str, timeout: int = 180) -> dict:
    payload = json.dumps({"query": question, "top_k": 6, "candidate_pool": 20}).encode()
    request = Request(url, data=payload, method="POST",
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def compare(baseline: dict, experimental: dict, query: str) -> dict:
    baseline_sources = {str(s.get("source", "")) for s in
                        baseline.get("inspection", {}).get("stitched_hits", []) if s.get("source")}
    graph_sources = {str(s.get("source", "")) for s in experimental.get("sources", [])
                     if s.get("source")}
    union = baseline_sources | graph_sources
    return {"query": query, "graph_status": experimental.get("status"),
            "graph_attempts": experimental.get("attempts", 0),
            "baseline_source_count": len(baseline_sources),
            "graph_source_count": len(graph_sources),
            "source_overlap_jaccard": round(len(baseline_sources & graph_sources) / len(union), 4)
            if union else None,
            "baseline_sources": sorted(baseline_sources),
            "graph_sources": sorted(graph_sources),
            "graph_evidence_grade": experimental.get("evidence_grade"),
            "requires_human_source_review": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="graph-retrieval-audit.json")
    parser.add_argument("--questions", help="Optional JSON file containing an array of question strings")
    args = parser.parse_args()
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8")) if args.questions else CASES
    if not isinstance(questions, list) or not all(isinstance(q, str) and q.strip() for q in questions):
        parser.error("Questions must be a JSON array of non-blank strings")
    rows = []
    for question in questions:
        started = time.monotonic()
        try:
            baseline = request_json(args.api.rstrip("/") + "/query/inspect", question)
            graph = request_json(args.api.rstrip("/") + "/query/graph/inspect", question)
            row = compare(baseline, graph, question)
            row["duration_seconds"] = round(time.monotonic() - started, 2)
            print("OK", question, row["graph_status"], row["graph_attempts"])
        except Exception as exc:
            row = {"query": question, "error_type": type(exc).__name__,
                   "requires_human_source_review": True}
            print("ERROR", question, type(exc).__name__)
        rows.append(row)
    Path(args.output).write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")
    print(f"Audit saved to {args.output}")
    raise SystemExit(1 if any("error_type" in row for row in rows) else 0)


if __name__ == "__main__":
    main()
