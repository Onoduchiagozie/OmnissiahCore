"""Read-only diagnosis of explicit subject/event coverage in real Oracle metadata.

No application imports, FAISS index changes, Gemini calls or API restarts.
  cd /opt/omnissiah/app
  .venv/bin/python /tmp/oracle_anchor_audit.py

Loads the metadata JSON in a temporary second process, then exits.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    path = Path("/opt/omnissiah/app/Db/metadata.json")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    print("=== LOADING EXISTING METADATA (READ ONLY) ===", flush=True)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        metadata = json.load(stream)
    print(f"Loaded {len(metadata):,} chunks", flush=True)

    pair_counts = Counter()
    all_source_counts = Counter()
    subject_counts = Counter()
    event_counts = Counter()
    close_counts = Counter()
    samples = defaultdict(list)
    discard = re.compile(
        r"blacklibrary\.com|more horus heresy stories|all rights reserved",
        re.I,
    )
    a = re.compile(r"\bhorus\b")
    b = re.compile(r"\bdavin\b")
    for idx, chunk in enumerate(metadata):
        source = str(chunk.get("source", "?"))
        all_source_counts[source] += 1
        body = (chunk.get("text") or "").lower()
        if "horus" in body and a.search(body):
            subject_counts[source] += 1
        if "davin" in body and b.search(body):
            event_counts[source] += 1
        if "horus" not in body or "davin" not in body or discard.search(body):
            continue
        aa = [x.start() for x in a.finditer(body)]
        bb = [x.start() for x in b.finditer(body)]
        if not aa or not bb:
            continue
        nearest = min(abs(x-y) for x in aa for y in bb)
        if nearest > 850:
            continue
        pair_counts[source] += 1
        if nearest <= 350:
            close_counts[source] += 1
        if len(samples[source]) < 3:
            samples[source].append({
                "chunk_id": chunk.get("chunk_id", idx),
                "distance": nearest,
                "text": " ".join((chunk.get("text") or "")[:300].split()),
            })

    print("\n=== HORUS + DAVIN IN SAME CHUNK (<=850 characters) ===")
    for source, count in pair_counts.most_common(15):
        print(f"{count:>4} direct | {close_counts[source]:>4} close |"
              f" {source}")

    primary = [
        s for s in all_source_counts
        if "false gods" in s.casefold()
    ]
    print("\n=== FALSE GODS SOURCE AUDIT ===")
    for source in primary:
        print(f"{source}\n  total chunks={all_source_counts[source]}, "
              f"Horus mentions={subject_counts[source]}, "
              f"Davin mentions={event_counts[source]}, "
              f"co-occurrences={pair_counts[source]}, "
              f"close={close_counts[source]}")
        for sample in samples[source]:
            print("  SAMPLE:", sample)

    report = {
        "total_chunks": len(metadata),
        "ranked_source_cooccurrences": pair_counts.most_common(30),
        "primary_sources": [
            {
                "name": s, "total_chunks": all_source_counts[s],
                "horus_mentions": subject_counts[s],
                "davin_mentions": event_counts[s],
                "cooccurrences": pair_counts[s],
                "close": close_counts[s],
                "samples": samples[s],
            }
            for s in primary
        ],
    }
    out = Path("/tmp/oracle_anchor_audit.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nReport:", out)
    if not primary:
        print("WARNING: No 'False Gods' source found in metadata.")
        return 1
    if sum(pair_counts[s] for s in primary) == 0:
        print("WARNING: No same-chunk Horus/Davin passages in False Gods.")
        print("A neighboring-chunk or source-first recovery strategy is required.")
    else:
        print("CONFIRMED: False Gods has direct subject/event candidates.")
        print("If /query/evidence still drops it, investigate candidate/reranker stage.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
