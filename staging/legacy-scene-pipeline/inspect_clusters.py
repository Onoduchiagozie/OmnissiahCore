"""
inspect_clusters.py — quick sanity check on clusters_raw.json

Prints a handful of sample books/scenes so you can eyeball whether
gap-fill/dedup/trim produced clean stitched_text, and specifically
searches for Vengeful Spirit / Molech to see if the cave confrontation
scene got picked out as its own distinct cluster.

Run from the same folder as clusters_raw.json:
    python inspect_clusters.py
"""

import json
from pathlib import Path

PATH = Path(__file__).parent / "clusters_raw.json"


def main():
    print(f"Loading {PATH} ...")
    with open(PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total books in output: {len(data):,}\n")

    # ── Sample: first 3 books, all scenes ──────────────────────────────
    print("=" * 80)
    print("SAMPLE BOOKS (first 3)")
    print("=" * 80)
    for i, (source_raw, book) in enumerate(data.items()):
        if i >= 3:
            break
        print(f"\nBOOK: {book['title']}")
        print(f"  source_raw: {source_raw}")
        print(f"  chunk_count: {book['chunk_count']}  scenes: {len(book['scenes'])}")
        for scene in book["scenes"]:
            print(f"  {'-'*70}")
            print(f"  rank {scene['rank']}  score={scene['cluster_score']:.3f}  "
                  f"span={scene['chunk_id_start']}-{scene['chunk_id_end']}  "
                  f"chunks={scene['chunk_count']}")
            preview = scene["stitched_text"][:500].replace("\n", " ")
            print(f"  stitched_text: {preview}...")

    # ── Search for Molech / Vengeful Spirit specifically ───────────────
    print("\n" + "=" * 80)
    print("SEARCHING FOR MOLECH / VENGEFUL SPIRIT")
    print("=" * 80)

    found_any = False
    for source_raw, book in data.items():
        title_l = book["title"].lower()
        source_l = source_raw.lower()
        if "molech" in title_l or "molech" in source_l \
           or "vengeful spirit" in title_l or "vengeful spirit" in source_l:
            found_any = True
            print(f"\nMATCH: {book['title']}  ({source_raw})")
            print(f"  chunk_count: {book['chunk_count']}  scenes: {len(book['scenes'])}")
            for scene in book["scenes"]:
                print(f"  {'-'*70}")
                print(f"  rank {scene['rank']}  score={scene['cluster_score']:.3f}  "
                      f"span={scene['chunk_id_start']}-{scene['chunk_id_end']}")
                preview = scene["stitched_text"][:600].replace("\n", " ")
                print(f"  stitched_text: {preview}...")

    if not found_any:
        print("\nNo book with 'molech' or 'vengeful spirit' in title/source found.")
        print("Listing all book titles containing 'horus' in case it's filed differently:")
        for source_raw, book in data.items():
            if "horus" in book["title"].lower() or "horus" in source_raw.lower():
                print(f"  - {book['title']}  ({source_raw})")

    # ── Scene-count distribution across the whole corpus ────────────────
    print("\n" + "=" * 80)
    print("SCENE-COUNT DISTRIBUTION (sanity check for collapse)")
    print("=" * 80)
    from collections import Counter
    counts = Counter(len(book["scenes"]) for book in data.values())
    for n in sorted(counts):
        bar = "#" * min(60, counts[n] // 5 or 1)
        print(f"  {n} scene(s): {counts[n]:>5,} books  {bar}")

    total_books = len(data)
    one_scene = counts.get(1, 0)
    print(f"\n  Books with exactly 1 scene: {one_scene:,} / {total_books:,} "
          f"({one_scene/total_books:.1%})")
    print(f"  Books with 3+ scenes:       "
          f"{sum(v for k,v in counts.items() if k>=3):,} / {total_books:,} "
          f"({sum(v for k,v in counts.items() if k>=3)/total_books:.1%})")

    print("\nDone.")


if __name__ == "__main__":
    main()