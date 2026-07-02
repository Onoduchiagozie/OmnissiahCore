"""
preview_top_scenes.py — quality check before Phase 2 LLM weaving.

Reads clusters_raw.json and prints the top N scenes by cluster_score
across the entire corpus, with full stitched_text, so you can read
the actual prose that will be handed to the LLM and judge quality
before committing to a full weaving run.

Run from the same folder as clusters_raw.json:
    python preview_top_scenes.py           -> top 10
    python preview_top_scenes.py 20        -> top 20
    python preview_top_scenes.py 10 battle -> top 10 battle type only
    python preview_top_scenes.py 10 confrontation -> confrontation only
"""

import json
import sys
from pathlib import Path

PATH = Path(__file__).parent / "clusters_raw.json"

N           = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TYPE_FILTER = sys.argv[2].lower() if len(sys.argv) > 2 else None


def main():
    print(f"Loading {PATH} ...")
    with open(PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten all scenes across all books into one list
    all_scenes = []
    for source_raw, book in data.items():
        for scene in book['scenes']:
            all_scenes.append({
                'book_title':   book['title'],
                'source_raw':   source_raw,
                'chunk_count':  book['chunk_count'],
                'rank':         scene['rank'],
                'score':        scene['cluster_score'],
                'scene_type':   scene.get('scene_type', 'battle'),
                'span_start':   scene['chunk_id_start'],
                'span_end':     scene['chunk_id_end'],
                'scene_chunks': scene['chunk_count'],
                'stitched':     scene.get('stitched_text', ''),
            })

    # Apply type filter if requested
    if TYPE_FILTER:
        all_scenes = [s for s in all_scenes if s['scene_type'] == TYPE_FILTER]

    # Sort by score descending
    all_scenes.sort(key=lambda s: s['score'], reverse=True)
    top = all_scenes[:N]

    filter_label = f" ({TYPE_FILTER} only)" if TYPE_FILTER else ""
    print(f"\nTotal scenes in corpus: {len(all_scenes):,}{filter_label}")
    print(f"Showing top {N} by cluster_score\n")

    for i, scene in enumerate(top):
        print("=" * 70)
        print(f"#{i+1}  {scene['book_title']}")
        print(f"     type={scene['scene_type']}  score={scene['score']:.3f}  "
              f"span={scene['span_start']}-{scene['span_end']}  "
              f"chunks={scene['scene_chunks']}  (book rank {scene['rank']})")
        print("-" * 70)
        text = scene['stitched'].strip()
        if text:
            print(text[:2000])
            if len(text) > 2000:
                print(f"\n  ... [{len(text)-2000} more chars not shown] ...")
        else:
            print("  [no stitched_text — may be from an older clusters_raw.json]")
        print()

    # ── Quick quality signals ─────────────────────────────────────────────
    print("=" * 70)
    print("QUALITY SIGNALS across all shown scenes:")
    empty = sum(1 for s in top if not s['stitched'].strip())
    short = sum(1 for s in top if 0 < len(s['stitched']) < 200)
    good  = sum(1 for s in top if len(s['stitched']) >= 200)
    print(f"  Empty stitched_text   : {empty}")
    print(f"  Too short (<200 chars): {short}")
    print(f"  Good length (200+)    : {good}")
    print()
    print(f"  Score range across ALL {len(all_scenes):,} scenes:")
    scores = [s['score'] for s in all_scenes]
    print(f"    min={min(scores):.3f}  median={sorted(scores)[len(scores)//2]:.3f}  max={max(scores):.3f}")
    print()
    type_counts = {}
    for s in all_scenes:
        type_counts[s['scene_type']] = type_counts.get(s['scene_type'], 0) + 1
    print("  Scene type breakdown:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()