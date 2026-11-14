import json
from collections import defaultdict

with open("clusters_raw.json", encoding="utf-8") as f:
    data = json.load(f)

buckets = [(0,50),(50,200),(200,800),(800,3000),(3000,10000),(10000,25000),(25000,10**9)]

def bucket_for(n):
    for lo, hi in buckets:
        if lo <= n < hi:
            return f"{lo}-{hi}"
    return "unbucketed"

table = defaultdict(lambda: defaultdict(int))
for book in data.values():
    b = bucket_for(book["chunk_count"])
    table[b][len(book["scenes"])] += 1

for b in [f"{lo}-{hi}" for lo,hi in buckets]:
    row = table.get(b, {})
    total = sum(row.values())
    if total == 0:
        continue
    print(f"\n{b}  ({total} books)")
    for n in sorted(row):
        pct = row[n] / total * 100
        print(f"  {n} scenes: {row[n]:>4}  ({pct:4.1f}%)")