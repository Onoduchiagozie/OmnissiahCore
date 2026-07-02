import json

with open("clusters_raw.json", encoding="utf-8") as f:
    data = json.load(f)

# Grab Vengeful Spirit specifically since we already know it's interesting
for source_raw, book in data.items():
    if "Vengeful Spirit" in book.get("title", ""):
        with open("one_book_sample.json", "w", encoding="utf-8") as out:
            json.dump(book, out, ensure_ascii=False, indent=2)
        print(f"Wrote one_book_sample.json — {len(json.dumps(book))} chars")
        break