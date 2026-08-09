"""
generate_manifest.py — Auto-generate manifest.json from your Mobi/ and Pdfs/ folders

Scans both folders, guesses a clean book title from each filename, deduplicates
.mobi/.pdf pairs of the same book, and writes out manifest.json in the format
heresy_embed_builder.py expects.

You should review the generated manifest.json afterwards and fix any titles
or reorder book_order numbers (e.g. to match Horus Heresy reading order).

Run from your OmnissiahCore folder:
    python generate_manifest.py --mobi-dir ./Mobi --pdf-dir ./Pdfs --output manifest.json
"""

import json
import re
import argparse
from pathlib import Path

NORMALIZE_PATTERN = re.compile(r'[\s\-_\.#]', re.IGNORECASE)


def normalize_filename(filename: str) -> str:
    """Same normalization logic as heresy_embed_builder.py, for consistent dedup."""
    name_without_ext = Path(filename).stem
    name_without_nums = re.sub(r'^[\d\.]+\s*', '', name_without_ext, flags=re.IGNORECASE)
    name_without_trailing = re.sub(r'\s*[\(\[][\d\w\s]*[\)\]]$', '', name_without_nums)
    name_without_trailing = re.sub(r'\s*#?\d+\s*$', '', name_without_trailing)
    normalized = NORMALIZE_PATTERN.sub('', name_without_trailing).lower()
    return normalized


def guess_title(filename: str) -> str:
    """
    Turn a messy filename into a readable title.
    "Horus_Rising_by_Graham_McNeill_2014.pdf" -> "Horus Rising"
    "0.5 Horus Rising (Remastered).mobi"      -> "Horus Rising"
    """
    stem = Path(filename).stem

    # Drop leading numbers like "0.5 " or "12. "
    stem = re.sub(r'^[\d\.]+\s*', '', stem)

    # Replace underscores/dashes with spaces
    stem = re.sub(r'[_\-]+', ' ', stem)

    # Cut off " by Author Name" if present
    stem = re.sub(r'\s+by\s+.+$', '', stem, flags=re.IGNORECASE)

    # Drop trailing parenthetical/bracketed junk, e.g. "(Remastered)", "[2014]"
    stem = re.sub(r'\s*[\(\[].*?[\)\]]\s*$', '', stem)

    # Drop trailing standalone year or number
    stem = re.sub(r'\s+\d{2,4}\s*$', '', stem)

    return stem.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate manifest.json from Mobi/ and Pdfs/ folders")
    parser.add_argument('--mobi-dir', type=Path, default=Path('./Mobi'))
    parser.add_argument('--pdf-dir', type=Path, default=Path('./Pdfs'))
    parser.add_argument('--output', type=Path, default=Path('manifest.json'))
    args = parser.parse_args()

    mobi_files = sorted(args.mobi_dir.glob('*.mobi')) if args.mobi_dir.exists() else []
    pdf_files = sorted(args.pdf_dir.glob('*.pdf')) if args.pdf_dir.exists() else []

    print(f"Found {len(mobi_files)} .mobi files, {len(pdf_files)} .pdf files")

    all_files = mobi_files + pdf_files
    seen_normalized = {}
    books = []
    order = 1

    for filepath in all_files:
        norm = normalize_filename(filepath.name)

        if norm in seen_normalized:
            print(f"  SKIP (duplicate of '{seen_normalized[norm]}'): {filepath.name}")
            continue

        seen_normalized[norm] = filepath.name
        title = guess_title(filepath.name)

        books.append({
            "filename": filepath.name,
            "book_order": order,
            "title": title
        })
        order += 1

    manifest = {"books": books}

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(books)} unique book entries to {args.output}")
    print("\n⚠️  IMPORTANT: Open manifest.json and check:")
    print("   1. Titles look right (some filenames are messy and may need manual fixes)")
    print("   2. book_order matches your preferred reading order (currently just")
    print("      file-scan order, not Horus Heresy chronology)")


if __name__ == "__main__":
    main()