"""
backfill_chapter_text.py — One-time text backfill for the Horus Heresy embed pipeline

PROBLEM:
heresy_embed_builder.py embedded chapter text directly into vectors but
never saved the raw text itself into heresy_embeddings_meta.json — only
book_title, chapter_title, chapter_number, source_file, word_count, and
embedding_index were kept. Stage 2 (scene scoring/clustering) needs the
actual text to run BM25, do name/verb speech detection, and stitch final
scene prose.

WHAT THIS DOES:
Re-runs the EXACT SAME extraction + chapter-splitting logic used during
embedding (identical regexes, identical MAX_CHAPTER_WORDS split), against
your original .mobi/.pdf files. For each re-extracted chapter, it builds
a matching key of (source_file, chapter_number, chapter_title) — the same
fields already stored in heresy_embeddings_meta.json — and uses that key
to look up the correct embedding_index. This means alignment is done by
IDENTITY MATCHING, not by re-deriving position/order, so it stays correct
even if file-processing order differs slightly between the original
embedding run and this backfill run.

OUTPUT:
Db/heresy_chapter_text.db — a SQLite table:
    chapter_text(embedding_index INTEGER PRIMARY KEY, book_title TEXT,
                 chapter_title TEXT, chapter_number INTEGER,
                 source_file TEXT, word_count INTEGER, text TEXT)

Any embedding_index that fails to find a text match is logged to
backfill_unmatched.json for manual review — Stage 2 should skip those
rather than silently working with missing text.

Run:
    python backfill_chapter_text.py --mobi-dir ./Mobi --pdf-dir ./Pdfs --db-dir ./Db
"""

import json
import re
import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import PyPDF2
except ImportError:
    print("ERROR: PyPDF2 not installed. Run: pip install PyPDF2 --break-system-packages")
    exit(1)

try:
    import mobi
except ImportError:
    print("ERROR: mobi not installed. Run: pip install mobi --break-system-packages")
    exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  SAME CONSTANTS AS heresy_embed_builder.py — must match exactly for
#  chapter splitting to produce identical chapter_title values.
# ─────────────────────────────────────────────────────────────────────────────
MAX_CHAPTER_WORDS = 8000
MIN_CHAPTER_WORDS = 100

_CHAPTER_PATTERN = re.compile(
    r'(?:Chapter|CHAPTER|\d+\.|Part|PART|Section|SECTION)\s*[\d\w]*:?\s*(.+?)(?=(?:Chapter|CHAPTER|Part|PART|Section|SECTION)|$)',
    re.IGNORECASE | re.DOTALL
)


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION — copied verbatim from heresy_embed_builder.py so chapter
#  splitting/titling is byte-for-byte identical, which is what the
#  matching key depends on.
# ─────────────────────────────────────────────────────────────────────────────
def extract_mobi_chapters(mobi_path: Path) -> list[dict]:
    try:
        # Real API of the 'mobi' PyPI package (v0.4.1): mobi.extract(path)
        # returns (tempdir, filepath) — there is no Mobi class. filepath
        # usually points at the main extracted html/epub content file.
        tempdir_str, filepath_str = mobi.extract(str(mobi_path))
        tempdir = Path(tempdir_str)
        main_file = Path(filepath_str)

        html_content = None

        if main_file.exists() and main_file.suffix.lower() in ('.html', '.htm'):
            html_content = main_file.read_text(encoding='utf-8', errors='ignore')
        else:
            # Fallback: scan the extracted tempdir for the largest html file —
            # covers cases where filepath points at an .epub or non-html asset.
            html_candidates = list(tempdir.rglob('*.html')) + list(tempdir.rglob('*.htm'))
            if html_candidates:
                largest = max(html_candidates, key=lambda p: p.stat().st_size)
                html_content = largest.read_text(encoding='utf-8', errors='ignore')

        if html_content is None:
            print(f"    ERROR: No html content found for {mobi_path.name} "
                  f"(tempdir={tempdir}, main_file={main_file})")
            return []

        text = re.sub(r'<[^>]+>', '', html_content)
        text = re.sub(r'\s+', ' ', text).strip()

        chapters = []
        for i, match in enumerate(_CHAPTER_PATTERN.finditer(text)):
            chapter_title = match.group(1).strip()[:100]
            chapter_text = match.group(0).strip()
            word_count = len(chapter_text.split())
            if word_count >= MIN_CHAPTER_WORDS:
                chapters.append({
                    "chapter_number": i,
                    "chapter_title": chapter_title,
                    "text": chapter_text,
                    "word_count": word_count
                })

        if not chapters:
            word_count = len(text.split())
            if word_count >= MIN_CHAPTER_WORDS:
                chapters.append({
                    "chapter_number": 0,
                    "chapter_title": "Full Text",
                    "text": text,
                    "word_count": word_count
                })

        return chapters

    except Exception as e:
        print(f"    ERROR extracting {mobi_path.name}: {e}")
        return []


def extract_pdf_chapters(pdf_path: Path) -> list[dict]:
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += (page.extract_text() or "") + "\n"

        text = re.sub(r'\s+', ' ', text).strip()

        chapters = []
        for i, match in enumerate(_CHAPTER_PATTERN.finditer(text)):
            chapter_title = match.group(1).strip()[:100]
            chapter_text = match.group(0).strip()
            word_count = len(chapter_text.split())
            if word_count >= MIN_CHAPTER_WORDS:
                chapters.append({
                    "chapter_number": i,
                    "chapter_title": chapter_title,
                    "text": chapter_text,
                    "word_count": word_count
                })

        if not chapters:
            word_count = len(text.split())
            if word_count >= MIN_CHAPTER_WORDS:
                chapters.append({
                    "chapter_number": 0,
                    "chapter_title": "Full Text",
                    "text": text,
                    "word_count": word_count
                })

        return chapters

    except Exception as e:
        print(f"    ERROR extracting {pdf_path.name}: {e}")
        return []


def split_long_chapter(chapter: dict, max_words: int) -> list[dict]:
    text = chapter["text"]
    words = text.split()

    if len(words) <= max_words:
        return [chapter]

    chunks = []
    chunk_num = 0
    for i in range(0, len(words), max_words):
        chunk_words = words[i:i + max_words]
        chunk_text = ' '.join(chunk_words)
        chunks.append({
            "chapter_number": chapter["chapter_number"],
            "chapter_title": f"{chapter['chapter_title']} (Part {chunk_num + 1})",
            "text": chunk_text,
            "word_count": len(chunk_words),
            "is_split": True
        })
        chunk_num += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  MATCHING KEY — must match exactly how metadata entries are keyed
# ─────────────────────────────────────────────────────────────────────────────
def make_key(source_file: str, chapter_number: int, chapter_title: str) -> tuple:
    return (source_file, chapter_number, chapter_title)


# ─────────────────────────────────────────────────────────────────────────────
#  DB SETUP
# ─────────────────────────────────────────────────────────────────────────────
def init_text_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chapter_text (
            embedding_index INTEGER PRIMARY KEY,
            book_title      TEXT,
            chapter_title   TEXT,
            chapter_number  INTEGER,
            source_file     TEXT,
            word_count      INTEGER,
            text            TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_chaptertext_source ON chapter_text(source_file);
    """)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Backfill chapter text for existing Horus Heresy embeddings")
    parser.add_argument('--mobi-dir', type=Path, default=Path('./Mobi'))
    parser.add_argument('--pdf-dir', type=Path, default=Path('./Pdfs'))
    parser.add_argument('--db-dir', type=Path, default=Path('./Db'))
    args = parser.parse_args()

    meta_path = args.db_dir / "heresy_embeddings_meta.json"
    out_db_path = args.db_dir / "heresy_chapter_text.db"
    unmatched_path = Path("backfill_unmatched.json")

    print("=" * 70)
    print("  Chapter Text Backfill")
    print("=" * 70)

    # ── Load existing metadata, build key -> embedding_index lookup ────────
    print(f"\n[1/4]  Loading {meta_path} ...")
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"       {len(metadata):,} metadata entries loaded")

    key_to_index: dict[tuple, int] = {}
    dup_keys = 0
    for entry in metadata:
        key = make_key(entry["source_file"], entry["chapter_number"], entry["chapter_title"])
        if key in key_to_index:
            dup_keys += 1
            continue  # keep first; duplicates logged in summary
        key_to_index[key] = entry["embedding_index"]

    if dup_keys:
        print(f"       WARNING: {dup_keys} duplicate (source_file, chapter_number, "
              f"chapter_title) keys in metadata — only first occurrence kept per key")

    # ── Group metadata by source_file so we only re-extract each file once ──
    files_needed = sorted({entry["source_file"] for entry in metadata})
    print(f"       {len(files_needed)} unique source files referenced in metadata")

    # ── Locate actual files on disk ─────────────────────────────────────────
    print(f"\n[2/4]  Locating files on disk ...")
    mobi_files = {f.name: f for f in args.mobi_dir.glob('*.mobi')} if args.mobi_dir.exists() else {}
    pdf_files = {f.name: f for f in args.pdf_dir.glob('*.pdf')} if args.pdf_dir.exists() else {}
    all_files_on_disk = {**mobi_files, **pdf_files}

    missing_files = [f for f in files_needed if f not in all_files_on_disk]
    if missing_files:
        print(f"       WARNING: {len(missing_files)} source files from metadata "
              f"not found in --mobi-dir/--pdf-dir:")
        for mf in missing_files[:10]:
            print(f"         - {mf}")
        if len(missing_files) > 10:
            print(f"         ... and {len(missing_files) - 10} more")

    # ── Re-extract + split, matched by key, written straight to DB ──────────
    print(f"\n[3/4]  Re-extracting text and matching to embedding_index ...")
    conn = sqlite3.connect(str(out_db_path))
    init_text_db(conn)

    matched_count = 0
    unmatched_keys = []
    processed_files = 0

    for filename in files_needed:
        filepath = all_files_on_disk.get(filename)
        if filepath is None:
            continue  # already warned above

        if filepath.suffix.lower() == '.mobi':
            raw_chapters = extract_mobi_chapters(filepath)
        elif filepath.suffix.lower() == '.pdf':
            raw_chapters = extract_pdf_chapters(filepath)
        else:
            continue

        all_chapters = []
        for chapter in raw_chapters:
            if chapter["word_count"] > MAX_CHAPTER_WORDS:
                all_chapters.extend(split_long_chapter(chapter, MAX_CHAPTER_WORDS))
            else:
                all_chapters.append(chapter)

        book_title = None
        # book_title isn't derivable from the file alone — pull it from
        # whichever metadata entry matches this source_file (any chapter).
        for entry in metadata:
            if entry["source_file"] == filename:
                book_title = entry["book_title"]
                break

        for chapter in all_chapters:
            key = make_key(filename, chapter["chapter_number"], chapter["chapter_title"])
            embedding_index = key_to_index.get(key)

            if embedding_index is None:
                unmatched_keys.append({
                    "source_file": filename,
                    "chapter_number": chapter["chapter_number"],
                    "chapter_title": chapter["chapter_title"],
                })
                continue

            conn.execute("""
                INSERT OR REPLACE INTO chapter_text
                (embedding_index, book_title, chapter_title, chapter_number,
                 source_file, word_count, text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                embedding_index, book_title, chapter["chapter_title"],
                chapter["chapter_number"], filename,
                chapter["word_count"], chapter["text"]
            ))
            matched_count += 1

        processed_files += 1
        if processed_files % 10 == 0:
            conn.commit()
            print(f"       ... {processed_files}/{len(files_needed)} files processed, "
                  f"{matched_count:,} chapters matched so far")

    conn.commit()

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n[4/4]  Writing unmatched-key report ...")
    with open(unmatched_path, 'w', encoding='utf-8') as f:
        json.dump(unmatched_keys, f, indent=2)

    total_meta_entries = len(metadata)
    coverage = matched_count / total_meta_entries if total_meta_entries else 0

    print("\n" + "=" * 70)
    print("  Backfill Complete")
    print("=" * 70)
    print(f"  Metadata entries total    : {total_meta_entries:,}")
    print(f"  Text matched              : {matched_count:,}  ({coverage:.1%})")
    print(f"  Unmatched (no text found) : {len(unmatched_keys):,}")
    print(f"  Files missing from disk   : {len(missing_files)}")
    print(f"\n  Db/heresy_chapter_text.db  →  ready for Stage 2 lookups")
    if unmatched_keys:
        print(f"  backfill_unmatched.json    →  review before running Stage 2")
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    main()