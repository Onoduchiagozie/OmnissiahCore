"""
recover_missing_books.py — Patch specific books into an existing embedding corpus

Some .mobi files silently failed extraction during the original
heresy_embed_builder.py run (the 'mobi' package API bug: `from mobi import
Mobi` never existed in this package version). Because dedup marks a book
as "seen" as soon as its filename is processed — regardless of whether
extraction actually succeeded — any book whose only copy was a failed
.mobi got dropped from the corpus entirely, with no PDF fallback.

This script extracts + embeds ONLY the specific files you list, using the
now-fixed extraction logic, and APPENDS them to your existing:
  - heresy_faiss.index          (new vectors added, old ones untouched)
  - heresy_embeddings_meta.json (new entries appended)
  - heresy_chapter_text.db      (new rows added, matching backfill's schema)

Existing embedding_index values are never touched or renumbered — new
entries simply continue from the current max index.

Run:
    python recover_missing_books.py --mobi-dir ./Mobi --pdf-dir ./Pdfs \
        --manifest manifest.json --db-dir . \
        --files "Garro - James Swallow.mobi" "Tallarn - John French.mobi"
"""

import json
import re
import sqlite3
import argparse
from pathlib import Path

import numpy as np
import faiss

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed")
    exit(1)

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


EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_DIM = 1024
BATCH_SIZE = 8
MAX_CHAPTER_WORDS = 8000
MIN_CHAPTER_WORDS = 100

_CHAPTER_PATTERN = re.compile(
    r'(?:Chapter|CHAPTER|\d+\.|Part|PART|Section|SECTION)\s*[\d\w]*:?\s*(.+?)(?=(?:Chapter|CHAPTER|Part|PART|Section|SECTION)|$)',
    re.IGNORECASE | re.DOTALL
)


# ─────────────────────────────────────────────────────────────────────────────
#  EXTRACTION — same fixed logic as backfill_chapter_text.py
# ─────────────────────────────────────────────────────────────────────────────
def extract_mobi_chapters(mobi_path: Path) -> list[dict]:
    try:
        tempdir_str, filepath_str = mobi.extract(str(mobi_path))
        tempdir = Path(tempdir_str)
        main_file = Path(filepath_str)

        html_content = None
        if main_file.exists() and main_file.suffix.lower() in ('.html', '.htm'):
            html_content = main_file.read_text(encoding='utf-8', errors='ignore')
        else:
            html_candidates = list(tempdir.rglob('*.html')) + list(tempdir.rglob('*.htm'))
            if html_candidates:
                largest = max(html_candidates, key=lambda p: p.stat().st_size)
                html_content = largest.read_text(encoding='utf-8', errors='ignore')

        if html_content is None:
            print(f"    ERROR: No html content found for {mobi_path.name}")
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
                    "chapter_number": i, "chapter_title": chapter_title,
                    "text": chapter_text, "word_count": word_count
                })

        if not chapters:
            word_count = len(text.split())
            if word_count >= MIN_CHAPTER_WORDS:
                chapters.append({
                    "chapter_number": 0, "chapter_title": "Full Text",
                    "text": text, "word_count": word_count
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
                    "chapter_number": i, "chapter_title": chapter_title,
                    "text": chapter_text, "word_count": word_count
                })

        if not chapters:
            word_count = len(text.split())
            if word_count >= MIN_CHAPTER_WORDS:
                chapters.append({
                    "chapter_number": 0, "chapter_title": "Full Text",
                    "text": text, "word_count": word_count
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
        chunks.append({
            "chapter_number": chapter["chapter_number"],
            "chapter_title": f"{chapter['chapter_title']} (Part {chunk_num + 1})",
            "text": ' '.join(chunk_words),
            "word_count": len(chunk_words),
            "is_split": True
        })
        chunk_num += 1
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Recover specific missing books into existing embedding corpus")
    parser.add_argument('--mobi-dir', type=Path, default=Path('./Mobi'))
    parser.add_argument('--pdf-dir', type=Path, default=Path('./Pdfs'))
    parser.add_argument('--manifest', type=Path, default=Path('manifest.json'))
    parser.add_argument('--db-dir', type=Path, default=Path('.'))
    parser.add_argument('--files', nargs='+', required=True,
                         help="Exact filenames (as they appear in Mobi/Pdfs folders) to recover")
    args = parser.parse_args()

    faiss_path = args.db_dir / "heresy_faiss.index"
    meta_path = args.db_dir / "heresy_embeddings_meta.json"
    text_db_path = args.db_dir / "heresy_chapter_text.db"

    print("=" * 70)
    print("  Recovering Missing Books")
    print("=" * 70)

    # ── Load manifest for book_order/title lookup ──────────────────────────
    manifest_lookup = {}
    if args.manifest.exists():
        with open(args.manifest, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        manifest_lookup = {b['filename']: (b['book_order'], b['title']) for b in manifest['books']}

    # ── Load existing FAISS index + metadata ────────────────────────────────
    print(f"\n[1/4]  Loading existing index + metadata ...")
    faiss_index = faiss.read_index(str(faiss_path))
    with open(meta_path, 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)
    print(f"       Current embeddings: {faiss_index.ntotal:,}")

    # ── Load embedder ────────────────────────────────────────────────────────
    print(f"\n[2/4]  Loading embedding model (GPU) ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device='cuda')

    # ── Open text DB ─────────────────────────────────────────────────────────
    conn = sqlite3.connect(str(text_db_path))
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
    """)

    # ── Process each requested file ─────────────────────────────────────────
    print(f"\n[3/4]  Extracting + embedding {len(args.files)} file(s) ...")

    for filename in args.files:
        filepath = None
        for candidate_dir in (args.mobi_dir, args.pdf_dir):
            candidate = candidate_dir / filename
            if candidate.exists():
                filepath = candidate
                break

        if filepath is None:
            print(f"  SKIP — file not found in --mobi-dir or --pdf-dir: {filename}")
            continue

        print(f"  Processing {filename} ...")

        if filepath.suffix.lower() == '.mobi':
            raw_chapters = extract_mobi_chapters(filepath)
        elif filepath.suffix.lower() == '.pdf':
            raw_chapters = extract_pdf_chapters(filepath)
        else:
            print(f"    SKIP (unknown format)")
            continue

        if not raw_chapters:
            print(f"    ERROR — extraction still produced 0 chapters. Investigate manually.")
            continue

        all_chapters = []
        for chapter in raw_chapters:
            if chapter["word_count"] > MAX_CHAPTER_WORDS:
                all_chapters.extend(split_long_chapter(chapter, MAX_CHAPTER_WORDS))
            else:
                all_chapters.append(chapter)

        book_order, book_title = manifest_lookup.get(filename, (999, filepath.stem))

        texts_to_embed = [ch["text"] for ch in all_chapters]
        embeddings = embedder.encode(
            texts_to_embed, batch_size=BATCH_SIZE,
            convert_to_tensor=False, show_progress_bar=False
        )
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        embeddings = embeddings.astype(np.float32)

        faiss_index.add(embeddings)

        for i, chapter in enumerate(all_chapters):
            embedding_index = faiss_index.ntotal - len(embeddings) + i

            all_metadata.append({
                "book_order": book_order,
                "book_title": book_title,
                "source_file": filename,
                "source_format": filepath.suffix.lower(),
                "chapter_number": chapter["chapter_number"],
                "chapter_title": chapter["chapter_title"],
                "word_count": chapter["word_count"],
                "is_split": chapter.get("is_split", False),
                "embedding_index": embedding_index
            })

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

        print(f"    ✓ {len(all_chapters)} chapters embedded + text saved")

    conn.commit()
    conn.close()

    # ── Save updated index + metadata ───────────────────────────────────────
    print(f"\n[4/4]  Saving updated index + metadata ...")
    faiss.write_index(faiss_index, str(faiss_path))
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("  Recovery Complete")
    print("=" * 70)
    print(f"  Total embeddings now: {faiss_index.ntotal:,}")
    print(f"  Total metadata entries: {len(all_metadata):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()