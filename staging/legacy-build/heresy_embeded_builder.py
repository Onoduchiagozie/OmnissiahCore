"""
heresy_embed_builder.py  —  GPU-Accelerated FAISS Embedding Builder
Horus Heresy Memorium

Reads .mobi and .pdf files, extracts chapters, embeds with mxbai-embed-large-v1,
builds FAISS index. Deduplicates by filename normalization. Checkpoints every
10 books so you can resume if it crashes.

GPU Requirements:
  - mxbai-embed-large-v1: ~6GB VRAM (1024 dims, batch processing)
  - FAISS index: ~2-4GB RAM depending on number of embeddings
  - Total: ~8GB VRAM + 4GB system RAM

Run:
    python heresy_embed_builder.py --mobi-dir ./Mobi --pdf-dir ./PDFs --manifest manifest.json

Outputs:
    heresy_faiss.index          (FAISS index)
    heresy_embeddings_meta.json (chapter metadata)
    heresy_embed_checkpoint.json (progress tracking)
"""

import json
import re
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed")
    print("Install with: pip install sentence-transformers --break-system-packages")
    exit(1)

try:
    import PyPDF2
except ImportError:
    print("WARNING: PyPDF2 not installed. PDF extraction will fail.")
    print("Install with: pip install PyPDF2 --break-system-packages")

try:
    from mobi import Mobi
except ImportError:
    print("WARNING: mobi not installed. MOBI extraction will fail.")
    print("Install with: pip install mobi --break-system-packages")

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
FAISS_INDEX_PATH = BASE_DIR / "heresy_faiss.index"
METADATA_PATH = BASE_DIR / "heresy_embeddings_meta.json"
CHECKPOINT_PATH = BASE_DIR / "heresy_embed_checkpoint.json"

# Embedding model
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_DIM = 1024
BATCH_SIZE = 8  # Conservative for 8GB VRAM

# Chapter extraction
MAX_CHAPTER_WORDS = 8000  # Max words per embedding (split longer chapters)
MIN_CHAPTER_WORDS = 100  # Skip chapters shorter than this

# Checkpointing
CHECKPOINT_INTERVAL = 10  # Save progress every N books

# Deduplication
NORMALIZE_PATTERN = re.compile(r'[\s\-_\.#]', re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
#  DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def normalize_filename(filename: str) -> str:
    """
    Normalize filename for deduplication.
    Removes numbers, extensions, spacing, case-insensitive comparison.

    Examples:
    - "Horus Rising - Graham McNeill.mobi"
    - "Horus_Rising_by_Graham_McNeill_2014.pdf"
    - "0.5 Horus Rising (Remastered).epub"

    All become: "horusrisinggrahammcneill"
    """
    # Remove extension
    name_without_ext = Path(filename).stem

    # Remove leading numbers (e.g. "0.5 ", "1. ")
    name_without_nums = re.sub(r'^[\d\.]+\s*', '', name_without_ext, flags=re.IGNORECASE)

    # Remove trailing numbers and edition marks (e.g. "(2014)", "#2")
    name_without_trailing = re.sub(r'\s*[\(\[][\d\w\s]*[\)\]]$', '', name_without_nums)
    name_without_trailing = re.sub(r'\s*#?\d+\s*$', '', name_without_trailing)

    # Remove all special characters and spaces
    normalized = NORMALIZE_PATTERN.sub('', name_without_trailing).lower()

    return normalized


def check_duplicates(mobi_files: list, pdf_files: list) -> dict:
    """
    Returns mapping of normalized_name -> list of file paths that match.
    Alerts if duplicates found.
    """
    all_files = mobi_files + pdf_files
    duplicates = defaultdict(list)

    for filepath in all_files:
        normalized = normalize_filename(filepath.name)
        duplicates[normalized].append(str(filepath))

    # Report duplicates
    found_dups = {k: v for k, v in duplicates.items() if len(v) > 1}

    if found_dups:
        print("\n⚠️  DUPLICATES FOUND (same book in multiple formats):")
        for normalized_name, paths in found_dups.items():
            print(f"\n   {normalized_name}:")
            for path in paths:
                print(f"     - {path}")
        print("\n   → Will process only the FIRST occurrence, skipping others.")

    return duplicates


def get_unique_files(mobi_files: list, pdf_files: list) -> list:
    """
    Returns list of files to process, skipping duplicates (keeps first occurrence).
    """
    duplicates = check_duplicates(mobi_files, pdf_files)

    seen_normalized = set()
    unique_files = []

    for filepath in mobi_files + pdf_files:
        normalized = normalize_filename(filepath.name)
        if normalized not in seen_normalized:
            unique_files.append(filepath)
            seen_normalized.add(normalized)

    return unique_files


# ─────────────────────────────────────────────────────────────────────────────
#  FILE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_mobi_chapters(mobi_path: Path) -> list[dict]:
    """
    Extract chapters from .mobi file.
    Returns: [{"chapter_number": 0, "chapter_title": "...", "text": "..."}]
    """
    try:
        tempdir = Path('/tmp/mobi_extract')
        tempdir.mkdir(exist_ok=True)

        mobi = Mobi(str(mobi_path))
        mobi.extract(str(tempdir))

        html_path = tempdir / 'index.html'
        if not html_path.exists():
            print(f"    ERROR: No index.html in {mobi_path.name}")
            return []

        html_content = html_path.read_text(encoding='utf-8', errors='ignore')

        # Strip HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Split by common chapter markers
        chapter_pattern = re.compile(
            r'(?:Chapter|CHAPTER|\d+\.|Part|PART|Section|SECTION)\s*[\d\w]*:?\s*(.+?)(?=(?:Chapter|CHAPTER|Part|PART|Section|SECTION)|$)',
            re.IGNORECASE | re.DOTALL
        )

        chapters = []
        for i, match in enumerate(chapter_pattern.finditer(text)):
            chapter_title = match.group(1).strip()[:100]  # First 100 chars as title
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
            # Fallback: treat entire book as one chapter if no markers found
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
    """
    Extract text from .pdf file.
    Returns: [{"chapter_number": 0, "chapter_title": "...", "text": "..."}]
    """
    try:
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""

            for page_num, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Split by chapter markers (same as MOBI)
        chapter_pattern = re.compile(
            r'(?:Chapter|CHAPTER|\d+\.|Part|PART|Section|SECTION)\s*[\d\w]*:?\s*(.+?)(?=(?:Chapter|CHAPTER|Part|PART|Section|SECTION)|$)',
            re.IGNORECASE | re.DOTALL
        )

        chapters = []
        for i, match in enumerate(chapter_pattern.finditer(text)):
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
    """
    Splits a chapter longer than max_words into multiple sub-chunks.
    """
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
#  CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    """Load checkpoint if it exists."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, 'r') as f:
            return json.load(f)
    return {
        "books_processed": 0,
        "total_embeddings": 0,
        "processed_files": [],
        "last_updated": None
    }


def save_checkpoint(checkpoint: dict):
    """Save progress checkpoint."""
    checkpoint["last_updated"] = datetime.utcnow().isoformat()
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING & FAISS
# ─────────────────────────────────────────────────────────────────────────────
def create_faiss_index(dim: int) -> faiss.Index:
    """Create empty FAISS index."""
    return faiss.IndexFlatIP(dim)


def add_embeddings_to_index(index: faiss.Index, embeddings: np.ndarray) -> faiss.Index:
    """Add embeddings to FAISS index."""
    embeddings = embeddings.astype(np.float32)
    index.add(embeddings)
    return index


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN BUILD PROCESS
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from Horus Heresy .mobi and .pdf files"
    )
    parser.add_argument('--mobi-dir', type=Path, required=True, help="Directory with .mobi files")
    parser.add_argument('--pdf-dir', type=Path, help="Directory with .pdf files (optional)")
    parser.add_argument('--manifest', type=Path, required=True, help="manifest.json with book metadata")

    args = parser.parse_args()

    print("=" * 70)
    print("  Horus Heresy  —  FAISS Embedding Builder")
    print("  Model: mixedbread-ai/mxbai-embed-large-v1")
    print("=" * 70)

    # ── Load manifest ────────────────────────────────────────────────────────
    print(f"\n[1/6]  Loading manifest ...")
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    manifest_lookup = {b['filename']: (b['book_order'], b['title']) for b in manifest['books']}
    print(f"       {len(manifest_lookup)} books in manifest")

    # ── Find files ───────────────────────────────────────────────────────────
    print(f"\n[2/6]  Scanning for .mobi and .pdf files ...")
    mobi_files = sorted(args.mobi_dir.glob('*.mobi')) if args.mobi_dir else []
    pdf_files = sorted(args.pdf_dir.glob('*.pdf')) if args.pdf_dir else []

    print(f"       Found {len(mobi_files)} .mobi files")
    print(f"       Found {len(pdf_files)} .pdf files")

    # ── Deduplication ────────────────────────────────────────────────────────
    print(f"\n[3/6]  Deduplicating ...")
    unique_files = get_unique_files(mobi_files, pdf_files)
    print(f"       Processing {len(unique_files)} unique files")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"\n[4/6]  Checking checkpoint ...")
    checkpoint = load_checkpoint()
    already_processed = set(checkpoint["processed_files"])
    print(f"       Already processed: {len(already_processed)} files")

    files_todo = [f for f in unique_files if f.name not in already_processed]
    print(f"       Files to process: {len(files_todo)}")

    if not files_todo:
        print("\n  All files already processed!")
        return

    # ── Load embedding model ─────────────────────────────────────────────────
    print(f"\n[5/6]  Loading embedding model (GPU) ...")
    print(f"       This may take 1-2 minutes on first run...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device='cuda')
    embedder.eval()
    print(f"       Model loaded. Dimension: {EMBEDDING_DIM}")

    # ── Load or create FAISS index ───────────────────────────────────────────
    if FAISS_INDEX_PATH.exists():
        print(f"\n[6/6]  Loading existing FAISS index ...")
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"       Index loaded. Current embeddings: {faiss_index.ntotal}")
        with open(METADATA_PATH, 'r') as f:
            all_metadata = json.load(f)
    else:
        print(f"\n[6/6]  Creating new FAISS index ...")
        faiss_index = create_faiss_index(EMBEDDING_DIM)
        all_metadata = []

    # ── Process files ────────────────────────────────────────────────────────
    print(f"\n  Starting embedding process (batch_size={BATCH_SIZE}, conservative VRAM usage)...\n")

    for file_idx, filepath in enumerate(files_todo):
        print(f"  [{file_idx + 1}/{len(files_todo)}]  {filepath.name} ...", end='')

        # Extract chapters
        if filepath.suffix.lower() == '.mobi':
            chapters = extract_mobi_chapters(filepath)
        elif filepath.suffix.lower() == '.pdf':
            chapters = extract_pdf_chapters(filepath)
        else:
            print(" SKIP (unknown format)")
            continue

        if not chapters:
            print(" SKIP (no chapters extracted)")
            continue

        # Split long chapters
        all_chapters = []
        for chapter in chapters:
            if chapter["word_count"] > MAX_CHAPTER_WORDS:
                all_chapters.extend(split_long_chapter(chapter, MAX_CHAPTER_WORDS))
            else:
                all_chapters.append(chapter)

        # Get metadata
        book_order, book_title = manifest_lookup.get(
            filepath.name,
            (999, f"Unknown ({filepath.name})")
        )

        # Embed chapters
        texts_to_embed = [ch["text"] for ch in all_chapters]

        try:
            embeddings = embedder.encode(
                texts_to_embed,
                batch_size=BATCH_SIZE,
                convert_to_tensor=False,
                show_progress_bar=False
            )

            # Normalize embeddings for IP distance
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

            # Add to FAISS
            faiss_index = add_embeddings_to_index(faiss_index, embeddings)

            # Track metadata
            for i, (chapter, embedding) in enumerate(zip(all_chapters, embeddings)):
                metadata_entry = {
                    "book_order": book_order,
                    "book_title": book_title,
                    "source_file": filepath.name,
                    "source_format": filepath.suffix.lower(),
                    "chapter_number": chapter["chapter_number"],
                    "chapter_title": chapter["chapter_title"],
                    "word_count": chapter["word_count"],
                    "is_split": chapter.get("is_split", False),
                    "embedding_index": faiss_index.ntotal - len(embeddings) + i
                }
                all_metadata.append(metadata_entry)

            print(f" ✓ {len(all_chapters)} chapters")

        except Exception as e:
            print(f" ERROR: {e}")
            continue

        # Checkpoint every N books
        if (file_idx + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint["books_processed"] = file_idx + 1
            checkpoint["total_embeddings"] = faiss_index.ntotal
            checkpoint["processed_files"].append(filepath.name)
            save_checkpoint(checkpoint)

            faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
            with open(METADATA_PATH, 'w') as f:
                json.dump(all_metadata, f, indent=2)

            print(f"\n  → Checkpoint: {faiss_index.ntotal} embeddings, {file_idx + 1} books processed")

    # ── Final save ───────────────────────────────────────────────────────────
    print(f"\n  Saving final index and metadata ...")

    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    checkpoint["books_processed"] = len(files_todo)
    checkpoint["total_embeddings"] = faiss_index.ntotal
    checkpoint["processed_files"] = [f.name for f in files_todo]
    save_checkpoint(checkpoint)

    print("\n" + "=" * 70)
    print("  Embedding Complete")
    print("=" * 70)
    print(f"  Total embeddings      : {faiss_index.ntotal:,}")
    print(f"  Total metadata entries: {len(all_metadata):,}")
    print(f"  FAISS index size      : {FAISS_INDEX_PATH.stat().st_size / 1_048_576:.1f} MB")
    print(f"  Metadata file size    : {METADATA_PATH.stat().st_size / 1_048_576:.1f} MB")
    print(f"\n  Next: python heresy_phase1_vector.py")
    print("=" * 70)


if __name__ == "__main__":
    main()