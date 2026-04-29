"""
OmnissiahCore - build.py
FAISS index builder with sentence-aware chunking for Warhammer lore RAG.
Supports: PDF, TXT (CBR, EPUB, AZW3 stubs ready to wire up)
Device: auto (CPU fallback if CUDA not compatible)
"""

import os
# top of build.py, before imports
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import sys
import json
import re
import faiss
import numpy as np
import torch

from hashlib import md5
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── nltk sentence tokenizer ──────────────────────────────────────────────────
import nltk
try:

    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    print("[*] Downloading NLTK punkt tokenizer...")
    nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from Core.config_loader import embedding_cfg, chunking_cfg, paths
except ImportError:
    print("[ERROR] Core.config_loader not found. Run from project root.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_device(cfg_device: str) -> str:
    """
    Resolve the best available device.
    cfg_device: 'auto' | 'cuda' | 'cpu'
    Falls back to CPU if CUDA is present but incompatible (RTX 5050 etc.)
    """
    if cfg_device == "cpu":
        print("[*] Device forced to CPU by config.")
        return "cpu"

    if torch.cuda.is_available():
        # Quick smoke-test: allocate a tiny tensor to check actual compatibility
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + 1          # forces a kernel launch
            del t
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[*] CUDA OK — using GPU: {gpu_name}")
            return "cuda"
        except RuntimeError as e:
            print(f"[!] CUDA detected but not usable ({e}). Falling back to CPU.")
            return "cpu"
    else:
        print("[*] No CUDA device found. Using CPU.")
        return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# FILE EXTRACTORS  (add new formats here — wire them into EXTRACTORS below)
# ─────────────────────────────────────────────────────────────────────────────

def extract_txt(filepath: str) -> str | None:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip() or None
    except Exception as e:
        print(f"  [WARN] TXT read failed: {filepath} — {e}")
        return None


def extract_pdf(filepath: str) -> str | None:
    try:
        reader = PdfReader(filepath)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        full = "\n".join(pages).strip()
        return full if len(full) > 50 else None
    except Exception as e:
        print(f"  [WARN] PDF read failed: {filepath} — {e}")
        return None


# Stubs — implement when you add those libraries
def extract_epub(filepath: str) -> str | None:
    # pip install ebooklib beautifulsoup4
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        book = epub.read_epub(filepath)
        texts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            texts.append(soup.get_text(separator=" "))
        return "\n".join(texts).strip() or None
    except ImportError:
        print("  [SKIP] ebooklib not installed. Run: pip install ebooklib beautifulsoup4")
        return None
    except Exception as e:
        print(f"  [WARN] EPUB read failed: {filepath} — {e}")
        return None


def extract_cbr(filepath: str) -> str | None:
    # CBR/CBZ are image archives — OCR would be needed, skip for now
    print(f"  [SKIP] CBR/CBZ files are image archives and need OCR — skipping: {os.path.basename(filepath)}")
    return None


def extract_azw3(filepath: str) -> str | None:
    # pip install mobi   OR   convert with calibre's ebook-convert first
    print(f"  [SKIP] AZW3 support not yet implemented — skipping: {os.path.basename(filepath)}")
    return None


# ── Dispatcher: extension → extractor function ───────────────────────────────
EXTRACTORS = {
    ".txt":  extract_txt,
    ".pdf":  extract_pdf,
    ".epub": extract_epub,
    ".cbr":  extract_cbr,
    ".cbz":  extract_cbr,
    ".azw3": extract_azw3,
    ".mobi": extract_azw3,
}

SUPPORTED_EXTS = set(EXTRACTORS.keys())


def extract_file(filepath: str) -> str | None:
    ext = os.path.splitext(filepath)[1].lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        print(f"  [SKIP] Unsupported type '{ext}': {os.path.basename(filepath)}")
        return None
    return extractor(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE-AWARE CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace and remove junk characters."""
    text = re.sub(r"[ \t]+", " ", text)          # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 blank lines
    text = re.sub(r"[^\x00-\x7F]+", " ", text)   # strip non-ASCII (optional — remove if you have non-English)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~1.3 tokens per word (conservative for BPE models).
    Good enough for chunking without loading the actual tokenizer.
    """
    return int(len(text.split()) * 1.3)


def chunk_text(
    text: str,
    target_tokens: int = 400,
    overlap_sentences: int = 2,
) -> list[str]:
    """
    Sentence-aware chunking:
    - Splits on real sentence boundaries (NLTK)
    - Accumulates sentences until we hit target_tokens
    - Backs up overlap_sentences sentences before starting next chunk
    - Preserves paragraph structure hints by splitting on double-newlines first

    This keeps dialogue, lore paragraphs, and named entities intact.
    """
    # Split on paragraph breaks first to avoid merging unrelated sections
    paragraphs = re.split(r"\n{2,}", text)

    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        sents = sent_tokenize(para)
        sentences.extend(sents)

    if not sentences:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(sentences):
        current_sents: list[str] = []
        current_tokens = 0
        i = start

        while i < len(sentences):
            sent = sentences[i]
            sent_tokens = estimate_tokens(sent)

            # If a single sentence is already huge, include it alone
            if not current_sents or current_tokens + sent_tokens <= target_tokens:
                current_sents.append(sent)
                current_tokens += sent_tokens
                i += 1
            else:
                break  # chunk is full

        if current_sents:
            chunk = " ".join(current_sents).strip()
            if chunk:
                chunks.append(chunk)

        # Move start forward, but back-step by overlap_sentences for context continuity
        advance = max(1, len(current_sents) - overlap_sentences)
        start += advance

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# DB PATHS (derived from config paths + fixed filenames)
# ─────────────────────────────────────────────────────────────────────────────

def get_db_paths(db_dir: str) -> dict:
    return {
        "faiss":     os.path.join(db_dir, "index.faiss"),
        "metadata":  os.path.join(db_dir, "metadata.json"),
        "processed": os.path.join(db_dir, "processed_files.json"),
        "failed":    os.path.join(db_dir, "failed_files.json"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Device ───────────────────────────────────────────────────────────────
    cfg_device = embedding_cfg.get("device", "auto")
    device = get_device(cfg_device)

    # ── Batch size ────────────────────────────────────────────────────────────
    if device == "cuda":
        batch_size = embedding_cfg.get("batch_size_gpu", 32)
    else:
        batch_size = embedding_cfg.get("batch_size_cpu", 8)
    print(f"[*] Batch size: {batch_size}")

    # ── Load model ────────────────────────────────────────────────────────────
    model_name = embedding_cfg["model"]
    print(f"[*] Loading model: {model_name}  (this may take a minute on first run)...")
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = embedding_cfg.get("max_seq_length", 512)

    # ── DB paths ──────────────────────────────────────────────────────────────
    db_dir   = paths["db_dir"]
    src_dir  = paths["pdf_dir"]
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(paths.get("failed_dir", "Data/failed_pdfs"), exist_ok=True)

    db = get_db_paths(db_dir)

    # ── Load existing state ───────────────────────────────────────────────────
    index_exists = os.path.exists(db["faiss"]) and os.path.exists(db["metadata"])

    if index_exists:
        print("[*] Existing index found. Loading for incremental update...")
        index = faiss.read_index(db["faiss"])
        with open(db["metadata"], "r", encoding="utf-8") as f:
            metadata: list[dict] = json.load(f)
    else:
        print("[!] No existing index. Starting fresh.")
        index = None
        metadata = []

    processed_files: set[str] = set()
    if os.path.exists(db["processed"]) and index_exists:
        with open(db["processed"], "r") as f:
            processed_files = set(json.load(f))

    failed_files: set[str] = set()
    if os.path.exists(db["failed"]):
        with open(db["failed"], "r") as f:
            failed_files = set(json.load(f))

    # ── Scan for new files ────────────────────────────────────────────────────
    all_files = [
        f for f in os.listdir(src_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]
    to_process = [f for f in all_files if f not in processed_files]

    if not to_process:
        print(f"[OK] No new files. Index contains {len(metadata)} chunks from {len(processed_files)} files.")
        return

    # Checkpoint every N files — safe for long overnight runs
    CHECKPOINT_EVERY = 50

    print(f"[*] {len(to_process)} new file(s) to process  ({len(processed_files)} already indexed).")
    print(f"[*] Checkpoint save every {CHECKPOINT_EVERY} files — safe to Ctrl+C and resume anytime.\n")

    target_tokens = chunking_cfg.get("target_tokens", 400)
    overlap_sents = chunking_cfg.get("overlap_sentences", 2)
    normalize     = embedding_cfg.get("normalize", True)

    def save_checkpoint(index, metadata, processed_files, failed_files):
        """Persist everything to disk atomically."""
        faiss.write_index(index, db["faiss"])
        with open(db["metadata"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        with open(db["processed"], "w") as f:
            json.dump(sorted(processed_files), f, indent=2)
        with open(db["failed"], "w") as f:
            json.dump(sorted(failed_files), f, indent=2)

    # ── Process in checkpoint windows ─────────────────────────────────────────
    total_done  = 0
    grand_total = len(to_process)

    for window_start in range(0, grand_total, CHECKPOINT_EVERY):
        window = to_process[window_start : window_start + CHECKPOINT_EVERY]
        window_num = (window_start // CHECKPOINT_EVERY) + 1
        total_windows = (grand_total + CHECKPOINT_EVERY - 1) // CHECKPOINT_EVERY

        print(f"── Batch {window_num}/{total_windows}  "
              f"(files {window_start + 1}–{min(window_start + CHECKPOINT_EVERY, grand_total)} "
              f"of {grand_total}) ──")

        batch_chunks: list[str]  = []
        batch_meta:   list[dict] = []

        for filename in tqdm(window, desc="  Extracting"):
            filepath = os.path.join(src_dir, filename)
            raw = extract_file(filepath)

            if not raw:
                failed_files.add(filename)
                continue

            text   = clean_text(raw)
            chunks = chunk_text(text, target_tokens=target_tokens, overlap_sentences=overlap_sents)

            if not chunks:
                failed_files.add(filename)
                continue

            base_id = len(metadata) + len(batch_meta)
            for idx, chunk in enumerate(chunks):
                batch_chunks.append(chunk)
                batch_meta.append({
                    "chunk_id": base_id + idx,
                    "source":   filename,
                    "text":     chunk,
                })

            processed_files.add(filename)
            total_done += 1

        if not batch_chunks:
            print("  [!] No chunks from this batch — skipping embed.")
            # Still save failed list
            with open(db["failed"], "w") as f:
                json.dump(sorted(failed_files), f, indent=2)
            continue

        print(f"  [*] Embedding {len(batch_chunks)} chunks on {device.upper()}...")
        embeddings = model.encode(
            batch_chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype("float32")

        # Init FAISS on first successful batch
        if index is None:
            dim   = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            print(f"  [*] Created FAISS IndexFlatIP  (dim={dim})")

        index.add(embeddings)
        metadata.extend(batch_meta)

        print(f"  [*] Saving checkpoint...  "
              f"({total_done}/{grand_total} files | {len(metadata)} total chunks)")
        save_checkpoint(index, metadata, processed_files, failed_files)
        print(f"  [OK] Checkpoint saved.\n")

    # ── Final summary ──────────────────────────────────────────────────────────
    if index is not None:
        print(
            f"\n[SUCCESS] Index holds {len(metadata)} chunks "
            f"from {len(processed_files)} file(s).  "
            f"Failed: {len(failed_files)}."
        )
    else:
        print("[!] No usable text extracted from any file.")


if __name__ == "__main__":
    main()

