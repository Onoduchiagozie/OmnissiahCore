"""
OmnissiahCore - build.py  v2.0
FAISS index builder — handles ALL file types including:
  - PDF (text-based + scanned/OCR auto-detect)
  - AZW3 (via Calibre ebook-convert — auto-detected)
  - EPUB (via ebooklib)
  - CBR/CBZ (via OCR — pytesseract + rarfile)
  - TXT

Retries previously failed files automatically.
GPU via ONNX if available, else CPU fallback.
Checkpoints every 50 files — safe to interrupt and resume.
Full failure_report.json written explaining every failure.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys
import json
import re
import subprocess
import tempfile
import shutil

import faiss
import numpy as np
import torch
from tqdm import tqdm
from pypdf import PdfReader

# ── NLTK ─────────────────────────────────────────────────────────────────────
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
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def get_device(cfg_device: str) -> str:
    if cfg_device == "cpu":
        print("[*] Device forced to CPU by config.")
        return "cpu"
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + 1
            del t
            print(f"[*] CUDA OK — using GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        except RuntimeError as e:
            print(f"[!] CUDA not usable ({e}). Falling back to CPU.")
            return "cpu"
    print("[*] No CUDA device. Using CPU.")
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

failure_reasons: dict[str, str] = {}


def log_failure(filename: str, reason: str):
    failure_reasons[filename] = reason
    print(f"  [FAIL] {os.path.basename(filename)}")
    print(f"         Reason: {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTOR — with scanned PDF detection + OCR fallback
# ─────────────────────────────────────────────────────────────────────────────

def is_scanned_pdf(reader: PdfReader, sample_pages: int = 3) -> bool:
    total = ""
    for page in reader.pages[:sample_pages]:
        t = page.extract_text()
        if t:
            total += t
    return len(total.strip()) < 50


def extract_pdf_ocr(filepath: str) -> str | None:
    try:
        from pdf2image import convert_from_path
        import pytesseract

        for p in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break

        print(f"  [OCR] Scanning: {os.path.basename(filepath)}")
        pages = convert_from_path(filepath, dpi=200)
        texts = [pytesseract.image_to_string(p, lang="eng") for p in pages]
        full = "\n".join(t for t in texts if t.strip())
        return full if len(full) > 50 else None

    except ImportError:
        return None
    except Exception as e:
        log_failure(filepath, f"OCR runtime error: {e}")
        return None


def extract_pdf(filepath: str) -> str | None:
    try:
        reader = PdfReader(filepath)

        if is_scanned_pdf(reader):
            print(f"  [SCAN] Scanned PDF detected: {os.path.basename(filepath)}")
            result = extract_pdf_ocr(filepath)
            if result:
                return result
            try:
                import pytesseract, pdf2image  # noqa
                log_failure(filepath, "Scanned PDF — OCR returned no text (image-only or handwritten?)")
            except ImportError:
                log_failure(
                    filepath,
                    "Scanned PDF — OCR libraries not installed. Fix:\n"
                    "  1. pip install pdf2image pytesseract\n"
                    "  2. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "  3. Install Poppler: https://github.com/oschwartz10612/poppler-windows/releases"
                )
            return None

        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        full = "\n".join(pages).strip()

        if len(full) < 50:
            log_failure(filepath, "PDF text extraction returned <50 chars — likely image-only or corrupted")
            return None
        return full

    except Exception as e:
        log_failure(filepath, f"PDF parse error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AZW3 — Calibre conversion pipeline
# ─────────────────────────────────────────────────────────────────────────────

CALIBRE_CANDIDATES = [
    r"C:\Program Files\Calibre2\ebook-convert.exe",
    r"C:\Program Files (x86)\Calibre2\ebook-convert.exe",
    "ebook-convert",
]


def find_calibre() -> str | None:
    for p in CALIBRE_CANDIDATES:
        if p == "ebook-convert":
            if shutil.which("ebook-convert"):
                return "ebook-convert"
        elif os.path.exists(p):
            return p
    return None


def extract_azw3(filepath: str) -> str | None:
    calibre = find_calibre()
    if not calibre:
        log_failure(
            filepath,
            "AZW3 — Calibre not found. Fix:\n"
            "  1. Install Calibre: https://calibre-ebook.com/download\n"
            "  2. Rerun build.py — it will auto-detect and convert"
        )
        return None

    tmp_dir = tempfile.mkdtemp()
    try:
        epub_out = os.path.join(tmp_dir, "out.epub")
        result = subprocess.run(
            [calibre, filepath, epub_out],
            capture_output=True, text=True, timeout=120
        )
        if not os.path.exists(epub_out):
            log_failure(filepath, f"AZW3 — Calibre conversion failed: {result.stderr[:300]}")
            return None
        text = extract_epub(epub_out)
        if not text:
            log_failure(filepath, "AZW3 — Converted to EPUB but EPUB extraction returned no text")
        return text
    except subprocess.TimeoutExpired:
        log_failure(filepath, "AZW3 — Calibre timed out after 120s")
        return None
    except Exception as e:
        log_failure(filepath, f"AZW3 — Unexpected error: {e}")
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# EPUB
# ─────────────────────────────────────────────────────────────────────────────

def extract_epub(filepath: str) -> str | None:
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(filepath)
        texts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            t = soup.get_text(separator=" ").strip()
            if t:
                texts.append(t)
        full = "\n".join(texts).strip()
        return full if len(full) > 50 else None

    except ImportError:
        log_failure(
            filepath,
            "EPUB — missing libraries. Fix: pip install ebooklib beautifulsoup4"
        )
        return None
    except Exception as e:
        log_failure(filepath, f"EPUB parse error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CBR/CBZ — OCR each page image
# ─────────────────────────────────────────────────────────────────────────────

def extract_cbr(filepath: str) -> str | None:
    try:
        import pytesseract
        from PIL import Image
        import zipfile

        for p in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break

        ext  = os.path.splitext(filepath)[1].lower()
        texts = []
        img_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

        print(f"  [OCR] Comic: {os.path.basename(filepath)}")

        if ext == ".cbz":
            with zipfile.ZipFile(filepath, "r") as z:
                names = sorted(n for n in z.namelist() if n.lower().endswith(img_exts))
                for name in names:
                    with z.open(name) as f:
                        img  = Image.open(f)
                        text = pytesseract.image_to_string(img, lang="eng")
                        if text.strip():
                            texts.append(text)

        elif ext == ".cbr":
            try:
                import rarfile
                with rarfile.RarFile(filepath, "r") as r:
                    names = sorted(n for n in r.namelist() if n.lower().endswith(img_exts))
                    for name in names:
                        with r.open(name) as f:
                            img  = Image.open(f)
                            text = pytesseract.image_to_string(img, lang="eng")
                            if text.strip():
                                texts.append(text)
            except ImportError:
                log_failure(
                    filepath,
                    "CBR — rarfile not installed. Fix: pip install rarfile  "
                    "+ install WinRAR or unrar and add to PATH"
                )
                return None

        full = "\n".join(texts).strip()
        if not full:
            log_failure(filepath, "CBR/CBZ — OCR found no text (art-only comic, no speech bubbles?)")
            return None
        return full

    except ImportError as e:
        log_failure(
            filepath,
            f"CBR/CBZ — missing library ({e}). Fix:\n"
            "  pip install pytesseract Pillow rarfile\n"
            "  + install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  + install WinRAR or unrar"
        )
        return None
    except Exception as e:
        log_failure(filepath, f"CBR/CBZ error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TXT
# ─────────────────────────────────────────────────────────────────────────────

def extract_txt(filepath: str) -> str | None:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip() or None
    except Exception as e:
        log_failure(filepath, f"TXT read error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTORS = {
    ".txt":  extract_txt,
    ".pdf":  extract_pdf,
    ".epub": extract_epub,
    ".azw3": extract_azw3,
    ".mobi": extract_azw3,
    ".cbr":  extract_cbr,
    ".cbz":  extract_cbr,
}
SUPPORTED_EXTS = set(EXTRACTORS.keys())


def extract_file(filepath: str) -> str | None:
    ext = os.path.splitext(filepath)[1].lower()
    fn  = EXTRACTORS.get(ext)
    if fn is None:
        log_failure(filepath, f"Unsupported type: '{ext}'")
        return None
    return fn(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def chunk_text(text: str, target_tokens: int = 350, overlap_sentences: int = 1) -> list[str]:
    sentences: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if para:
            sentences.extend(sent_tokenize(para))

    if not sentences:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        cur_sents: list[str] = []
        cur_tokens = 0
        i = start
        while i < len(sentences):
            st = estimate_tokens(sentences[i])
            if not cur_sents or cur_tokens + st <= target_tokens:
                cur_sents.append(sentences[i])
                cur_tokens += st
                i += 1
            else:
                break
        if cur_sents:
            chunk = " ".join(cur_sents).strip()
            if chunk:
                chunks.append(chunk)
        start += max(1, len(cur_sents) - overlap_sentences)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# DB PATHS
# ─────────────────────────────────────────────────────────────────────────────

def get_db_paths(db_dir: str) -> dict:
    return {
        "faiss":          os.path.join(db_dir, "index.faiss"),
        "metadata":       os.path.join(db_dir, "metadata.json"),
        "processed":      os.path.join(db_dir, "processed_files.json"),
        "failed":         os.path.join(db_dir, "failed_files.json"),
        "failure_report": os.path.join(db_dir, "failure_report.json"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg_device = embedding_cfg.get("device", "auto")
    device     = get_device(cfg_device)
    batch_size = (
        embedding_cfg.get("batch_size_gpu", 32) if device == "cuda"
        else embedding_cfg.get("batch_size_cpu", 16)
    )
    print(f"[*] Batch size: {batch_size}")

    model_name = embedding_cfg["model"]
    print(f"[*] Loading model: {model_name}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    model.max_seq_length = embedding_cfg.get("max_seq_length", 512)

    db_dir  = paths["db_dir"]
    src_dir = paths["pdf_dir"]
    os.makedirs(db_dir, exist_ok=True)

    db = get_db_paths(db_dir)

    # ── Load existing state ───────────────────────────────────────────────────
    index_exists = os.path.exists(db["faiss"]) and os.path.exists(db["metadata"])
    if index_exists:
        print("[*] Existing index found. Loading...")
        index = faiss.read_index(db["faiss"])
        with open(db["metadata"], "r", encoding="utf-8") as f:
            metadata: list[dict] = json.load(f)
    else:
        print("[!] No existing index. Starting fresh.")
        index    = None
        metadata = []

    processed_files: set[str] = set()
    if os.path.exists(db["processed"]) and index_exists:
        with open(db["processed"], "r") as f:
            processed_files = set(json.load(f))

    # Previously failed → retry them
    previously_failed: set[str] = set()
    if os.path.exists(db["failed"]):
        with open(db["failed"], "r") as f:
            previously_failed = set(json.load(f))

    # Load old failure reasons so we don't lose them for non-retried files
    if os.path.exists(db["failure_report"]):
        with open(db["failure_report"], "r", encoding="utf-8") as f:
            failure_reasons.update(json.load(f))

    if previously_failed:
        print(f"[*] {len(previously_failed)} previously failed file(s) will be retried.")

    # ── Scan source folder ────────────────────────────────────────────────────
    all_files = [
        f for f in os.listdir(src_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]

    # Process: new files + retries (not already successfully processed)
    to_process = [
        f for f in all_files
        if f not in processed_files or f in previously_failed
    ]

    # Keep failures for files NOT being retried
    failed_files: set[str] = previously_failed - set(to_process)

    if not to_process:
        print(f"[OK] Nothing new to process.")
        print(f"     Index: {len(metadata)} chunks / {len(processed_files)} files")
        return

    print(f"[*] {len(to_process)} file(s) to process ({len(previously_failed)} retries)\n")

    CHECKPOINT_EVERY = 50
    target_tokens = chunking_cfg.get("target_tokens", 350)
    overlap_sents = chunking_cfg.get("overlap_sentences", 1)
    normalize     = embedding_cfg.get("normalize", True)

    def save_checkpoint():
        if index is None:
            return
        faiss.write_index(index, db["faiss"])
        with open(db["metadata"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        with open(db["processed"], "w") as f:
            json.dump(sorted(processed_files), f, indent=2)
        with open(db["failed"], "w") as f:
            json.dump(sorted(failed_files), f, indent=2)
        with open(db["failure_report"], "w", encoding="utf-8") as f:
            json.dump(failure_reasons, f, ensure_ascii=False, indent=2)

    total_done  = 0
    grand_total = len(to_process)

    for window_start in range(0, grand_total, CHECKPOINT_EVERY):
        window     = to_process[window_start: window_start + CHECKPOINT_EVERY]
        window_num = (window_start // CHECKPOINT_EVERY) + 1
        total_wins = (grand_total + CHECKPOINT_EVERY - 1) // CHECKPOINT_EVERY

        print(f"── Batch {window_num}/{total_wins}  "
              f"(files {window_start+1}–{min(window_start+CHECKPOINT_EVERY, grand_total)} "
              f"of {grand_total}) ──")

        batch_chunks: list[str]  = []
        batch_meta:   list[dict] = []

        for filename in tqdm(window, desc="  Extracting"):
            filepath = os.path.join(src_dir, filename)

            if not os.path.exists(filepath):
                log_failure(filename, "File missing from source directory")
                failed_files.add(filename)
                continue

            raw = extract_file(filepath)

            if not raw:
                failed_files.add(filename)
                continue

            text   = clean_text(raw)
            chunks = chunk_text(text, target_tokens=target_tokens, overlap_sentences=overlap_sents)

            if not chunks:
                log_failure(filename, "Text cleaned to zero chunks")
                failed_files.add(filename)
                continue

            # Retry succeeded — remove from failure tracking
            failed_files.discard(filename)
            failure_reasons.pop(filename, None)

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
            print("  [!] No usable chunks this batch.")
            # Save failure state even with no embeddings
            with open(db["failed"], "w") as f:
                json.dump(sorted(failed_files), f, indent=2)
            with open(db["failure_report"], "w", encoding="utf-8") as f:
                json.dump(failure_reasons, f, ensure_ascii=False, indent=2)
            continue

        print(f"  [*] Embedding {len(batch_chunks)} chunks on {device.upper()}...")
        embeddings = model.encode(
            batch_chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype("float32")

        if index is None:
            dim   = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            print(f"  [*] Created FAISS IndexFlatIP  (dim={dim})")

        index.add(embeddings)
        metadata.extend(batch_meta)

        print(f"  [*] Saving checkpoint... "
              f"({total_done}/{grand_total} files | {len(metadata)} chunks total)")
        save_checkpoint()
        print(f"  [OK] Checkpoint saved.\n")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"[DONE] Index: {len(metadata)} chunks from {len(processed_files)} files.")
    print(f"       Failed: {len(failed_files)} files.")

    if failure_reasons:
        print(f"\n  Failure summary:")
        groups: dict[str, list] = {}
        for fname, reason in failure_reasons.items():
            key = reason.split("\n")[0][:70]
            groups.setdefault(key, []).append(fname)
        for reason, files in sorted(groups.items(), key=lambda x: -len(x[1])):
            print(f"  [{len(files):>3}x] {reason}")

    print(f"\n  Full details → {db['failure_report']}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()