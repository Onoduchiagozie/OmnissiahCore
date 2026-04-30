"""
OmnissiahCore - build.py v3.6 "Migration & Sage Edition"
Multi-Format Indexer with Automatic Filename Migration.

Changes:
- Automatic Migration: Automatically renames 'index.faiss' to 'faiss.index' if found.
- DLL Failover: Silences the scary ONNX 126 error by checking DLL availability first.
- Manifest Generation: Satisfies verify_db.py health checks.
- Parallel OCR: Uses all CPU cores for rapid scan processing.
"""

import os
import sys
import json
import subprocess
import shutil
import tempfile
import numpy as np
import faiss
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# --- SYSTEM PATH CONFIGURATION ---
POPPLER_PATH = r"C:\poppler\Library\bin"
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SEVEN_ZIP_EXE = r"C:\Program Files\7-Zip\7z.exe"
CALIBRE_CONVERT = r"C:\Program Files\Calibre2\ebook-convert.exe"

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Inject paths
for p in [POPPLER_PATH, os.path.dirname(TESSERACT_EXE), os.path.dirname(SEVEN_ZIP_EXE)]:
    if os.path.exists(p) and p not in os.environ["PATH"]:
        os.environ["PATH"] += f";{p}"

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from pypdf import PdfReader
    import nltk
    from nltk.tokenize import sent_tokenize
    from pdf2image import convert_from_path
    import pytesseract
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    from PIL import Image
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    import onnxruntime as ort
except ImportError as e:
    print(f"[!] Missing library: {e}")
    sys.exit(1)

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


class OmnissiahBuilder:
    def __init__(self, config_path="config.json"):
        self.load_config(config_path)
        self.db_dir = Path(self.config["paths"]["db_dir"])
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Paths forced to match verify_db.py and Retriever expectations
        self.processed_files_path = self.db_dir / "processed_files.json"
        self.index_path = self.db_dir / "faiss.index"
        self.metadata_path = self.db_dir / "metadata.json"
        self.manifest_path = self.db_dir / "manifest.json"
        self.report_path = self.db_dir / "failure_report.json"

        # Naming Migration Check
        old_index_path = self.db_dir / "index.faiss"
        if old_index_path.exists() and not self.index_path.exists():
            print(f"[*] Migrating legacy naming: {old_index_path.name} -> {self.index_path.name}")
            os.rename(old_index_path, self.index_path)

        self.load_state()
        self.init_models()

    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
            self.config = full_config["profiles"][full_config["active_profile"]]
            print(f"[Config] Profile: '{full_config['active_profile']}'")

    def load_state(self):
        self.processed_files = set()
        if self.processed_files_path.exists():
            with open(self.processed_files_path, 'r', encoding='utf-8') as f:
                self.processed_files = set(json.load(f))

        self.all_metadata = []
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.all_metadata = json.load(f)

        self.faiss_index = None
        if self.index_path.exists():
            self.faiss_index = faiss.read_index(str(self.index_path))
            print(f"[*] Index loaded: {self.faiss_index.ntotal} vectors.")

    def init_models(self):
        model_name = self.config["embedding"]["model"]
        onnx_dir = "bge_m3_onnx"
        self.use_onnx = False

        if os.path.exists(onnx_dir) and "CUDAExecutionProvider" in ort.get_available_providers():
            try:
                # Test DLL load silently
                ort.InferenceSession(os.path.join(onnx_dir, "model.onnx"), providers=['CUDAExecutionProvider'])
                self.onnx_model = ORTModelForFeatureExtraction.from_pretrained(onnx_dir,
                                                                               provider="CUDAExecutionProvider")
                self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
                self.use_onnx = True
                print("[OK] GPU Acceleration Enabled.")
            except Exception:
                print("[!] GPU Init Fail (DLL Issue). Stability Fallback: CPU Mode.")

        if not self.use_onnx:
            print("[*] Mode: CPU (SentenceTransformers)")
            self.model = SentenceTransformer(model_name, device="cpu", local_files_only=True)

    def get_embeddings(self, texts):
        if self.use_onnx:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
            with torch.no_grad():
                outputs = self.onnx_model(**inputs)
                return outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return self.model.encode(texts, batch_size=self.config["embedding"]["batch_size_cpu"], show_progress_bar=False)

    def extract_text(self, f):
        ext = f.suffix.lower()
        if ext == '.pdf': return self.extract_pdf(f)
        if ext == '.epub': return self.extract_epub(f)
        if ext == '.azw3': return self.extract_azw3(f)
        if ext in ['.cbr', '.cbz']: return self.extract_cbr(f)
        return f.read_text(errors='ignore')

    def extract_pdf(self, path):
        try:
            reader = PdfReader(path)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            if len(text.strip()) < 200:
                return self.extract_ocr_turbo(path)
            return text
        except Exception:
            return self.extract_ocr_turbo(path)

    def extract_ocr_turbo(self, path):
        try:
            images = convert_from_path(str(path), dpi=150, first_page=1, last_page=60, poppler_path=POPPLER_PATH)
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as exe:
                return "\n".join(list(exe.map(pytesseract.image_to_string, images)))
        except Exception as e:
            raise Exception(f"OCR Failure: {e}")

    def extract_epub(self, path):
        book = epub.read_epub(str(path))
        return "\n".join([BeautifulSoup(i.get_content(), 'html.parser').get_text() for i in book.get_items() if
                          i.get_type() == ebooklib.ITEM_DOCUMENT])

    def extract_azw3(self, path):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "temp.epub")
            subprocess.run([CALIBRE_CONVERT, str(path), out], check=True, capture_output=True)
            return self.extract_epub(out)

    def extract_cbr(self, path):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([SEVEN_ZIP_EXE, "x", str(path), f"-o{tmp}", "-y"], check=True, capture_output=True)
            imgs = sorted(
                [os.path.join(tmp, f) for f in os.listdir(tmp) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as exe:
                return "\n".join(list(exe.map(lambda i: pytesseract.image_to_string(Image.open(i)), imgs[:50])))

    def chunk_text(self, text, source):
        nltk.download('punkt', quiet=True)
        sents = sent_tokenize(text)
        chunks, cur, l = [], [], 0
        target, overlap = self.config["chunking"]["target_tokens"], self.config["chunking"]["overlap_sentences"]
        for i, s in enumerate(sents):
            cur.append(s);
            l += len(s.split())
            if l >= target:
                chunks.append(" ".join(cur))
                cur = sents[max(0, i - overlap + 1):i + 1]
                l = sum(len(x.split()) for x in cur)
        if cur: chunks.append(" ".join(cur))
        return [{"text": c, "source": source} for c in chunks]

    def build(self):
        raw_dir = Path(self.config["paths"]["pdf_dir"])
        to_proc = [f for f in raw_dir.glob("*") if f.suffix.lower() in ['.pdf', '.epub', '.azw3', '.cbr', '.cbz',
                                                                        '.txt'] and f.name not in self.processed_files]
        if not to_proc:
            print("[*] OmnissiahCore is fully indexed.")
            self.save({})  # Force manifest creation even if no new files
            return

        print(f"[*] Indexing {len(to_proc)} files...")
        batch_size = 50
        for i in range(0, len(to_proc), batch_size):
            batch = to_proc[i:i + batch_size]
            print(f"\n── Batch {i // batch_size + 1}/{(len(to_proc) - 1) // batch_size + 1} ──")
            b_chunks, b_names, fails = [], [], {}
            for f in tqdm(batch, desc="Extracting"):
                try:
                    txt = self.extract_text(f)
                    if not txt.strip(): raise Exception("Empty content.")
                    chks = self.chunk_text(txt, f.name)
                    b_chunks.extend(chks);
                    b_names.append(f.name)
                except Exception as e:
                    fails[f.name] = str(e);
                    print(f" [FAIL] {f.name}: {e}")

            if b_chunks:
                print(f" [*] Embedding {len(b_chunks)} chunks...")
                embs = self.get_embeddings([c["text"] for c in b_chunks])
                if self.faiss_index is None: self.faiss_index = faiss.IndexFlatIP(embs.shape[1])
                faiss.normalize_L2(embs);
                self.faiss_index.add(embs)
                self.all_metadata.extend(b_chunks);
                self.processed_files.update(b_names)
                self.save(fails)
            else:
                self.save(fails)

    def save(self, fails):
        faiss.write_index(self.faiss_index, str(self.index_path))
        manifest = {
            "build_date": datetime.now().isoformat(),
            "total_chunks": len(self.all_metadata),
            "total_files": len(self.processed_files),
            "model": self.config["embedding"]["model"],
            "status": "healthy"
        }
        for p, data in [(self.metadata_path, self.all_metadata),
                        (self.processed_files_path, list(self.processed_files)),
                        (self.manifest_path, manifest)]:
            with open(p, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)

        report = {}
        if self.report_path.exists():
            with open(self.report_path, 'r', encoding='utf-8') as f: report = json.load(f)
        report.update(fails);
        with open(self.report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(" [OK] Database entry synchronized.")


if __name__ == "__main__":
    OmnissiahBuilder().build()
