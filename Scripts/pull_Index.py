"""
OmnissiahCore — Scripts/pull_index.py

Pulls all large files from Hugging Face if missing on current machine.

  HF Repo 1 (index):  Onoduchiagozie/OmnissiahCore-Index  (dataset)
  HF Repo 2 (model):  Onoduchiagozie/Omni-Bge-Engine      (dataset)

Usage:
    python Scripts/pull_index.py            # download missing only
    python Scripts/pull_index.py --force    # re-download everything
    python Scripts/pull_index.py --check    # status only, no download

Called automatically on server startup via Api/server.py.
Set HF_TOKEN env var if the repos are private.
"""

import os
import sys
import argparse

INDEX_REPO  = "Onoduchiagozie/OmnissiahCore-Index"
ENGINE_REPO = "Onoduchiagozie/Omni-Bge-Engine"
REPO_TYPE   = "dataset"

INDEX_FILES = {
    "Db/faiss.index":          "Db/faiss.index",
    "Db/metadata.json":        "Db/metadata.json",
    "Db/manifest.json":        "Db/manifest.json",
    "Db/failed_files.json":    "Db/failed_files.json",
    "Db/processed_files.json": "Db/processed_files.json",
}

ENGINE_FILES = {
    "bge_m3_onnx/config.json":              "bge_m3_onnx/config.json",
    "bge_m3_onnx/model.onnx":               "bge_m3_onnx/model.onnx",
    "bge_m3_onnx/model.onnx_data":          "bge_m3_onnx/model.onnx_data",
    "bge_m3_onnx/sentencepiece.bpe.model":  "bge_m3_onnx/sentencepiece.bpe.model",
    "bge_m3_onnx/special_tokens_map.json":  "bge_m3_onnx/special_tokens_map.json",
    "bge_m3_onnx/tokenizer.json":           "bge_m3_onnx/tokenizer.json",
    "bge_m3_onnx/tokenizer_config.json":    "bge_m3_onnx/tokenizer_config.json",
}

def _size(path):
    size = os.path.getsize(path)
    for unit in ("B","KB","MB","GB"):
        if size < 1024: return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"

def _login():
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)

def _pull_file(repo_id, remote, local, force, silent):
    if not force and os.path.exists(local):
        if not silent:
            print(f"  OK  {local:<45} ({_size(local)})")
        return True
    if not silent:
        print(f"  DL  {local:<45} downloading...", end="", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=repo_id,
            filename=remote,
            repo_type=REPO_TYPE,
            local_dir=".",
            local_dir_use_symlinks=False,
        )
        if not silent: print(f"  done ({_size(local)})")
        return True
    except Exception as exc:
        print(f"\n  FAILED: {exc}")
        return False

def pull(force=False, silent=False):
    try:
        import huggingface_hub
        _login()
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    os.makedirs("Db", exist_ok=True)
    os.makedirs("bge_m3_onnx", exist_ok=True)
    all_ok = True

    if not silent: print("\n-- Index files (OmnissiahCore-Index) --")
    for local, remote in INDEX_FILES.items():
        all_ok = _pull_file(INDEX_REPO, remote, local, force, silent) and all_ok

    if not silent: print("\n-- Embedding engine (Omni-Bge-Engine) --")
    for local, remote in ENGINE_FILES.items():
        all_ok = _pull_file(ENGINE_REPO, remote, local, force, silent) and all_ok

    if not silent:
        print("\nAll files ready." if all_ok else "\nSome files failed.")
    return all_ok

def check_status():
    all_present = True
    for local in list(INDEX_FILES.keys()) + list(ENGINE_FILES.keys()):
        exists = os.path.exists(local)
        size = f"({_size(local)})" if exists else "MISSING"
        print(f"  {'OK' if exists else 'XX'}  {local:<45} {size}")
        if not exists: all_present = False
    return all_present

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if check_status() else 1)
    sys.exit(0 if pull(force=args.force, silent=False) else 1)