"""
OmnissiahCore - Core/config_loader.py

Central configuration loader. Every module imports from here.
Never hardcode paths, model names, or parameters anywhere else.
"""

import json
import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def _fatal(message: str):
    print(f"[FATAL] {message}")
    sys.exit(1)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        _fatal(f"Environment variable {name} must be an integer, got: {value}")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        _fatal(f"Environment variable {name} must be a number, got: {value}")


def _require_keys(section_name: str, data: dict, keys: list[str]):
    missing = [key for key in keys if key not in data]
    if missing:
        _fatal(f"Profile section '{section_name}' is missing keys: {', '.join(missing)}")


def _apply_env_overrides(profile: dict) -> dict:
    profile = json.loads(json.dumps(profile))

    override_role = os.getenv("OMNISSIAH_MACHINE_ROLE")
    if override_role:
        profile["machine_role"] = override_role

    embedding = profile["embedding"]
    retrieval = profile["retrieval"]
    ollama = profile["ollama"]

    embedding["device"] = os.getenv("OMNISSIAH_EMBED_DEVICE", embedding["device"])
    ollama["url"] = os.getenv("OMNISSIAH_OLLAMA_URL", ollama["url"])
    ollama["model"] = os.getenv("OMNISSIAH_OLLAMA_MODEL", ollama["model"])
    ollama["num_ctx"] = _env_int("OMNISSIAH_OLLAMA_NUM_CTX", ollama["num_ctx"])
    ollama["timeout"] = _env_int("OMNISSIAH_OLLAMA_TIMEOUT", ollama["timeout"])
    ollama["temperature"] = _env_float("OMNISSIAH_OLLAMA_TEMPERATURE", ollama["temperature"])
    ollama["top_p"] = _env_float("OMNISSIAH_OLLAMA_TOP_P", ollama["top_p"])
    retrieval["top_k"] = _env_int("OMNISSIAH_TOP_K", retrieval["top_k"])
    retrieval["candidate_pool"] = _env_int("OMNISSIAH_CANDIDATE_POOL", retrieval["candidate_pool"])
    retrieval["stitching_window"] = _env_int(
        "OMNISSIAH_STITCHING_WINDOW",
        retrieval["stitching_window"],
    )
    return profile


def _load_config() -> tuple[str, dict]:
    if not os.path.exists(CONFIG_PATH):
        _fatal(f"config.json not found at: {CONFIG_PATH}")

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        _fatal(f"config.json is malformed: {e}")

    active = os.getenv("OMNISSIAH_ACTIVE_PROFILE", raw.get("active_profile", "lenovo_build"))
    profiles = raw.get("profiles", {})
    if active not in profiles:
        _fatal(f"Active profile '{active}' not found in config.json")

    profile = profiles[active]
    _require_keys(
        "profile",
        profile,
        ["machine_role", "embedding", "retrieval", "ollama", "chunking", "paths"],
    )
    _require_keys(
        "embedding",
        profile["embedding"],
        ["model", "max_seq_length", "normalize", "batch_size_gpu", "batch_size_cpu", "device"],
    )
    _require_keys(
        "retrieval",
        profile["retrieval"],
        ["use_faiss", "use_bm25", "use_reranker", "candidate_pool", "top_k", "stitching_window", "rrf_k"],
    )
    _require_keys(
        "ollama",
        profile["ollama"],
        ["url", "model", "num_ctx", "temperature", "top_p", "max_tokens", "stream", "timeout"],
    )
    _require_keys("chunking", profile["chunking"], ["target_tokens", "overlap_sentences"])
    _require_keys("paths", profile["paths"], ["pdf_dir", "failed_dir", "db_dir"])

    profile = _apply_env_overrides(profile)
    print(f"[Config] Loaded profile: '{active}' | role: {profile.get('machine_role', '?')}")
    return active, profile


active_profile, _profile = _load_config()

embedding_cfg = _profile["embedding"]
retrieval_cfg = _profile["retrieval"]
ollama_cfg = _profile["ollama"]
chunking_cfg = _profile["chunking"]
machine_role = _profile["machine_role"]

_raw_paths = _profile["paths"]
paths = {
    "pdf_dir": os.path.join(BASE_DIR, _raw_paths["pdf_dir"]),
    "failed_dir": os.path.join(BASE_DIR, _raw_paths["failed_dir"]),
    "db_dir": os.path.join(BASE_DIR, _raw_paths["db_dir"]),
    "faiss": os.path.join(BASE_DIR, _raw_paths["db_dir"], "faiss.index"),
    "metadata": os.path.join(BASE_DIR, _raw_paths["db_dir"], "metadata.json"),
    "manifest": os.path.join(BASE_DIR, _raw_paths["db_dir"], "manifest.json"),
    "processed": os.path.join(BASE_DIR, _raw_paths["db_dir"], "processed_files.json"),
    "failed_log": os.path.join(BASE_DIR, _raw_paths["db_dir"], "failed_files.json"),
    "base": BASE_DIR,
}

cfg = _profile
