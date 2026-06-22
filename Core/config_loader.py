"""
OmnissiahCoreOld - Core/config_loader.py

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
    llm = profile.get("llm", {})  # Use 'llm' instead of 'ollama'

    embedding["device"] = os.getenv("OMNISSIAH_EMBED_DEVICE", embedding["device"])
    llm["url"] = os.getenv("OMNISSIAH_LLM_URL", llm.get("url", "http://localhost:11434/api/chat"))  # Default URL for LM Studio
    llm["model"] = os.getenv("OMNISSIAH_LLM_MODEL", llm.get("model", "gemma4:latest"))
    llm["num_ctx"] = _env_int("OMNISSIAH_LLM_NUM_CTX", llm.get("num_ctx", 16192))
    llm["timeout"] = _env_int("OMNISSIAH_LLM_TIMEOUT", llm.get("timeout", 900))
    llm["temperature"] = _env_float("OMNISSIAH_LLM_TEMPERATURE", llm.get("temperature", 0.4))
    llm["top_p"] = _env_float("OMNISSIAH_LLM_TOP_P", llm.get("top_p", 0.9))
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
        ["machine_role", "embedding", "retrieval", "llm", "chunking", "paths"],  # Use 'llm' instead of 'ollama'
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
        "llm",  # Use 'llm' instead of 'ollama'
        profile["llm"],
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
llm_cfg = _profile.get("llm", {})  # Use 'llm' instead of 'ollama'
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
