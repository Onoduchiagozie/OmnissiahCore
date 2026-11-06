"""Standalone development and release commands. Python 3.10+, Git and OpenSSH."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import uuid
import urllib.request

ROOT = Path(__file__).resolve().parent
PRIVATE = Path.home() / ".config" / "omnissiahcore"
RUNTIME_ROOTS = ("Api/", "Core/", "Scripts/")
RUNTIME_FILES = {"config.json", "app_text.json", "main.py", "manage.py",
                 "requirements-runtime.txt", "requirements-build.txt", "requirements-test.txt"}


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def runtime_path(name):
    path = Path(name)
    return (not path.is_absolute() and ".." not in path.parts and
            (name in RUNTIME_FILES or
             (name.startswith(RUNTIME_ROOTS) and name.endswith(".py"))))


def private_file(name):
    return PRIVATE / name


def write_private(path, data):
    path = path.expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Private settings must be outside the source checkout.")
    if path.exists():
        raise ValueError("Settings already exist; edit them instead of overwriting them.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    if os.name != "nt":
        path.chmod(0o600)
    print("Created private settings:", path)


def initialize(args):
    write_private(private_file("settings.json"), {
        "active_profile": "lenovo_build",
        "data_root": str(Path.home() / "OmnissiahCoreData"),
        "environment": {},
        "profiles": {"lenovo_build": {"llm": {
            "provider": "openai", "url": "http://127.0.0.1:1234/v1/chat/completions",
            "model": "replace-with-your-loaded-local-model"
        }}}
    })


def python_in_venv():
    return ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def setup(args):
    target = python_in_venv()
    if not target.exists():
        run([sys.executable, "-m", "venv", ROOT / ".venv"])
    run([target, "-m", "pip", "install", "-r", ROOT / "requirements-test.txt"])
    if args.full:
        # Choose the wheel index from pytorch.org for the actual machine/driver.
        index = args.torch_index or "https://download.pytorch.org/whl/cpu"
        run([target, "-m", "pip", "install", "torch==2.12.1", "--index-url", index])
        run([target, "-m", "pip", "install", "-r", ROOT / "requirements-runtime.txt"])
        if args.build:
            run([target, "-m", "pip", "install", "-r", ROOT / "requirements-build.txt"])
    print("Use this Python executable for the commands below:", target)


def doctor(args):
    from Core.config_loader import paths, embedding_cfg, llm_cfg, active_profile
    import importlib.util
    checks = {"python_3_10_plus": sys.version_info >= (3, 10),
              "private_settings": private_file("settings.json").exists() or
                  bool(os.environ.get("OMNISSIAH_PRIVATE_CONFIG"))}
    for package in ("fastapi", "uvicorn", "requests", "numpy", "faiss", "torch",
                    "sentence_transformers", "bm25s"):
        checks[package] = importlib.util.find_spec(package) is not None
    checks["faiss_index"] = Path(paths["faiss"]).is_file()
    checks["metadata"] = Path(paths["metadata"]).is_file()
    if llm_cfg.get("provider") == "gemini":
        checks["provider_credentials"] = bool(os.environ.get("GEMINI_API_KEY"))
    else:
        checks["local_model_selected"] = not llm_cfg["model"].startswith("replace-")
    if checks["torch"]:
        import torch
        checks["requested_gpu_available"] = (embedding_cfg["device"] != "cuda" or
                                             torch.cuda.is_available())
        print("CUDA available:", torch.cuda.is_available())
    print("Profile:", active_profile)
    print("Data directory:", paths["db_dir"])
    for name, okay in checks.items():
        print(("OK      " if okay else "MISSING ") + name)
    print("Scene Archive additionally requires its private scene database/index files.")
    return 0 if all(checks.values()) else 1


def test(args):
    run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "Tests" / "unit"), "-v"],
        cwd=ROOT)


def serve(args):
    from Core.config_loader import paths
    for key in ("faiss", "metadata"):
        if not Path(paths[key]).is_file():
            raise ValueError("Missing private index data. Run 'doctor' and copy or build your index first.")
    run([sys.executable, "-m", "uvicorn", "Api.server:app",
         "--host", args.host, "--port", str(args.port)] +
        (["--reload"] if args.reload else []), cwd=ROOT)


def api_request(base, path, body=None):
    request = urllib.request.Request(base.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def smoke(args):
    health = api_request(args.url, "/health")
    if health.get("status") != "online" or not health.get("metadata_loaded", 0):
        raise ValueError("API responded but the index is not ready.")
    inspection = api_request(args.url, "/query/inspect",
                             {"query": args.query, "top_k": 3})
    hits = inspection.get("grounded_hits") or inspection.get("dense_hits") or inspection.get("results")
    # The inspect API nests its stage arrays under inspection in some releases.
    if not hits and isinstance(inspection.get("inspection"), dict):
        hits = inspection["inspection"].get("grounded_hits")
    print("Health: online; index loaded:", health["metadata_loaded"])
    print("Inspect response fields:", ", ".join(inspection))
    if not hits:
        raise ValueError("No retrieval candidates were returned; inspect the API response before publishing.")
    print("Retrieval candidates:", len(hits))
    print("This checks transport and retrieval; it does not certify answer accuracy or spend LLM quota.")


def scan(args):
    """Inspect tracked source without echoing potential secret values."""
    names = git("ls-files", "-z").split("\0")
    prohibited = (".env", ".pem", ".key", ".pfx", ".p12", ".pdf", ".epub", ".mobi",
                  ".azw3", ".db", ".sqlite", ".index", ".onnx", ".npy")
    issues = []
    patterns = [re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
                re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
                re.compile(rb"gh[pousr]_[0-9A-Za-z]{30,}")]
    for name in names:
        if not name:
            continue
        path = ROOT / name
        if name.endswith(prohibited) or name.endswith(".private.json") or "/.env" in name:
            issues.append(name + ": private/data file is tracked")
        if path.is_file() and path.stat().st_size < 2_000_000:
            content = path.read_bytes()
            if any(pattern.search(content) for pattern in patterns):
                issues.append(name + ": possible embedded credential")
    if issues:
        raise ValueError("\n".join(issues))
    print("Tracked-file scan passed. No credential values were printed.")


def require_main():
    if git("status", "--porcelain"):
        raise ValueError("Publish from a clean checkout; commit or preserve outstanding work first.")
    if git("branch", "--show-current") != "main":
        raise ValueError("Only main can be published.")
    git("fetch", "origin", "main")
    commit = git("rev-parse", "HEAD")
    if commit != git("rev-parse", "origin/main"):
        raise ValueError("Local main must exactly match origin/main.")
    return commit


def package(commit, destination):
    """Package only runtime source read directly from a Git commit."""
    manifest = {"commit": commit, "files": {}}
    with tarfile.open(destination, "w:gz") as archive:
        for line in git("ls-tree", "-r", commit).splitlines():
            metadata, name = line.split("\t", 1)
            if not runtime_path(name):
                continue
            if metadata.split()[0] != "100644" and metadata.split()[0] != "100755":
                raise ValueError("Runtime source must be regular files.")
            data = subprocess.check_output(["git", "-C", str(ROOT), "show", commit + ":" + name])
            manifest["files"][name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
        data = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo("release.json")
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
    return manifest


def connection(args):
    path = Path(args.connection).expanduser()
    settings = json.loads(path.read_text(encoding="utf-8-sig"))
    host = settings["ssh_host"]
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", host) or host.startswith("-"):
        raise ValueError("Invalid SSH host.")
    options = ["-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15"]
    if settings.get("identity_file"):
        options += ["-i", str(Path(settings["identity_file"]).expanduser())]
    return settings, host, options


def remote_command(args, operation, archive=None, commit=None):
    settings, host, options = connection(args)
    remote_args = ["python3", "-", operation, "--app", settings["app_root"],
                   "--service", settings.get("service", "omnissiah-api.service"),
                   "--url", settings.get("health_url", "http://127.0.0.1:8000")]
    if archive:
        remote_args += ["--archive", archive, "--commit", commit]
    if getattr(args, "apply", None):
        if args.apply != commit:
            raise ValueError("--apply must be the full main commit shown by prepare.")
        remote_args += ["--apply", args.apply]
    if getattr(args, "backup", None):
        remote_args += ["--backup", args.backup]
    script = (ROOT / "tools" / "deploy_oracle.py").read_text(encoding="utf-8")
    run(["ssh", *options, host, shlex.join(remote_args)],
        input=script, text=True)


def publish(args):
    commit = require_main()
    scan(args)
    test(args)
    settings, host, options = connection(args)
    with tempfile.TemporaryDirectory(prefix="omnissiah-release-") as temp:
        archive = Path(temp) / "release.tar.gz"
        package(commit, archive)
        remote_archive = "/tmp/omnissiah-release-" + uuid.uuid4().hex + ".tar.gz"
        run(["scp", *options, archive, host + ":" + remote_archive])
        remote_command(args, "publish", remote_archive, commit)
    print("Main commit:", commit)
    if not args.apply:
        print("Prepared only. Review the backup and changes, then publish --apply", commit)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create private settings outside Git").set_defaults(handler=initialize)
    install = commands.add_parser("setup", help="Create a venv; tests only unless --full")
    install.add_argument("--full", action="store_true")
    install.add_argument("--build", action="store_true")
    install.add_argument("--torch-index", help="Official PyTorch wheel index for this machine")
    install.set_defaults(handler=setup)
    for name, handler in (("doctor", doctor), ("test", test), ("scan", scan)):
        commands.add_parser(name).set_defaults(handler=handler)
    start = commands.add_parser("api", help="Start a local-only backend")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8000)
    start.add_argument("--reload", action="store_true")
    start.set_defaults(handler=serve)
    check = commands.add_parser("smoke", help="Check a running API without generation")
    check.add_argument("--url", default="http://127.0.0.1:8000")
    check.add_argument("--query", default="Horus Lupercal")
    check.set_defaults(handler=smoke)
    for name, script in (("cli", "CLI.py"), ("build", "build_db.py"), ("verify", "verify_db.py")):
        command = commands.add_parser(name)
        command.set_defaults(handler=lambda args, script=script:
            run([sys.executable, ROOT / "Scripts" / script], cwd=ROOT).returncode)
    for name in ("publish", "status", "rollback"):
        command = commands.add_parser(name)
        command.add_argument("--connection", default=str(private_file("publish.json")))
        if name == "publish":
            command.add_argument("--apply", metavar="FULL_MAIN_SHA")
            command.set_defaults(handler=publish)
        elif name == "rollback":
            command.add_argument("--backup", required=True)
            command.set_defaults(handler=lambda args: remote_command(args, "rollback"))
        else:
            command.set_defaults(handler=lambda args: remote_command(args, "status"))
    return result


def main():
    args = parser().parse_args()
    try:
        return args.handler(args) or 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        # CalledProcessError text omits input and private settings.
        print("ERROR:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
