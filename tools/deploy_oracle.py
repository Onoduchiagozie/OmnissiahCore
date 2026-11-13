"""Oracle release helper, sent over SSH by manage.py. No credentials are packaged."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import py_compile
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

TOP_FILES = {"config.json", "app_text.json", "main.py", "manage.py",
             "requirements-runtime.txt", "requirements-build.txt", "requirements-test.txt",
             "requirements-graph.txt", "requirements-graph-test.txt", "requirements-langgraph.txt"}


def permitted(name):
    p = PurePosixPath(name)
    return (not p.is_absolute() and ".." not in p.parts and "\\" not in name and
            (name in TOP_FILES or (len(p.parts) > 1 and p.parts[0] in
             ("Core", "Api", "Scripts") and name.endswith(".py"))))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def target(root, name):
    root = root.resolve()
    if name != "release.json" and not permitted(name):
        raise ValueError("Invalid release path.")
    p = root / name
    if root not in p.resolve().parents:
        raise ValueError("Release path escapes application directory.")
    return p


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".release-", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, name)
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(data, output, indent=2)
            output.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def request(base, path):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=10) as response:
        return json.load(response)


def ready(base):
    payload = request(base, "/health")
    if payload.get("status") != "online" or not payload.get("metadata_loaded", 0):
        raise ValueError("The live API is not ready.")
    return payload


def idle(base):
    ready(base)
    status = request(base, "/llm/status")
    if status.get("active", 0) or status.get("queued", 0):
        raise ValueError("Requests are active or queued. Wait before publishing.")


def service(action, name):
    subprocess.run(["sudo", "-n", "systemctl", action, name], check=True)


def wait_ready(base, commit=None, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ready(base)
            if commit is None or request(base, "/version").get("commit") == commit:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("The service did not become ready with the expected release.")


def unpack(archive, stage, expected_commit):
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        seen = set()
        for member in members:
            if (member.name in seen or not member.isfile() or
                (member.name != "release.json" and not permitted(member.name)) or
                member.size > 5_000_000):
                raise ValueError("Archive contains a disallowed file.")
            seen.add(member.name)
        if sum(member.size for member in members) > 25_000_000:
            raise ValueError("Release archive exceeds the source-only size limit.")
        for member in members:
            output = target(stage, member.name)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(source.extractfile(member).read())
    manifest = read_json(stage / "release.json")
    if manifest["commit"] != expected_commit or not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("Release commit mismatch.")
    if seen != set(manifest["files"]) | {"release.json"}:
        raise ValueError("Release manifest does not match the archive.")
    for name, expected in manifest["files"].items():
        path = target(stage, name)
        if digest(path) != expected:
            raise ValueError("Release file checksum mismatch: " + name)
        if name.endswith(".py"):
            py_compile.compile(str(path), doraise=True)
    return manifest


def snapshot(app, manifest, backups):
    previous = read_json(app / "release.json") if (app / "release.json").is_file() else None
    names = set(manifest["files"]) | {"release.json"}
    if previous:
        names.update(previous.get("files", {}))
    before = {name: digest(target(app, name)) for name in sorted(names)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backups / (stamp + "-" + manifest["commit"][:12])
    backup.mkdir(parents=True)
    for name, old_hash in before.items():
        if old_hash:
            atomic_copy(target(app, name), target(backup, name))
    plan = {"commit": manifest["commit"], "backup": str(backup),
            "before": before, "files": manifest["files"], "previous": previous,
            "created_at": stamp}
    write_json(backup / "plan.json", plan)
    return plan


def verify_before(app, plan):
    for name, expected in plan["before"].items():
        if digest(target(app, name)) != expected:
            raise ValueError("Live files changed after preparation: " + name + ". Prepare again.")


def validate_backup(app, plan):
    backup = Path(plan["backup"]).resolve()
    for name, expected in plan["before"].items():
        destination = target(app, name)
        if expected:
            original = target(backup, name)
            if digest(original) != expected:
                raise ValueError("Backup checksum mismatch: " + name)
        elif destination.exists() and not destination.is_file():
            raise ValueError("Rollback target is not a regular file.")


def restore(app, plan):
    validate_backup(app, plan)
    backup = Path(plan["backup"]).resolve()
    for name, expected in plan["before"].items():
        destination = target(app, name)
        if expected:
            atomic_copy(target(backup, name), destination)
        elif destination.exists():
            destination.unlink()


def apply_release(app, stage, manifest, plan, service_name, base):
    verify_before(app, plan)
    validate_backup(app, plan)
    idle(base)
    service("stop", service_name)
    try:
        for name in plan["before"]:
            if name != "release.json" and name not in manifest["files"]:
                obsolete = target(app, name)
                if obsolete.exists():
                    obsolete.unlink()
        for name in manifest["files"]:
            atomic_copy(target(stage, name), target(app, name))
        manifest["deployed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(app / "release.json", manifest)
        service("start", service_name)
        wait_ready(base, manifest["commit"])
    except Exception:
        # Restore exact previous source, including removing newly added files.
        service("stop", service_name)
        restore(app, plan)
        service("start", service_name)
        wait_ready(base, (plan.get("previous") or {}).get("commit"))
        raise RuntimeError("Release failed; the previous source was restored and is responding.") from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["publish", "status", "rollback"])
    parser.add_argument("--app", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--archive")
    parser.add_argument("--commit")
    parser.add_argument("--apply")
    parser.add_argument("--backup")
    args = parser.parse_args()
    app = Path(args.app).resolve()
    if not (app / "Api").is_dir() or not (app / "Core").is_dir():
        raise ValueError("Expected an existing OmnissiahCore application directory.")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", args.service):
        raise ValueError("Invalid service name.")
    backups = app / "backups" / "releases"
    if args.operation == "status":
        health = ready(args.url)
        release = read_json(app / "release.json") if (app / "release.json").exists() else {}
        mismatches = [name for name, expected in release.get("files", {}).items()
                      if digest(target(app, name)) != expected]
        print(json.dumps({"status": health["status"], "commit": release.get("commit", "unrecorded"),
                          "metadata_loaded": health.get("metadata_loaded"), "modified_files": mismatches}))
        return 1 if mismatches else 0
    # Serialize preparations, activations and rollbacks on this Linux host.
    import fcntl
    backups.mkdir(parents=True, exist_ok=True)
    deployment_lock = (backups / ".deploy.lock").open("a")
    try:
        fcntl.flock(deployment_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("Another release operation is running.") from None
    if args.operation == "rollback":
        backup = Path(args.backup).resolve()
        if backups.resolve() not in backup.parents:
            raise ValueError("Rollback backup must be inside this app's release-backup directory.")
        plan = read_json(backup / "plan.json")
        if Path(plan["backup"]).resolve() != backup:
            raise ValueError("Backup record points to another directory.")
        # Reject rollback of an unrelated current release.
        current = read_json(app / "release.json")
        if current.get("commit") != plan["commit"]:
            raise ValueError("This backup does not belong to the currently deployed release.")
        validate_backup(app, plan)
        idle(args.url)
        service("stop", args.service)
        restore(app, plan)
        service("start", args.service)
        wait_ready(args.url, (plan.get("previous") or {}).get("commit"))
        print(json.dumps({"rolled_back": True, "restored_commit": (plan.get("previous") or {}).get("commit", "legacy")}))
        return 0
    if not args.archive or not args.commit:
        raise ValueError("Archive and commit are required.")
    archive = Path(args.archive).resolve()
    if archive.parent != Path("/tmp") or not archive.name.startswith("omnissiah-release-"):
        raise ValueError("Unexpected upload path.")
    try:
        with tempfile.TemporaryDirectory(prefix="omnissiah-stage-") as temp:
            stage = Path(temp).resolve()
            manifest = unpack(archive, stage, args.commit)
            # No second model/index load, no LLM call and no environment mutation.
            env = dict(os.environ, OMNISSIAH_ACTIVE_PROFILE="dell_query", OMNISSIAH_DATA_DIR=str(app))
            subprocess.run([str(app / ".venv/bin/python"), "-c",
                "from Core.config_loader import paths; from pathlib import Path; "
                "assert Path(paths['faiss']).is_file(); assert Path(paths['metadata']).is_file()"],
                cwd=stage, env=env, check=True)
            pending = backups / ("pending-" + args.commit + ".json")
            if args.apply:
                if args.apply != args.commit or not pending.exists():
                    raise ValueError("Prepare this exact main commit before applying it.")
                plan = read_json(pending)
                if plan["files"] != manifest["files"]:
                    raise ValueError("Archive differs from the prepared release.")
                apply_release(app, stage, manifest, plan, args.service, args.url)
                write_json(Path(plan["backup"]) / "outcome.json",
                           {"deployed_commit": args.commit, "healthy": True})
                pending.unlink()
                print(json.dumps({"deployed_commit": args.commit, "healthy": True,
                                  "rollback_backup": plan["backup"]}))
            else:
                plan = snapshot(app, manifest, backups)
                write_json(pending, plan)
                changed = [name for name in plan["before"] if
                           plan["before"][name] != manifest["files"].get(name)]
                print(json.dumps({"prepared_commit": args.commit, "backup": plan["backup"],
                                  "changed_files": changed, "production_changed": False}))
    finally:
        archive.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR:", str(exc), file=sys.stderr)
        raise SystemExit(1)
