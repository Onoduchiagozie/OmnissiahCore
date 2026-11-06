import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import manage
from Core.private_config import load_settings
from tools import deploy_oracle as deploy


class PrivateSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.defaults = self.root / "defaults.json"
        self.defaults.write_text(json.dumps({"active_profile": "local", "profiles": {
            "local": {"paths": {"db_dir": "Db"}, "llm": {"model": "default", "timeout": 10}}}}))
        self.private = self.root / "settings.json"

    def load(self, data, environment=None):
        self.private.write_text(json.dumps(data))
        with patch.dict(os.environ, dict(environment or {},
                         OMNISSIAH_PRIVATE_CONFIG=str(self.private)), clear=True):
            result = load_settings(self.defaults)
            return result, dict(os.environ)

    def test_partial_override_preserves_defaults(self):
        data, _ = self.load({"profiles": {"local": {"llm": {"model": "local-model"}}}})
        self.assertEqual(data["profiles"]["local"]["llm"],
                         {"model": "local-model", "timeout": 10})

    def test_process_environment_wins_over_private_file(self):
        _, environment = self.load({"environment": {"EXAMPLE_SETTING": "file"}},
                                   {"EXAMPLE_SETTING": "process"})
        self.assertEqual(environment["EXAMPLE_SETTING"], "process")

    def test_relative_data_directory_is_relative_to_private_file(self):
        data, _ = self.load({"data_root": "../data"})
        self.assertEqual(Path(data["data_root"]), (self.root / "../data").resolve())

    def test_explicit_missing_configuration_fails(self):
        with patch.dict(os.environ, {"OMNISSIAH_PRIVATE_CONFIG": str(self.private)}, clear=True):
            with self.assertRaises(FileNotFoundError):
                load_settings(self.defaults)

    def test_reject_non_string_environment(self):
        with self.assertRaises(ValueError):
            self.load({"environment": {"EXAMPLE_SETTING": 4}})

    def test_initializer_refuses_checkout_path_and_existing_settings(self):
        with self.assertRaises(ValueError):
            manage.write_private(manage.ROOT / "private.json", {})
        self.private.write_text("{}")
        with self.assertRaises(ValueError):
            manage.write_private(self.private, {})

    def test_api_configuration_uses_private_data_root(self):
        self.private.write_text(json.dumps({"data_root": str(self.root / "data"),
                                           "active_profile": "dell_query"}))
        env = dict(os.environ, OMNISSIAH_PRIVATE_CONFIG=str(self.private))
        env.pop("OMNISSIAH_DATA_DIR", None)
        env.pop("OMNISSIAH_ACTIVE_PROFILE", None)
        result = subprocess.check_output([sys.executable, "-c",
            "from Core.config_loader import paths; print(paths['faiss'])"],
            cwd=manage.ROOT, env=env, text=True)
        self.assertIn(str(self.root / "data" / "Db" / "faiss.index"), result)


class PackagingTests(unittest.TestCase):
    def test_runtime_allowlist_excludes_data_keys_and_experiments(self):
        for name in ("Core/agent.py", "Api/routes/query_routes.py", "config.json"):
            self.assertTrue(manage.runtime_path(name))
        for name in (".env", "Data/book.pdf", "Db/metadata.json", "../Core/agent.py",
                     "staging/openai-migration/agent.py", "/Core/agent.py"):
            self.assertFalse(manage.runtime_path(name))

    def test_package_reads_committed_content_only(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "source.tar.gz"
            commit = manage.git("rev-parse", "HEAD")
            manifest = manage.package(commit, destination)
            with tarfile.open(destination) as archive:
                self.assertEqual(set(archive.getnames()), set(manifest["files"]) | {"release.json"})
                self.assertNotIn("tools/deploy_oracle.py", archive.getnames())
                for name, expected in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(), expected)

    def test_publish_rejects_dirty_checkout(self):
        with patch.object(manage, "git", return_value=" M Core/agent.py"):
            with self.assertRaisesRegex(ValueError, "clean"):
                manage.require_main()

    def test_publish_rejects_non_main_and_stale_main(self):
        with patch.object(manage, "git", side_effect=["", "work"]):
            with self.assertRaisesRegex(ValueError, "Only main"):
                manage.require_main()
        with patch.object(manage, "git", side_effect=["", "main", "", "a"*40, "b"*40]):
            with self.assertRaisesRegex(ValueError, "exactly"):
                manage.require_main()


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.app = self.root / "app"
        self.stage = self.root / "stage"
        for directory in (self.app / "Core", self.stage / "Core"):
            directory.mkdir(parents=True)
        (self.app / "Core/agent.py").write_text("value = 'old'\n")
        (self.stage / "Core/agent.py").write_text("value = 'new'\n")
        (self.stage / "Core/new.py").write_text("value = 'added'\n")
        self.manifest = {"commit": "a"*40, "files": {
            name: deploy.digest(self.stage / name)
            for name in ("Core/agent.py", "Core/new.py")}}
        self.plan = deploy.snapshot(self.app, self.manifest, self.app / "backups/releases")

    def test_backup_and_restore_exact_bytes_and_absent_files(self):
        for name in self.manifest["files"]:
            deploy.atomic_copy(self.stage / name, self.app / name)
        deploy.write_json(self.app / "release.json", self.manifest)
        deploy.restore(self.app, self.plan)
        self.assertEqual((self.app / "Core/agent.py").read_text(), "value = 'old'\n")
        self.assertFalse((self.app / "Core/new.py").exists())
        self.assertFalse((self.app / "release.json").exists())

    def test_source_drift_prevents_activation(self):
        (self.app / "Core/agent.py").write_text("someone edited this")
        with self.assertRaisesRegex(ValueError, "changed"):
            deploy.verify_before(self.app, self.plan)

    def test_corrupted_backup_prevents_restore(self):
        (Path(self.plan["backup"]) / "Core/agent.py").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            deploy.restore(self.app, self.plan)

    def test_healthy_activation_records_exact_commit(self):
        with patch.object(deploy, "idle"), patch.object(deploy, "service") as service, \
             patch.object(deploy, "wait_ready") as wait:
            deploy.apply_release(self.app, self.stage, self.manifest, self.plan, "api.service", "http://localhost")
        self.assertEqual(deploy.read_json(self.app / "release.json")["commit"], "a"*40)
        self.assertEqual([c.args[0] for c in service.call_args_list], ["stop", "start"])
        self.assertEqual(wait.call_args.args[1], "a"*40)

    def test_corrupted_backup_prevents_service_stop(self):
        (Path(self.plan["backup"]) / "Core/agent.py").write_text("corrupt")
        with patch.object(deploy, "service") as service:
            with self.assertRaisesRegex(ValueError, "checksum"):
                deploy.apply_release(self.app, self.stage, self.manifest, self.plan, "api.service", "http://localhost")
        service.assert_not_called()

    def test_failed_readiness_restores_working_source(self):
        with patch.object(deploy, "idle"), patch.object(deploy, "service") as service, \
             patch.object(deploy, "wait_ready", side_effect=[RuntimeError("unhealthy"), None]):
            with self.assertRaisesRegex(RuntimeError, "restored"):
                deploy.apply_release(self.app, self.stage, self.manifest, self.plan, "api.service", "http://localhost")
        self.assertEqual((self.app / "Core/agent.py").read_text(), "value = 'old'\n")
        self.assertFalse((self.app / "Core/new.py").exists())
        self.assertEqual([c.args[0] for c in service.call_args_list],
                         ["stop", "start", "stop", "start"])

    def test_busy_service_prevents_stop(self):
        with patch.object(deploy, "idle", side_effect=ValueError("busy")), \
             patch.object(deploy, "service") as service:
            with self.assertRaisesRegex(ValueError, "busy"):
                deploy.apply_release(self.app, self.stage, self.manifest, self.plan, "api.service", "http://localhost")
        service.assert_not_called()

    def test_archive_rejects_traversal_symlinks_and_bad_checksums(self):
        for bad_name, kind in (("../outside", tarfile.REGTYPE), ("Core/link.py", tarfile.SYMTYPE)):
            archive = self.root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as target:
                info = tarfile.TarInfo(bad_name)
                info.type = kind
                info.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
                target.addfile(info)
            with self.assertRaises(ValueError):
                deploy.unpack(archive, self.root / "unpack", "a"*40)
        archive = self.root / "bad-hash.tar.gz"
        with tarfile.open(archive, "w:gz") as target:
            for name, content in {"Core/a.py": b"x=1", "release.json": json.dumps({
                "commit": "a"*40, "files": {"Core/a.py": "0"*64}}).encode()}.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                target.addfile(info, io.BytesIO(content))
        with self.assertRaisesRegex(ValueError, "checksum"):
            deploy.unpack(archive, self.root / "unpack", "a"*40)


if __name__ == "__main__":
    unittest.main()
