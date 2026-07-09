from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

import app.corpus_sync as corpus_sync
from app.corpus_sync import CorpusSource, CorpusSourceError, load_sources, sync_all

VALID_ENTRY = {
    "host": "dev-laptop.example.com",
    "port": 22,
    "user": "jketreno",
    "remote_corpus_root": "~/.config/clare/corpus",
    "host_key": "dev-laptop.example.com ssh-ed25519 AAAAtest",
}


def _write_sources(path: pathlib.Path, entries: list[dict]) -> None:
    path.write_text(yaml.safe_dump({"sources": entries}), encoding="utf-8")


class LoadSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temp.name) / "corpus_sources.yml"

    def tearDown(self):
        self.temp.cleanup()

    def test_loads_valid_entry(self):
        _write_sources(self.path, [VALID_ENTRY])
        sources = load_sources(self.path)
        self.assertEqual(sources, [CorpusSource(**VALID_ENTRY)])

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_sources(self.path), [])

    def test_rejects_path_traversal_in_remote_root(self):
        entry = {**VALID_ENTRY, "remote_corpus_root": "../../etc"}
        _write_sources(self.path, [entry])
        with self.assertRaises(CorpusSourceError):
            load_sources(self.path)

    def test_rejects_missing_host_key(self):
        entry = {k: v for k, v in VALID_ENTRY.items() if k != "host_key"}
        _write_sources(self.path, [entry])
        with self.assertRaises(CorpusSourceError):
            load_sources(self.path)

    def test_rejects_blank_host_key(self):
        entry = {**VALID_ENTRY, "host_key": "   "}
        _write_sources(self.path, [entry])
        with self.assertRaises(CorpusSourceError):
            load_sources(self.path)

    def test_rejects_invalid_hostname(self):
        entry = {**VALID_ENTRY, "host": "host; rm -rf /"}
        _write_sources(self.path, [entry])
        with self.assertRaises(CorpusSourceError):
            load_sources(self.path)

    def test_rejects_invalid_port(self):
        entry = {**VALID_ENTRY, "port": 999999}
        _write_sources(self.path, [entry])
        with self.assertRaises(CorpusSourceError):
            load_sources(self.path)


class SyncAllTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.sources_path = self.root / "corpus_sources.yml"
        self.corpus_root = self.root / "corpus"
        self.key_path = self.root / "sync_key"
        self.key_path.write_text("fake-key", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_no_sources_is_a_noop(self):
        with patch.object(corpus_sync, "SOURCES_PATH", self.sources_path), patch.object(
            corpus_sync, "CORPUS_ROOT", self.corpus_root
        ):
            result = sync_all()
        self.assertEqual(result, {"hosts": 0, "succeeded": 0, "failed": 0})

    def test_one_host_failure_does_not_block_others_or_raise(self):
        second_entry = {**VALID_ENTRY, "host": "second-host.example.com"}
        _write_sources(self.sources_path, [VALID_ENTRY, second_entry])

        def fake_run(command, **kwargs):
            if "dev-laptop.example.com" in command[-2]:
                return subprocess.CompletedProcess(command, returncode=23, stdout="", stderr="rsync error")
            return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

        with patch.object(corpus_sync, "SOURCES_PATH", self.sources_path), patch.object(
            corpus_sync, "CORPUS_ROOT", self.corpus_root
        ), patch.object(corpus_sync, "SSH_KEY_PATH", self.key_path), patch(
            "app.corpus_sync.subprocess.run", side_effect=fake_run
        ):
            result = sync_all()

        self.assertEqual(result["hosts"], 2)
        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["results"]["second-host.example.com"], "succeeded")
        self.assertEqual(result["results"]["dev-laptop.example.com"], "failed")

    def test_missing_ssh_key_fails_closed_without_raising(self):
        _write_sources(self.sources_path, [VALID_ENTRY])
        missing_key = self.root / "does-not-exist"
        with patch.object(corpus_sync, "SOURCES_PATH", self.sources_path), patch.object(
            corpus_sync, "CORPUS_ROOT", self.corpus_root
        ), patch.object(corpus_sync, "SSH_KEY_PATH", missing_key):
            result = sync_all()
        self.assertEqual(result["succeeded"], 0)
        self.assertEqual(result["failed"], 1)

    def test_rsync_command_pins_known_hosts_and_strict_checking(self):
        _write_sources(self.sources_path, [VALID_ENTRY])
        captured_commands = []

        def fake_run(command, **kwargs):
            captured_commands.append(command)
            return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

        with patch.object(corpus_sync, "SOURCES_PATH", self.sources_path), patch.object(
            corpus_sync, "CORPUS_ROOT", self.corpus_root
        ), patch.object(corpus_sync, "SSH_KEY_PATH", self.key_path), patch(
            "app.corpus_sync.subprocess.run", side_effect=fake_run
        ):
            sync_all()

        self.assertEqual(len(captured_commands), 1)
        ssh_option = next(part for part in captured_commands[0] if part.startswith("ssh "))
        self.assertIn("StrictHostKeyChecking=yes", ssh_option)
        self.assertIn("UserKnownHostsFile=", ssh_option)
        self.assertNotIn("StrictHostKeyChecking=no", ssh_option)
        self.assertNotIn("/dev/null", ssh_option)

    def test_rsync_remote_path_is_relative_to_rrsync_fixed_root(self):
        # rrsync's forced authorized_keys command fixes the remote root to
        # <remote_corpus_root>/sessions itself; repeating that absolute path
        # as the rsync remote argument would double-join and fail (verified
        # against a real rrsync installation). The remote arg must be relative.
        _write_sources(self.sources_path, [VALID_ENTRY])
        captured_commands = []

        def fake_run(command, **kwargs):
            captured_commands.append(command)
            return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

        with patch.object(corpus_sync, "SOURCES_PATH", self.sources_path), patch.object(
            corpus_sync, "CORPUS_ROOT", self.corpus_root
        ), patch.object(corpus_sync, "SSH_KEY_PATH", self.key_path), patch(
            "app.corpus_sync.subprocess.run", side_effect=fake_run
        ):
            sync_all()

        remote_arg = captured_commands[0][-2]
        self.assertEqual(remote_arg, "jketreno@dev-laptop.example.com:./")
        self.assertNotIn("/sessions", remote_arg)

    def test_writes_status_summary(self):
        _write_sources(self.sources_path, [VALID_ENTRY])

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

        with patch.object(corpus_sync, "SOURCES_PATH", self.sources_path), patch.object(
            corpus_sync, "CORPUS_ROOT", self.corpus_root
        ), patch.object(corpus_sync, "SSH_KEY_PATH", self.key_path), patch(
            "app.corpus_sync.subprocess.run", side_effect=fake_run
        ):
            sync_all()

        status_path = self.corpus_root / "meta" / "corpus_sync_status.json"
        self.assertTrue(status_path.exists())


if __name__ == "__main__":
    unittest.main()
