"""Offline tests for SQLite conversation retention and isolation."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from Core.conversation_store import ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "private" / "conversation.sqlite3"
        self.store = ConversationStore(self.path, max_turns=4, ttl_seconds=60)

    def tearDown(self):
        self.temp.cleanup()

    def test_sessions_are_isolated(self):
        self.store.save("forge-session-0001", [{"query": "Horus", "response": "A"}])
        self.store.save("forge-session-0002", [{"query": "Loken", "response": "B"}])
        self.assertEqual(self.store.get("forge-session-0001")[0]["response"], "A")
        self.assertEqual(self.store.get("forge-session-0002")[0]["response"], "B")

    def test_persists_across_store_instances(self):
        self.store.save("forge-session-0001", [{"query": "Horus", "response": "A"}])
        reopened = ConversationStore(self.path, ttl_seconds=60)
        self.assertEqual(reopened.get("forge-session-0001")[0]["query"], "Horus")

    def test_retains_only_four_completed_turns(self):
        turns = [{"query": str(i), "response": str(i)} for i in range(7)]
        self.store.save("forge-session-0001", turns)
        self.assertEqual([t["query"] for t in self.store.get("forge-session-0001")],
                         ["3", "4", "5", "6"])

    def test_clear_is_scoped_to_one_session(self):
        self.store.save("forge-session-0001", [{"query": "one", "response": "A"}])
        self.store.save("forge-session-0002", [{"query": "two", "response": "B"}])
        self.store.clear("forge-session-0001")
        self.assertEqual(self.store.get("forge-session-0001"), [])
        self.assertEqual(len(self.store.get("forge-session-0002")), 1)

    def test_expiry_removes_old_turns(self):
        with patch("Core.conversation_store.time.time", return_value=1000):
            self.store.save("forge-session-0001", [{"query": "A", "response": "B"}])
        with patch("Core.conversation_store.time.time", return_value=1062):
            self.assertEqual(self.store.get("forge-session-0001"), [])

    def test_default_and_invalid_ids_rejected(self):
        for session_id in ("default", "", "a" * 129, "unsafe/id", "bad id"):
            with self.subTest(session_id=session_id):
                with self.assertRaises(ValueError):
                    self.store.get(session_id)

    def test_save_empty_clears(self):
        self.store.save("forge-session-0001", [{"query": "hello", "response": "world"}])
        self.store.save("forge-session-0001", [])
        self.assertEqual(self.store.get("forge-session-0001"), [])


if __name__ == "__main__":
    unittest.main()
