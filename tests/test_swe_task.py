import json
import os
import tempfile
import unittest
from pathlib import Path

from tasks.swe import TaskSWE


class TaskSWETests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "sample_id": "org__repo-1::reactivate",
            "repo": "org/repo",
            "base_commit": "abc123",
            "official_problem_statement": "Fix the active behavior.",
            "shared_history": [
                {"role": "user", "content": "Implement A."},
                {"role": "assistant", "content": "A is active."},
                {"role": "user", "content": "Park B unless I say activate B."},
            ],
            "future_request": "activate B",
        }

    def test_loads_jsonl_and_preserves_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
            old = os.environ.get("LOST_CONVERSATION_SWE_DATASET")
            os.environ["LOST_CONVERSATION_SWE_DATASET"] = str(path)
            try:
                sample = TaskSWE().get_samples()[0]
            finally:
                if old is None:
                    os.environ.pop("LOST_CONVERSATION_SWE_DATASET", None)
                else:
                    os.environ["LOST_CONVERSATION_SWE_DATASET"] = old
        self.assertEqual(sample["task"], "swe")
        self.assertIn("[ASSISTANT 2]", TaskSWE._transcript(sample))
        self.assertEqual(sample["shards"][-1]["shard"], "activate B")

    def test_extracts_diff_without_claiming_correctness(self):
        task = TaskSWE()
        patch = task.extract_answer("```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n```")
        result = task.evaluator_function(patch, self.row)
        self.assertTrue(result["valid_patch"])
        self.assertIsNone(result["is_correct"])
        self.assertTrue(result["requires_external_verifier"])


if __name__ == "__main__":
    unittest.main()
