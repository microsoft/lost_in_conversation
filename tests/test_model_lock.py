import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-only-placeholder")
from model_openai import validate_model_lock


class ModelLockTests(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.get("LOST_CONVERSATION_ALLOWED_MODEL")
        os.environ["LOST_CONVERSATION_ALLOWED_MODEL"] = "gpt-5.4-mini"

    def tearDown(self):
        if self.old is None:
            os.environ.pop("LOST_CONVERSATION_ALLOWED_MODEL", None)
        else:
            os.environ["LOST_CONVERSATION_ALLOWED_MODEL"] = self.old

    def test_accepts_matching_request_and_served_model(self):
        validate_model_lock("gpt-5.4-mini", "gpt-5.4-mini")

    def test_rejects_request_model_mismatch(self):
        with self.assertRaises(ValueError):
            validate_model_lock("gpt-4o-mini")

    def test_rejects_served_model_mismatch(self):
        with self.assertRaises(RuntimeError):
            validate_model_lock("gpt-5.4-mini", "fallback-model")


if __name__ == "__main__":
    unittest.main()
