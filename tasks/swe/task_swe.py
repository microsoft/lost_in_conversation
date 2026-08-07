"""SWE-Bench task adapter for evolving-intent conversations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from task_base import Task


DEFAULT_DATASET = "data/swe/evolving_swe_v1.jsonl"


class TaskSWE(Task):
    """Load public SWE tasks without exposing private verifier labels."""

    def __init__(self, version: str = "v1"):
        self.version = version
        self.task_name = "swe"
        self.answer_description = self.get_answer_description()
        self.answer_extraction_strategy = "task_specific"

    def get_task_name(self) -> str:
        return "swe"

    def get_dataset_file(self) -> str:
        return os.environ.get("LOST_CONVERSATION_SWE_DATASET", DEFAULT_DATASET)

    def get_samples(self, version: str | None = None, filter: str | None = None) -> List[Dict[str, Any]]:
        path = Path(self.get_dataset_file())
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        samples = [self._normalize(row) for row in rows]
        if filter:
            samples = [row for row in samples if filter in row["task_id"]]
        return samples

    def _normalize(self, row: Dict[str, Any]) -> Dict[str, Any]:
        sample = dict(row)
        sample["task"] = "swe"
        sample["task_id"] = row["sample_id"]
        user_turns = [turn["content"] for turn in row["shared_history"] if turn["role"] == "user"]
        user_turns.append(row["future_request"])
        sample["shards"] = [
            {"shard_id": index, "shard": text}
            for index, text in enumerate(user_turns, start=1)
        ]
        return sample

    def get_answer_description(self) -> str:
        return (
            "A source-only unified git diff beginning with 'diff --git'. "
            "It must not modify tests, dependencies, generated files, or documentation."
        )

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return (
            "You are implementing a change in a real repository at the supplied base commit. "
            "Track requirement lifecycle across the conversation: dormant decisions remain "
            "inactive unless their exact activation condition is met. Preserve the active "
            "issue contract. Return only a source-only unified git diff beginning with "
            "'diff --git'; do not modify tests, dependencies, generated files, or documentation."
        )

    @staticmethod
    def _transcript(sample: Dict[str, Any]) -> str:
        turns = "\n\n".join(
            f"[{turn['role'].upper()} {index}]\n{turn['content']}"
            for index, turn in enumerate(sample["shared_history"], start=1)
        )
        return f"{turns}\n\n[USER FUTURE]\n{sample['future_request']}"

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return (
            f"REPOSITORY: {sample['repo']}\nBASE COMMIT: {sample['base_commit']}\n\n"
            f"OFFICIAL ACTIVE ISSUE:\n{sample['official_problem_statement']}\n\n"
            f"CONVERSATION:\n{self._transcript(sample)}"
        )

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        requirements = "\n".join(f"- {shard['shard']}" for shard in sample["shards"])
        return (
            f"REPOSITORY: {sample['repo']}\nBASE COMMIT: {sample['base_commit']}\n\n"
            f"OFFICIAL ACTIVE ISSUE:\n{sample['official_problem_statement']}\n\n"
            f"CONCATENATED USER REQUIREMENTS:\n{requirements}"
        )

    def extract_answer(self, text: str) -> str:
        start = text.find("diff --git ")
        if start < 0:
            return ""
        patch = text[start:].strip()
        if patch.endswith("```"):
            patch = patch[:-3].rstrip()
        return patch + "\n"

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        valid_patch = extracted_answer.startswith("diff --git ")
        return {
            "is_correct": None,
            "score": None,
            "valid_patch": valid_patch,
            "requires_external_verifier": True,
        }

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": sample["task_id"],
            "repo": sample["repo"],
            "base_commit": sample["base_commit"],
            "prompt": self.populate_fully_specific_prompt(sample),
        }
