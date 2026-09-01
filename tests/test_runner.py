import json
from pathlib import Path
import tempfile
import unittest
import subprocess

from unittest.mock import patch

from evalbudget.runner import (
    collect_observations,
    collect_scored_observations,
    exact_match,
    load_cases,
    load_scored_cases,
    run_command,
)


class RunnerTests(unittest.TestCase):
    def test_exact_match_normalizes_case_and_whitespace(self):
        self.assertEqual(exact_match(" Yes \n", "yes"), 1.0)

    def test_load_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps({"id": "1", "prompt": "p", "expected": "e"}) + "\n")
            self.assertEqual(load_cases(path)[0]["prompt"], "p")

    def test_load_cases_validates_and_preserves_grader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                json.dumps({"id": "1", "prompt": "p", "expected": 3, "grader": "numeric"}) + "\n"
            )
            case = load_cases(path)[0]
            self.assertEqual(case["expected"], 3)
            self.assertEqual(case["grader"]["type"], "numeric")

    def test_load_cases_rejects_duplicate_ids_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            item = json.dumps({"id": "same", "prompt": "p", "expected": "e"})
            path.write_text(item + "\n" + item + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                load_cases(path)

    def test_load_cases_validates_category(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps({"id": "1", "prompt": "p", "expected": "e", "category": "  math "}) + "\n")
            self.assertEqual(load_cases(path)[0]["category"], "math")

    def test_collect_observations_uses_case_grader(self):
        cases = [{"id": "1", "prompt": "p", "expected": 3, "grader": {"type": "numeric"}}]
        with patch("evalbudget.runner.run_command", side_effect=["3 minutes", "The answer is 3."]):
            observation = next(iter(collect_observations(cases, "base", "candidate", timeout=1)))
        self.assertEqual(observation.baseline_score, 1.0)
        self.assertEqual(observation.candidate_score, 1.0)
        self.assertEqual(observation.grader_type, "numeric")
        self.assertEqual(observation.grader, {"type": "numeric"})

    def test_collect_observations_resumes_from_cache(self):
        cases = [{"id": "1", "prompt": "p", "expected": "yes", "grader": {"type": "exact"}}]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.jsonl"
            with patch("evalbudget.runner.run_command", side_effect=["no", "yes"]) as run:
                first = list(collect_observations(cases, "base", "candidate", timeout=1, cache_path=cache))
            with patch("evalbudget.runner.run_command") as run_again:
                second = list(collect_observations(cases, "base", "candidate", timeout=1, cache_path=cache))
            self.assertEqual(run.call_count, 2)
            run_again.assert_not_called()
            self.assertEqual(first, second)

    def test_cache_key_changes_with_command(self):
        cases = [{"id": "1", "prompt": "p", "expected": "yes", "grader": {"type": "exact"}}]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.jsonl"
            with patch("evalbudget.runner.run_command", side_effect=["no", "yes"]):
                list(collect_observations(cases, "base-v1", "candidate", timeout=1, cache_path=cache))
            with patch("evalbudget.runner.run_command", side_effect=["yes", "yes"]) as changed:
                list(collect_observations(cases, "base-v2", "candidate", timeout=1, cache_path=cache))
            self.assertEqual(changed.call_count, 2)

    def test_run_command_passes_prompt_on_stdin(self):
        output = run_command("python3 -c 'import sys; print(sys.stdin.read().upper())'", "hello", 2)
        self.assertEqual(output, "HELLO")

    @patch("evalbudget.runner.subprocess.run")
    def test_run_command_retries_then_returns_output(self, run):
        run.side_effect = [
            subprocess.TimeoutExpired("model", 1),
            subprocess.CompletedProcess(["model"], 0, stdout="recovered\n", stderr=""),
        ]
        output = run_command("model", "prompt", 1, retries=1)
        self.assertEqual(output, "recovered")
        self.assertEqual(run.call_count, 2)

    @patch("evalbudget.runner.subprocess.run")
    def test_run_command_stops_after_retry_limit(self, run):
        run.return_value = subprocess.CompletedProcess(["model"], 2, stdout="", stderr="bad")
        with self.assertRaisesRegex(RuntimeError, "bad"):
            run_command("model", "prompt", 1, retries=2)
        self.assertEqual(run.call_count, 3)

    def test_loads_pre_scored_cases_without_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text(
                json.dumps({"id": "1", "baseline_score": 0.25, "candidate_score": 1, "category": "math"}) + "\n"
            )
            cases = load_scored_cases(path)
            observation = next(iter(collect_scored_observations(cases)))
            self.assertEqual(observation.difference, 0.75)
            self.assertEqual(observation.category, "math")

    def test_rejects_invalid_pre_scored_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            path.write_text(json.dumps({"id": "1", "baseline_score": -1, "candidate_score": 1}) + "\n")
            with self.assertRaisesRegex(ValueError, "baseline_score"):
                load_scored_cases(path)


if __name__ == "__main__":
    unittest.main()
