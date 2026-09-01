import json
from pathlib import Path
import tempfile
import unittest

from unittest.mock import patch

from evalbudget.runner import collect_observations, exact_match, load_cases, run_command


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

    def test_collect_observations_uses_case_grader(self):
        cases = [{"id": "1", "prompt": "p", "expected": 3, "grader": {"type": "numeric"}}]
        with patch("evalbudget.runner.run_command", side_effect=["3 minutes", "The answer is 3."]):
            observation = next(iter(collect_observations(cases, "base", "candidate", timeout=1)))
        self.assertEqual(observation.baseline_score, 1.0)
        self.assertEqual(observation.candidate_score, 1.0)
        self.assertEqual(observation.grader_type, "numeric")
        self.assertEqual(observation.grader, {"type": "numeric"})

    def test_run_command_passes_prompt_on_stdin(self):
        output = run_command("python3 -c 'import sys; print(sys.stdin.read().upper())'", "hello", 2)
        self.assertEqual(output, "HELLO")


if __name__ == "__main__":
    unittest.main()
