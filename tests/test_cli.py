import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

from evalbudget.cli import main


class CliTests(unittest.TestCase):
    def test_runs_external_judge_end_to_end(self):
        echo_command = shlex.join(
            [sys.executable, "-c", "import sys; print(sys.stdin.read(), end='')"]
        )
        judge_command = shlex.join(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; x=json.load(sys.stdin); "
                    "print(json.dumps({'score': float(x['output'].casefold() == x['expected'].casefold())}))"
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "judge.jsonl"
            report_path = Path(directory) / "report.json"
            dataset.write_text(
                json.dumps({"id": "1", "prompt": "Paris", "expected": "paris", "grader": "judge"}) + "\n"
            )
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        str(dataset),
                        "--baseline", echo_command,
                        "--candidate", echo_command,
                        "--judge-command", judge_command,
                        "--min-samples", "1",
                        "--max-samples", "1",
                        "--output", str(report_path),
                    ]
                )
            report = json.loads(report_path.read_text())
            self.assertEqual(exit_code, 0)
            self.assertEqual(report["observations"][0]["baseline_score"], 1.0)
            self.assertEqual(report["observations"][0]["candidate_score"], 1.0)

    def test_analyzes_pre_scored_dataset_without_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "scores.jsonl"
            dataset.write_text(
                "".join(
                    json.dumps({"id": str(index), "baseline_score": 0, "candidate_score": 1}) + "\n"
                    for index in range(25)
                ),
                encoding="utf-8",
            )
            with redirect_stdout(StringIO()):
                exit_code = main([str(dataset), "--pre-scored", "--min-samples", "5", "--max-samples", "25"])
            self.assertEqual(exit_code, 0)

    def test_end_to_end_with_every_grader_type(self):
        cases = [
            {"id": "exact", "prompt": '"object"', "expected": "object"},
            {"id": "accepted", "prompt": "Yes", "expected": ["yes", "y"], "grader": "accepted"},
            {"id": "numeric", "prompt": "3 minutes", "expected": 3, "grader": "numeric"},
            {"id": "regex", "prompt": "ticket-42", "expected": r"ticket-\d+", "grader": "regex"},
        ]
        echo_command = shlex.join(
            [sys.executable, "-c", "import sys; print(sys.stdin.read(), end='')"]
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "cases.jsonl"
            report_path = Path(directory) / "report.json"
            dataset.write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")

            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        str(dataset),
                        "--baseline",
                        echo_command,
                        "--candidate",
                        echo_command,
                        "--min-samples",
                        "1",
                        "--max-samples",
                        "4",
                        "--output",
                        str(report_path),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["samples_used"], 4)
            self.assertEqual(report["ties"], 4)
            self.assertTrue(
                all(item["baseline_score"] == item["candidate_score"] == 1 for item in report["observations"])
            )
            self.assertEqual(
                {item["grader_type"] for item in report["observations"]},
                {"exact", "accepted", "numeric", "regex"},
            )
            self.assertTrue(all("grader" in item for item in report["observations"]))


if __name__ == "__main__":
    unittest.main()
