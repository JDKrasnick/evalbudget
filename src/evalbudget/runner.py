from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Iterable

from .engine import Observation
from .graders import grade_output, validate_case


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    id_lines: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: each case must be a JSON object")
            try:
                case = validate_case(item)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if case["id"] in id_lines:
                raise ValueError(
                    f"{path}:{line_number}: duplicate id {case['id']!r} "
                    f"(first used on line {id_lines[case['id']]})"
                )
            id_lines[case["id"]] = line_number
            cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def load_scored_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: each case must be a JSON object")
            missing = {"id", "baseline_score", "candidate_score"} - item.keys()
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {', '.join(sorted(missing))}")
            if not isinstance(item["id"], str):
                raise ValueError(f"{path}:{line_number}: id must be a string")
            if item["id"] in ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {item['id']!r}")
            for name in ("baseline_score", "candidate_score"):
                value = item[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise ValueError(f"{path}:{line_number}: {name} must be a number in [0, 1]")
            if "category" in item and (
                not isinstance(item["category"], str) or not item["category"].strip()
            ):
                raise ValueError(f"{path}:{line_number}: category must be a non-empty string")
            ids.add(item["id"])
            cases.append(item)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def collect_scored_observations(cases: Iterable[dict[str, Any]]) -> Iterable[Observation]:
    for case in cases:
        yield Observation(
            case_id=case["id"],
            baseline_score=float(case["baseline_score"]),
            candidate_score=float(case["candidate_score"]),
            grader_type="pre_scored",
            category=case.get("category", "").strip() or None,
        )


def run_command(command: str, prompt: str, timeout: float) -> str:
    arguments = shlex.split(command)
    if not arguments:
        raise ValueError("model command cannot be empty")
    try:
        completed = subprocess.run(
            arguments,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"command timed out after {timeout:g}s: {command}") from error
    except OSError as error:
        raise RuntimeError(f"could not run {arguments[0]!r}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise RuntimeError(f"command failed ({command}): {detail}")
    return completed.stdout.strip()


def exact_match(output: str, expected: str) -> float:
    """Backward-compatible entry point for the default exact grader."""
    return grade_output({"expected": expected, "grader": "exact"}, output)


def collect_observations(
    cases: Iterable[dict[str, Any]],
    baseline_command: str,
    candidate_command: str,
    *,
    timeout: float,
) -> Iterable[Observation]:
    for case in cases:
        baseline_output = run_command(baseline_command, case["prompt"], timeout)
        candidate_output = run_command(candidate_command, case["prompt"], timeout)
        yield Observation(
            case_id=case["id"],
            baseline_score=grade_output(case, baseline_output),
            candidate_score=grade_output(case, candidate_output),
            baseline_output=baseline_output,
            candidate_output=candidate_output,
            expected=case["expected"],
            grader_type=case["grader"]["type"],
            grader=case["grader"],
            category=case.get("category"),
        )
