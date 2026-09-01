from __future__ import annotations

import json
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import shlex
import subprocess
import time
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


def run_command(
    command: str,
    prompt: str,
    timeout: float,
    *,
    retries: int = 0,
    retry_delay: float = 0.0,
) -> str:
    arguments = shlex.split(command)
    if not arguments:
        raise ValueError("model command cannot be empty")
    if retries < 0 or retry_delay < 0:
        raise ValueError("retries and retry_delay must be non-negative")
    last_error: RuntimeError | None = None
    for attempt in range(retries + 1):
        try:
            completed = subprocess.run(
                arguments,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            if completed.returncode == 0:
                return completed.stdout.strip()
            detail = completed.stderr.strip() or f"exit status {completed.returncode}"
            last_error = RuntimeError(f"command failed ({command}): {detail}")
        except subprocess.TimeoutExpired:
            last_error = RuntimeError(f"command timed out after {timeout:g}s: {command}")
        except OSError as error:
            last_error = RuntimeError(f"could not run {arguments[0]!r}: {error}")
        if attempt < retries and retry_delay:
            time.sleep(retry_delay)
    assert last_error is not None
    raise last_error


def exact_match(output: str, expected: str) -> float:
    """Backward-compatible entry point for the default exact grader."""
    return grade_output({"expected": expected, "grader": "exact"}, output)


def parse_command_output(raw_output: str, output_format: str) -> tuple[str, float | None]:
    if output_format == "text":
        return raw_output, None
    if output_format != "json":
        raise ValueError("command output format must be text or json")
    try:
        envelope = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"command returned invalid JSON: {error.msg}") from error
    if not isinstance(envelope, dict) or not isinstance(envelope.get("output"), str):
        raise RuntimeError("command JSON must contain a string output field")
    cost = envelope.get("cost_usd")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0 or not math.isfinite(cost):
        raise RuntimeError("command JSON must contain a finite non-negative cost_usd")
    return envelope["output"], float(cost)


def collect_observations(
    cases: Iterable[dict[str, Any]],
    baseline_command: str,
    candidate_command: str,
    *,
    timeout: float,
    retries: int = 0,
    retry_delay: float = 0.0,
    cache_path: Path | None = None,
    command_output: str = "text",
) -> Iterable[Observation]:
    cache = _load_observation_cache(cache_path) if cache_path else {}
    for case in cases:
        cache_key = _observation_cache_key(
            case, baseline_command, candidate_command, command_output
        )
        if cache_key in cache:
            yield cache[cache_key]
            continue
        baseline_raw = run_command(
            baseline_command, case["prompt"], timeout, retries=retries, retry_delay=retry_delay
        )
        candidate_raw = run_command(
            candidate_command, case["prompt"], timeout, retries=retries, retry_delay=retry_delay
        )
        baseline_output, baseline_cost = parse_command_output(baseline_raw, command_output)
        candidate_output, candidate_cost = parse_command_output(candidate_raw, command_output)
        observation = Observation(
            case_id=case["id"],
            baseline_score=grade_output(case, baseline_output),
            candidate_score=grade_output(case, candidate_output),
            baseline_output=baseline_output,
            candidate_output=candidate_output,
            expected=case["expected"],
            grader_type=case["grader"]["type"],
            grader=case["grader"],
            category=case.get("category"),
            baseline_cost_usd=baseline_cost,
            candidate_cost_usd=candidate_cost,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": cache_key, "observation": asdict(observation)}) + "\n")
        yield observation


def _observation_cache_key(
    case: dict[str, Any],
    baseline_command: str,
    candidate_command: str,
    command_output: str,
) -> str:
    payload = json.dumps(
        {
            "case": case,
            "baseline": baseline_command,
            "candidate": candidate_command,
            "command_output": command_output,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_observation_cache(path: Path) -> dict[str, Observation]:
    if not path.exists():
        return {}
    cache: dict[str, Observation] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cache[item["key"]] = Observation(**item["observation"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid observation cache entry") from error
    return cache
