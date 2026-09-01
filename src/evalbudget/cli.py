from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

from .engine import evaluate_observations
from .runner import collect_observations, collect_scored_observations, load_cases, load_scored_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalbudget",
        description="Compare two model commands and stop when the evidence is sufficient.",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="JSONL with id, prompt, expected, and an optional grader",
    )
    parser.add_argument("--baseline", help="baseline command; receives the prompt on stdin")
    parser.add_argument("--candidate", help="candidate command; receives the prompt on stdin")
    parser.add_argument(
        "--pre-scored",
        action="store_true",
        help="analyze existing baseline_score and candidate_score fields without running commands",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--practical-effect", type=float, default=0.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per model invocation")
    parser.add_argument("--retries", type=int, default=0, help="retries after a failed model invocation")
    parser.add_argument("--retry-delay", type=float, default=0.0, help="seconds between retries")
    parser.add_argument("--seed", type=int, default=0, help="dataset shuffle seed")
    parser.add_argument("--output", type=Path, help="write the full JSON report here")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.pre_scored:
            cases = load_scored_cases(args.dataset)
            observations = collect_scored_observations(cases)
        else:
            if not args.baseline or not args.candidate:
                raise ValueError("--baseline and --candidate are required unless --pre-scored is used")
            cases = load_cases(args.dataset)
            observations = collect_observations(
                cases,
                args.baseline,
                args.candidate,
                timeout=args.timeout,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
        random.Random(args.seed).shuffle(cases)
        result = evaluate_observations(
            observations,
            confidence=args.confidence,
            practical_effect=args.practical_effect,
            min_samples=args.min_samples,
            max_samples=args.max_samples,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"evalbudget: error: {error}", file=sys.stderr)
        return 2

    report = result.to_dict()
    report["dataset"] = str(args.dataset)
    report["seed"] = args.seed
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            print(f"evalbudget: error: could not write report: {error}", file=sys.stderr)
            return 2

    summary = {key: value for key, value in report.items() if key != "observations"}
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Decision: {result.decision}")
        print(f"Stopped:  {result.stop_reason} after {result.samples_used} samples")
        print(
            f"Effect:   {result.mean_difference:+.3f} "
            f"({result.confidence:.0%} anytime-valid CS "
            f"[{result.confidence_lower:+.3f}, {result.confidence_upper:+.3f}])"
        )
        print(
            f"Outcomes: candidate {result.candidate_wins}, "
            f"baseline {result.baseline_wins}, ties {result.ties}"
        )
        if args.output:
            print(f"Report:   {args.output}")
    return 0
