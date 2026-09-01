from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class Observation:
    case_id: str
    baseline_score: float
    candidate_score: float
    baseline_output: str | None = None
    candidate_output: str | None = None
    expected: object | None = None
    grader_type: str | None = None
    grader: object | None = None
    category: str | None = None

    @property
    def difference(self) -> float:
        return self.candidate_score - self.baseline_score


@dataclass(frozen=True)
class EvaluationResult:
    decision: str
    stop_reason: str
    samples_used: int
    mean_difference: float
    confidence_lower: float
    confidence_upper: float
    candidate_wins: int
    baseline_wins: int
    ties: int
    confidence: float
    practical_effect: float
    observations: tuple[Observation, ...]
    category_summaries: dict[str, dict[str, int | float]]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["observations"] = [
            {**asdict(item), "difference": item.difference}
            for item in self.observations
        ]
        return data


def confidence_sequence(mean: float, sample_count: int, alpha: float) -> tuple[float, float]:
    """Return a confidence sequence for a mean of values bounded to [-1, 1].

    A summable alpha-spending schedule and Hoeffding's inequality make the
    interval simultaneously valid over every sample count, so inspecting it
    after each case does not inflate the requested error rate.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    alpha_at_n = alpha * 6.0 / (math.pi**2 * sample_count**2)
    radius = math.sqrt(2.0 * math.log(2.0 / alpha_at_n) / sample_count)
    return max(-1.0, mean - radius), min(1.0, mean + radius)


def evaluate_observations(
    observations: Iterable[Observation],
    *,
    confidence: float = 0.95,
    practical_effect: float = 0.0,
    min_samples: int = 20,
    max_samples: int = 500,
) -> EvaluationResult:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not 0 <= practical_effect < 1:
        raise ValueError("practical_effect must be in [0, 1)")
    if min_samples < 1 or max_samples < min_samples:
        raise ValueError("require 1 <= min_samples <= max_samples")

    used: list[Observation] = []
    total = 0.0
    lower, upper = -1.0, 1.0
    decision = "inconclusive"
    stop_reason = "available_cases_exhausted"

    for item in observations:
        if not 0 <= item.baseline_score <= 1 or not 0 <= item.candidate_score <= 1:
            raise ValueError(f"scores for case {item.case_id!r} must be in [0, 1]")
        used.append(item)
        total += item.difference
        sample_count = len(used)
        mean = total / sample_count
        lower, upper = confidence_sequence(mean, sample_count, 1.0 - confidence)

        if sample_count >= min_samples:
            if lower > practical_effect:
                decision, stop_reason = "candidate_better", "evidence_threshold_reached"
                break
            if upper < -practical_effect:
                decision, stop_reason = "baseline_better", "evidence_threshold_reached"
                break
            if practical_effect > 0 and lower >= -practical_effect and upper <= practical_effect:
                decision, stop_reason = "practically_equivalent", "equivalence_threshold_reached"
                break
        if sample_count >= max_samples:
            stop_reason = "max_samples_reached"
            break

    if not used:
        raise ValueError("at least one observation is required")

    wins = sum(item.difference > 0 for item in used)
    losses = sum(item.difference < 0 for item in used)
    categories: dict[str, list[Observation]] = {}
    for item in used:
        if item.category is not None:
            categories.setdefault(item.category, []).append(item)
    category_summaries = {
        name: {
            "samples": len(items),
            "mean_difference": sum(item.difference for item in items) / len(items),
            "candidate_wins": sum(item.difference > 0 for item in items),
            "baseline_wins": sum(item.difference < 0 for item in items),
            "ties": sum(item.difference == 0 for item in items),
        }
        for name, items in sorted(categories.items())
    }
    return EvaluationResult(
        decision=decision,
        stop_reason=stop_reason,
        samples_used=len(used),
        mean_difference=total / len(used),
        confidence_lower=lower,
        confidence_upper=upper,
        candidate_wins=wins,
        baseline_wins=losses,
        ties=len(used) - wins - losses,
        confidence=confidence,
        practical_effect=practical_effect,
        observations=tuple(used),
        category_summaries=category_summaries,
    )
