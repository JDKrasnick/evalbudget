"""Adaptive evaluation with anytime-valid confidence sequences."""

from .engine import EvaluationResult, Observation, evaluate_observations
from .graders import grade_output, validate_case

__all__ = ["EvaluationResult", "Observation", "evaluate_observations", "grade_output", "validate_case"]
__version__ = "0.2.0"
