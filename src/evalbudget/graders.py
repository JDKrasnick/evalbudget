from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
import math
import re
import unicodedata
from typing import Any, Mapping


GRADER_TYPES = {"exact", "accepted", "numeric", "regex"}
_NUMBER = re.compile(
    r"(?<![\w.])([+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?\d+(?:,\d{3})*)?)(?!\w)(?!\.\d)"
)
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’"}


def grader_config(case: Mapping[str, Any]) -> dict[str, Any]:
    raw = case.get("grader", "exact")
    if isinstance(raw, str):
        config: dict[str, Any] = {"type": raw}
    elif isinstance(raw, dict):
        config = dict(raw)
    else:
        raise ValueError("grader must be a string or object")

    grader_type = config.get("type")
    if grader_type not in GRADER_TYPES:
        choices = ", ".join(sorted(GRADER_TYPES))
        raise ValueError(f"grader type must be one of: {choices}")

    allowed = {
        "exact": {"type"},
        "accepted": {"type"},
        "numeric": {"type", "abs_tol", "rel_tol"},
        "regex": {"type", "flags", "fullmatch"},
    }[grader_type]
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unsupported {grader_type} grader option: {', '.join(sorted(unknown))}")

    if grader_type == "numeric":
        for name, default in (("abs_tol", 1e-9), ("rel_tol", 0.0)):
            value = config.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"numeric grader {name} must be a number")
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"numeric grader {name} must be finite and non-negative")
            config[name] = value

    if grader_type == "regex":
        flags = config.get("flags", "")
        if not isinstance(flags, str) or any(flag not in "ims" for flag in flags.casefold()):
            raise ValueError("regex grader flags may contain only i, m, and s")
        if not isinstance(config.get("fullmatch", True), bool):
            raise ValueError("regex grader fullmatch must be a boolean")
        config["flags"] = flags.casefold()
        config["fullmatch"] = config.get("fullmatch", True)
    return config


def validate_case(case: Mapping[str, Any]) -> dict[str, Any]:
    missing = {"id", "prompt", "expected"} - case.keys()
    if missing:
        raise ValueError(f"missing {', '.join(sorted(missing))}")
    if not isinstance(case["id"], str) or not isinstance(case["prompt"], str):
        raise ValueError("id and prompt must be strings")
    if "category" in case and (
        not isinstance(case["category"], str) or not case["category"].strip()
    ):
        raise ValueError("category must be a non-empty string")

    config = grader_config(case)
    expected = case["expected"]
    grader_type = config["type"]
    if grader_type in {"exact", "regex"} and not isinstance(expected, str):
        raise ValueError(f"{grader_type} grader expected must be a string")
    if grader_type == "accepted":
        if not isinstance(expected, list) or not expected or not all(isinstance(item, str) for item in expected):
            raise ValueError("accepted grader expected must be a non-empty list of strings")
    if grader_type == "numeric":
        if isinstance(expected, bool) or not isinstance(expected, (int, float, str)):
            raise ValueError("numeric grader expected must be a number or numeric string")
        _parse_expected_number(expected)
    if grader_type == "regex":
        try:
            re.compile(expected, _regex_flags(config["flags"]))
        except re.error as error:
            raise ValueError(f"invalid expected regex: {error}") from error

    validated = dict(case)
    if "category" in validated:
        validated["category"] = validated["category"].strip()
    validated["grader"] = config
    return validated


def normalize_exact(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    fenced = re.fullmatch(r"```(?:[\w.+-]+)?\s*\n?(.*?)\n?```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1).strip()
    if len(value) >= 2 and value[0] in _QUOTE_PAIRS and value[-1] == _QUOTE_PAIRS[value[0]]:
        value = value[1:-1].strip()
    return " ".join(value.split()).casefold()


def grade_output(case: Mapping[str, Any], output: str) -> float:
    config = grader_config(case)
    grader_type = config["type"]
    expected = case["expected"]

    if grader_type == "exact":
        return float(normalize_exact(output) == normalize_exact(expected))
    if grader_type == "accepted":
        return float(any(normalize_exact(output) == normalize_exact(item) for item in expected))
    if grader_type == "numeric":
        actual = _parse_output_number(output)
        if actual is None:
            return 0.0
        target = _parse_expected_number(expected)
        difference = abs(actual - target)
        tolerance = max(
            Decimal(str(config["abs_tol"])),
            Decimal(str(config["rel_tol"])) * abs(target),
        )
        return float(difference <= tolerance)
    if grader_type == "regex":
        pattern = re.compile(expected, _regex_flags(config["flags"]))
        matcher = pattern.fullmatch if config["fullmatch"] else pattern.search
        return float(matcher(output.strip()) is not None)
    raise AssertionError(f"unreachable grader type: {grader_type}")


def _regex_flags(flags: str) -> int:
    result = 0
    for flag, value in (("i", re.IGNORECASE), ("m", re.MULTILINE), ("s", re.DOTALL)):
        if flag in flags:
            result |= value
    return result


def _parse_expected_number(value: object) -> Decimal:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric grader expected must be finite")
        return Decimal(str(value))
    if isinstance(value, str):
        parsed = _parse_numeric_token(value.strip())
        if parsed is not None and _NUMBER.fullmatch(value.strip()):
            return parsed
    raise ValueError("numeric grader expected must contain exactly one valid number")


def _parse_output_number(value: str) -> Decimal | None:
    tokens = [match.group(1) for match in _NUMBER.finditer(unicodedata.normalize("NFKC", value))]
    parsed = [_parse_numeric_token(token) for token in tokens]
    numbers = [number for number in parsed if number is not None]
    return numbers[0] if len(numbers) == 1 else None


def _parse_numeric_token(token: str) -> Decimal | None:
    token = token.replace(",", "").replace(" ", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            with localcontext() as context:
                context.prec = 50
                divisor = Decimal(denominator)
                if divisor == 0:
                    return None
                return Decimal(numerator) / divisor
        value = Decimal(token)
        return value if value.is_finite() else None
    except (InvalidOperation, ValueError):
        return None
