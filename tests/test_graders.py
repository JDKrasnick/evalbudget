import unittest

from evalbudget.graders import grade_output, normalize_exact, validate_case


class GraderTests(unittest.TestCase):
    def test_exact_normalizes_unicode_whitespace_quotes_and_fences(self):
        self.assertEqual(normalize_exact('  “ＯＢＪＥＣＴ”  '), "object")
        self.assertEqual(grade_output({"expected": "object"}, '"object"'), 1.0)
        self.assertEqual(grade_output({"expected": "object"}, "```text\nobject\n```"), 1.0)

    def test_exact_does_not_discard_meaningful_punctuation(self):
        self.assertEqual(grade_output({"expected": "x-3bA"}, ".x-3bA"), 0.0)

    def test_accepted_allows_multiple_canonical_answers(self):
        case = {"expected": ["yes", "certainly"], "grader": "accepted"}
        self.assertEqual(grade_output(case, "Certainly"), 1.0)
        self.assertEqual(grade_output(case, "maybe"), 0.0)

    def test_numeric_accepts_units_fractions_and_tolerance(self):
        self.assertEqual(grade_output({"expected": 3, "grader": "numeric"}, "3 minutes"), 1.0)
        self.assertEqual(grade_output({"expected": "1/3", "grader": "numeric"}, "The answer is 1/3."), 1.0)
        case = {"expected": 0.333, "grader": {"type": "numeric", "abs_tol": 0.001}}
        self.assertEqual(grade_output(case, "0.3339"), 1.0)

    def test_numeric_rejects_ambiguous_or_non_numeric_output(self):
        case = {"expected": 3, "grader": "numeric"}
        self.assertEqual(grade_output(case, "between 2 and 4"), 0.0)
        self.assertEqual(grade_output(case, "three"), 0.0)

    def test_regex_supports_flags_and_search_mode(self):
        case = {
            "expected": r"answer:\s+(red|blue)",
            "grader": {"type": "regex", "flags": "i", "fullmatch": False},
        }
        self.assertEqual(grade_output(case, "My Answer: BLUE."), 1.0)

    def test_validation_rejects_bad_grader_configuration(self):
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            validate_case({"id": "1", "prompt": "p", "expected": [], "grader": "accepted"})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_case(
                {"id": "1", "prompt": "p", "expected": 1, "grader": {"type": "numeric", "tolerence": 1}}
            )
        with self.assertRaisesRegex(ValueError, "invalid expected regex"):
            validate_case({"id": "1", "prompt": "p", "expected": "[", "grader": "regex"})


if __name__ == "__main__":
    unittest.main()

