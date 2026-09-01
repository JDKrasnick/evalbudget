import unittest

from evalbudget.engine import Observation, confidence_sequence, evaluate_observations


class EngineTests(unittest.TestCase):
    def test_reports_category_breakdown(self):
        observations = [
            Observation("1", 0, 1, category="math"),
            Observation("2", 1, 1, category="math"),
            Observation("3", 1, 0, category="writing"),
        ]
        result = evaluate_observations(observations, min_samples=3, max_samples=3)
        self.assertEqual(result.category_summaries["math"]["samples"], 2)
        self.assertEqual(result.category_summaries["math"]["mean_difference"], 0.5)
        self.assertEqual(result.category_summaries["writing"]["baseline_wins"], 1)

    def test_strong_candidate_stops_early(self):
        observations = [Observation(str(index), 0, 1) for index in range(100)]
        result = evaluate_observations(observations, min_samples=10, max_samples=100)
        self.assertEqual(result.decision, "candidate_better")
        self.assertLess(result.samples_used, 100)
        self.assertGreater(result.confidence_lower, 0)

    def test_max_samples_returns_inconclusive(self):
        observations = [Observation(str(index), 1, 1) for index in range(20)]
        result = evaluate_observations(observations, min_samples=5, max_samples=10)
        self.assertEqual(result.decision, "inconclusive")
        self.assertEqual(result.stop_reason, "max_samples_reached")
        self.assertEqual(result.samples_used, 10)

    def test_rejects_out_of_range_scores(self):
        with self.assertRaisesRegex(ValueError, "must be in"):
            evaluate_observations([Observation("bad", 0, 2)], min_samples=1)

    def test_confidence_sequence_is_bounded(self):
        lower, upper = confidence_sequence(1.0, 1, 0.05)
        self.assertEqual(upper, 1.0)
        self.assertGreaterEqual(lower, -1.0)


if __name__ == "__main__":
    unittest.main()
