import unittest
from pathlib import Path

from src.predict import ModelLoadError, load_model_bundle, predict_message


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_model_bundle()

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Enter a message"):
            predict_message("  ", model_bundle=self.bundle)

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            predict_message("Sample", model_bundle=self.bundle, abstention_threshold=0.2)

    def test_model_loading_failure_is_explicit(self):
        with self.assertRaises(ModelLoadError):
            load_model_bundle(Path("missing-model-for-test.joblib"))

    def test_prediction_has_explanation_and_cues(self):
        result = predict_message(
            "Urgent: verify your password at hxxps://account[.]example[.]invalid",
            model_bundle=self.bundle,
        )
        self.assertIn(result["category"], {
            "Likely Phishing", "Likely Legitimate", "Uncertain — Manual Review Required"
        })
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIn("link-like text", result["risk_cues"])
        self.assertTrue(result["explanation"])

    def test_legitimate_urgent_password_reset_is_in_holdout_report(self):
        import pandas as pd

        from config import METRICS_PATH

        metrics = pd.read_csv(METRICS_PATH)
        self.assertTrue((metrics["legitimate_urgent_messages"] > 0).all())
        self.assertTrue(metrics["urgent_false_positive_rate"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
