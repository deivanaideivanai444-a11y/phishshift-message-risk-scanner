import json
import unittest
from unittest.mock import patch

import pandas as pd

from app import app
from config import METRICS_PATH, MODEL_PATH, REPORTS_DIR
from src.evaluation import harmless_perturbation
from src.predict import ModelLoadError, load_model_bundle, predict_message


class AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise RuntimeError("Train the model before running acceptance tests.")
        cls.bundle = load_model_bundle()
        cls.client = app.test_client()

    def test_at01_legitimate_urgent_notice_and_false_positive_report(self):
        message = (
            "Payroll password reset notice: your account reset is complete. "
            "Use the employer portal you normally visit; this urgent message has no link."
        )
        result = predict_message(message, channel="email", model_bundle=self.bundle)
        self.assertIn("category", result)
        metrics = pd.read_csv(METRICS_PATH)
        self.assertTrue((metrics.urgent_false_positive_rate >= 0).all())
        self.assertTrue((metrics.urgent_false_positive_rate <= 1).all())
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "acceptance_scenarios.json").write_text(
            json.dumps(
                {
                    "AT-01": {
                        "message": message,
                        "prediction": result,
                        "measured_urgent_false_positive_rates": {
                            row["model"]: row["urgent_false_positive_rate"]
                            for row in metrics.to_dict(orient="records")
                        },
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_at02_unfamiliar_wording_for_unseen_campaign(self):
        heldout = set(self.bundle["test_campaigns"])
        self.assertTrue(heldout)
        self.assertFalse(heldout.intersection(self.bundle["train_campaigns"]))
        test_message = (
            "A courier bulletin says the delivery window closes shortly. Confirm the route "
            "at hxxps://route[.]example[.]invalid to keep the request active; fictional exercise."
        )
        result = predict_message(
            test_message,
            channel="sms",
            model_bundle=self.bundle,
        )
        self.assertIn("category", result)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        existing = {}
        scenario_path = REPORTS_DIR / "acceptance_scenarios.json"
        if scenario_path.exists():
            existing = json.loads(scenario_path.read_text(encoding="utf-8"))
        existing["AT-02"] = {
            "associated_holdout_campaign": sorted(heldout)[0],
            "all_holdout_campaigns_are_excluded_from_training": True,
            "new_harmless_message": test_message,
            "prediction": result,
        }
        scenario_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def test_at03_defanged_url_and_punctuation_variations(self):
        original = (
            "Your weekly schedule is ready; review the note at "
            "hxxps://portal[.]example[.]invalid."
        )
        base = predict_message(original, model_bundle=self.bundle)
        variants = [harmless_perturbation(original, index) for index in range(8)]
        results = [predict_message(text, model_bundle=self.bundle) for text in variants]
        self.assertEqual(len(results), 8)
        for result in results:
            self.assertTrue(0 <= result["confidence"] <= 1)
            self.assertTrue(result["score_type"])
        self.assertTrue(0 <= base["confidence"] <= 1)
        score_deltas = [float(item["confidence"]) - float(base["confidence"]) for item in results]
        self.assertTrue(any(abs(delta) > 1e-9 for delta in score_deltas))
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "acceptance_perturbation_results.json").write_text(
            json.dumps(
                {
                    "base_message": original,
                    "base_category": base["category"],
                    "base_confidence": base["confidence"],
                    "variants": [
                        {
                            "index": index,
                            "message": text,
                            "category": result["category"],
                            "confidence": result["confidence"],
                            "confidence_delta": score_deltas[index],
                        }
                        for index, (text, result) in enumerate(zip(variants, results, strict=True))
                    ],
                    "score_changed_for_at_least_one_variant": any(
                        abs(delta) > 1e-9 for delta in score_deltas
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_flask_empty_and_invalid_input(self):
        empty = self.client.post("/analyse", data={"channel": "sms", "body_text": ""})
        invalid = self.client.post("/analyse", data={"channel": "fax", "body_text": "hello"})
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(invalid.status_code, 400)

    def test_flask_model_loading_failure_is_friendly(self):
        with patch("app.load_model_bundle", side_effect=ModelLoadError("test failure")):
            response = self.client.post(
                "/analyse", data={"channel": "sms", "body_text": "A harmless sample notice."}
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"scanner model is not available", response.data)

    def test_successful_analysis_keeps_only_token_in_cookie_session(self):
        text = "Payroll schedule notice for this synthetic test."
        response = self.client.post(
            "/analyse", data={"channel": "email", "body_text": text}
        )
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as client_session:
            self.assertEqual(set(client_session.keys()), {"analysis_token"})
        cookie = self.client.get_cookie(app.config["SESSION_COOKIE_NAME"])
        self.assertNotIn(text, cookie.value)

    def test_health_endpoint_confirms_loaded_model(self):
        self.assertEqual(self.client.get("/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
