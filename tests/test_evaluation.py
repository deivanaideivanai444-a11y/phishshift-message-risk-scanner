import json
import unittest

import pandas as pd

from config import METRICS_PATH, MODEL_PATH, REPORTS_DIR, RESULTS_DIR
from src.predict import load_model_bundle


class EvaluationIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise RuntimeError("Train the model before running evaluation integrity tests.")
        cls.bundle = load_model_bundle()
        cls.metrics = pd.read_csv(METRICS_PATH)
        cls.predictions = pd.read_csv(RESULTS_DIR / "holdout_predictions.csv")

    def test_models_use_same_campaign_heldout_rows(self):
        self.assertFalse(
            set(self.bundle["train_campaigns"]) & set(self.bundle["test_campaigns"])
        )
        campaign_sets = [
            set(rows["campaign_group"])
            for _, rows in self.predictions.groupby("model")
        ]
        message_sets = [
            set(rows["message_id"])
            for _, rows in self.predictions.groupby("model")
        ]
        self.assertEqual(len(campaign_sets), 3)
        self.assertTrue(all(value == campaign_sets[0] for value in campaign_sets[1:]))
        self.assertTrue(all(value == message_sets[0] for value in message_sets[1:]))

    def test_dataset_proxies_are_audited_and_metadata_is_not_model_input(self):
        dataset = pd.read_csv("data/synthetic_messages.csv")
        self.assertEqual(len(dataset), 480)
        self.assertEqual(dataset["label"].value_counts().to_dict(), {
            "phishing": 240,
            "legitimate": 240,
        })
        link_rates = dataset.assign(
            has_link=dataset["body_text"].str.contains("hxxps://")
        ).groupby("label")["has_link"].mean()
        self.assertEqual(float(link_rates["phishing"]), 0.5)
        self.assertEqual(float(link_rates["legitimate"]), 0.5)

        report = json.loads(
            (REPORTS_DIR / "cleaning_deduplication_report.json").read_text(
                encoding="utf-8"
            )
        )
        audit = report["feature_and_label_audit"]
        self.assertEqual(
            set(audit["features_used"]),
            {"channel", "subject_or_title", "body_text"},
        )
        self.assertNotIn("sender_domain_type", audit["features_used"])
        self.assertNotIn("campaign_group", audit["features_used"])
        self.assertEqual(report["train_holdout_exact_matches_removed"], 0)
        self.assertEqual(report["train_holdout_near_matches_removed"], 0)

    def test_confusion_counts_match_common_test_set(self):
        expected_rows = self.predictions["message_id"].nunique()
        for row in self.metrics.to_dict(orient="records"):
            self.assertEqual(row["holdout_messages"], expected_rows)
            self.assertEqual(row["tn"] + row["fp"] + row["fn"] + row["tp"], expected_rows)
            self.assertGreater(row["legitimate_urgent_messages"], 0)
            self.assertAlmostEqual(
                row["urgent_false_positive_rate"],
                row["urgent_false_positives"] / row["legitimate_urgent_messages"],
            )

    def test_error_analysis_records_false_positive_and_negative_examples(self):
        errors = pd.read_csv(RESULTS_DIR / "error_analysis.csv")
        self.assertEqual(set(errors.columns), {
            "model", "error_type", "message_id", "campaign_group", "channel",
            "subject_or_title", "body_text", "true_label", "predicted_label",
        })
        report = json.loads((REPORTS_DIR / "error_analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(len(errors), len(report["examples"]))
        for error in errors.to_dict(orient="records"):
            if error["error_type"] == "false_positive":
                self.assertEqual(error["true_label"], "legitimate")
                self.assertEqual(error["predicted_label"], "phishing")
            else:
                self.assertEqual(error["true_label"], "phishing")
                self.assertEqual(error["predicted_label"], "legitimate")

    def test_eight_perturbations_have_measured_predictions_for_each_model(self):
        rows = json.loads((REPORTS_DIR / "eight_perturbations.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 24)
        for model_name in self.metrics["model"]:
            model_rows = [row for row in rows if row["model"] == model_name]
            self.assertEqual(len(model_rows), 8)
            self.assertTrue(all(row["perturbation"] and row["text"] for row in model_rows))
            self.assertTrue(
                all(
                    row["prediction_changed"]
                    == (row["predicted_label"] != row["base_prediction"])
                    for row in model_rows
                )
            )


if __name__ == "__main__":
    unittest.main()
