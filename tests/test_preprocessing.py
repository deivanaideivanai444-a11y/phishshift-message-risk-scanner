import unittest

import pandas as pd

from src.deduplication import deduplicate_templates, split_by_campaign
from src.preprocessing import prepare_dataset, validate_dataset


def example_frame() -> pd.DataFrame:
    descriptors = (
        "alpha winter reading room with quiet desks and catalog cards",
        "bravo spring staff roster with shift coverage and office hours",
        "charlie summer garden calendar with planting notes and sunlight logs",
        "delta autumn music schedule with rehearsal notes and instrument lists",
        "echo coastal travel plan with ferry times and map references",
        "foxtrot mountain course outline with lesson names and campus rooms",
        "golf city museum visit with exhibit guides and gallery hours",
        "hotel forest science journal with field notes and specimen labels",
    )
    rows = []
    for campaign in range(8):
        for label in ("phishing", "legitimate"):
            rows.append(
                {
                    "message_id": f"{campaign}-{label}",
                    "channel": "sms",
                    "subject_or_title": "",
                    "body_text": (
                        (
                            f"Fictional security practice sample {campaign}: inspect a "
                            f"simulated notice about account access. Scenario context: {descriptors[campaign]}."
                        )
                        if label == "phishing"
                        else (
                            f"Routine classroom reminder {campaign}: the weekly reading "
                            f"session begins at its normal time. Scenario context: {descriptors[campaign]}."
                        )
                    ),
                    "sanitized_url_features": "no link",
                    "sender_domain_type": "synthetic",
                    "label": label,
                    "campaign_group": f"campaign-{campaign}",
                    "collected_date": "2025-01-01",
                }
            )
    return pd.DataFrame(rows)


class PreprocessingTests(unittest.TestCase):
    def test_missing_required_columns_is_reported(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_dataset(pd.DataFrame({"message_id": ["one"]}))

    def test_missing_text_fields_are_filled(self):
        frame = example_frame()
        frame.loc[0, "body_text"] = None
        cleaned, report = prepare_dataset(frame)
        self.assertEqual(cleaned.loc[0, "body_text"], "")
        self.assertEqual(report["missing_cells_filled"], 1)

    def test_exact_and_near_duplicates_are_removed_before_split(self):
        frame = example_frame()
        duplicate = frame.iloc[[0]].copy()
        duplicate["message_id"] = "exact-copy"
        cross_campaign_duplicate = frame.iloc[[0]].copy()
        cross_campaign_duplicate["message_id"] = "cross-campaign-exact-copy"
        cross_campaign_duplicate["campaign_group"] = "duplicate-campaign"
        conflicting_duplicate = frame.iloc[[0]].copy()
        conflicting_duplicate["message_id"] = "conflicting-label-copy"
        conflicting_duplicate["label"] = "legitimate"
        near = frame.iloc[[1]].copy()
        near["message_id"] = "near-copy"
        near["campaign_group"] = "near-duplicate-campaign"
        near["body_text"] = near["body_text"].str.replace(".", "!", regex=False)
        combined = pd.concat(
            [frame, duplicate, cross_campaign_duplicate, conflicting_duplicate, near],
            ignore_index=True,
        )
        clean, report = deduplicate_templates(combined, similarity_threshold=0.90)
        self.assertEqual(report["exact_duplicates_removed"], 2)
        self.assertGreaterEqual(report["near_duplicates_removed"], 1)
        self.assertGreaterEqual(report["cross_label_exact_duplicates_preserved"], 1)
        self.assertNotIn("duplicate-campaign", set(clean.campaign_group))
        self.assertNotIn("near-duplicate-campaign", set(clean.campaign_group))
        self.assertIn("campaign-0", set(clean.campaign_group))
        train, test = split_by_campaign(clean, test_fraction=0.25)
        self.assertFalse(set(train.campaign_group) & set(test.campaign_group))
        self.assertEqual(len(set(train.campaign_group) | set(test.campaign_group)), 8)
        train_signatures = set(train.body_text.str.casefold())
        test_signatures = set(test.body_text.str.casefold())
        self.assertFalse(train_signatures & test_signatures)


if __name__ == "__main__":
    unittest.main()
