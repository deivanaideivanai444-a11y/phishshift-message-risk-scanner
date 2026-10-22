"""Fit the model comparison and persist a reusable prediction bundle."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import MODEL_PATH, RANDOM_SEED, REPORTS_DIR
from src.deduplication import (
    deduplicate_templates,
    remove_train_holdout_near_duplicates,
    split_by_campaign,
)
from src.evaluation import evaluate_models, write_evaluation_artifacts
from src.preprocessing import combined_text, prepare_dataset


def _feature_weight_audit(models: dict[str, Pipeline], limit: int = 20) -> dict[str, object]:
    """Save largest linear weights so synthetic text shortcuts are inspectable."""
    audit: dict[str, object] = {}
    for name, model in models.items():
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]
        feature_names = vectorizer.get_feature_names_out()
        weights = classifier.coef_[0]
        positive_indices = weights.argsort()[::-1][:limit]
        negative_indices = weights.argsort()[:limit]
        audit[name] = {
            "positive_class": str(classifier.classes_[1]),
            "positive_terms": [
                {"term": str(feature_names[index]), "weight": float(weights[index])}
                for index in positive_indices
            ],
            "negative_class": str(classifier.classes_[0]),
            "negative_terms": [
                {"term": str(feature_names[index]), "weight": float(weights[index])}
                for index in negative_indices
            ],
        }
    return audit


def build_models(seed: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Return comparable word/character TF-IDF classifiers."""
    def pipeline(classifier: object) -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                        strip_accents="unicode",
                    ),
                ),
                ("classifier", classifier),
            ]
        )

    return {
        "TF-IDF + Logistic Regression": pipeline(
            LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed)
        ),
        "TF-IDF + Linear SVM": pipeline(
            LinearSVC(class_weight="balanced", random_state=seed)
        ),
    }


def fit_and_evaluate(dataset_path: Path, model_path: Path = MODEL_PATH) -> dict[str, object]:
    frame = pd.read_csv(dataset_path)
    cleaned, cleaning_report = prepare_dataset(frame)
    label_counts = cleaned["label"].value_counts().to_dict()
    audit = {
        "rows_by_label": {str(key): int(value) for key, value in label_counts.items()},
        "rows_by_channel_and_label": {
            str(channel): {
                str(label): int(count)
                for label, count in counts.items()
            }
            for channel, counts in cleaned.groupby("channel")["label"].value_counts().unstack(
                fill_value=0
            ).to_dict(orient="index").items()
        },
        "link_text_rate_by_label": {
            str(label): float(
                group["body_text"].str.contains(r"hxxps?://|https?://", case=False).mean()
            )
            for label, group in cleaned.groupby("label")
        },
        "sender_domain_type_rate_by_label": {
            str(label): {
                str(sender_type): float(rate)
                for sender_type, rate in group["sender_domain_type"]
                .value_counts(normalize=True)
                .to_dict()
                .items()
            }
            for label, group in cleaned.groupby("label")
        },
        "features_used": ["channel", "subject_or_title", "body_text"],
        "excluded_metadata": [
            "message_id",
            "sanitized_url_features",
            "sender_domain_type",
            "label",
            "campaign_group",
            "collected_date",
        ],
        "interpretation": (
            "The target and campaign metadata are never passed to vectorizers. "
            "Feature rates are recorded to expose synthetic text/metadata shortcuts."
        ),
    }
    deduplicated, dedup_report = deduplicate_templates(cleaned)
    train_frame, test_frame = split_by_campaign(deduplicated, seed=RANDOM_SEED)
    train_frame, cross_split_report = remove_train_holdout_near_duplicates(
        train_frame, test_frame
    )
    if set(train_frame["label"]) != {"phishing", "legitimate"}:
        raise ValueError("Deduplication left the training partition without both classes.")
    if set(test_frame["label"]) != {"phishing", "legitimate"}:
        raise ValueError("Campaign holdout does not contain both classes.")
    train_x = [
        combined_text(row.subject_or_title, row.body_text, row.channel)
        for row in train_frame.itertuples()
    ]
    test_x = [
        combined_text(row.subject_or_title, row.body_text, row.channel)
        for row in test_frame.itertuples()
    ]
    models = build_models()
    for model in models.values():
        model.fit(train_x, train_frame["label"])
    feature_audit = _feature_weight_audit(models)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_feature_audit.json").write_text(
        json.dumps(feature_audit, indent=2), encoding="utf-8"
    )
    results, predictions, robustness, errors, perturbation_rows = evaluate_models(
        models, train_frame, test_frame, train_x, test_x
    )
    write_evaluation_artifacts(
        results, predictions, robustness, test_frame, errors, perturbation_rows
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "labels": ["legitimate", "phishing"],
            "abstention_threshold": 0.65,
            "random_seed": RANDOM_SEED,
            "train_campaigns": sorted(train_frame["campaign_group"].unique()),
            "test_campaigns": sorted(test_frame["campaign_group"].unique()),
        },
        model_path,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        **cleaning_report,
        **dedup_report,
        **cross_split_report,
        "feature_and_label_audit": audit,
        "top_weighted_text_features": "reports/model_feature_audit.json",
        "train_rows": len(train_frame),
        "holdout_rows": len(test_frame),
        "train_campaigns": sorted(train_frame["campaign_group"].unique()),
        "holdout_campaigns": sorted(test_frame["campaign_group"].unique()),
    }
    (REPORTS_DIR / "cleaning_deduplication_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return {"results": results, "report": report, "model_path": str(model_path)}
