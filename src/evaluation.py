"""Campaign-separated model evaluation and labelled chart generation."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    METRICS_JSON_PATH,
    METRICS_PATH,
    RANDOM_SEED,
    REPORTS_DIR,
    RESULTS_DIR,
    STATIC_IMAGES_DIR,
)
MODEL_NAMES = ("Rule-based baseline", "TF-IDF + Logistic Regression", "TF-IDF + Linear SVM")


def _rule_prediction(text: str) -> str:
    lower = text.casefold()
    positive = sum(
        cue in lower
        for cue in (
            "verify your account",
            "claim your",
            "suspended",
            "hxxps://",
            "password",
            "urgent",
            "payment failed",
        )
    )
    negative = sum(cue in lower for cue in ("scheduled", "your appointment", "receipt"))
    return "phishing" if positive >= 2 and positive > negative else "legitimate"


def _safe_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    labels = ("legitimate", "phishing")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="phishing", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="phishing", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="phishing", zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def harmless_perturbation(text: str, index: int) -> str:
    """Apply one named, harmless formatting change without changing intent."""
    operations = (
        ("uppercase", lambda value: value.upper()),
        ("extra whitespace", lambda value: re.sub(r"\s+", "  ", value, count=1)),
        ("space around punctuation", lambda value: re.sub(r"([,:;.!?])", r" \1 ", value, count=1)),
        ("alternate defanged dot", lambda value: value.replace("[.]", "(dot)", 1)),
        ("spaced defanged scheme", lambda value: value.replace("hxxps://", "hxxps : //", 1)),
        ("terminal punctuation", lambda value: f"{value}!"),
        ("neutral courtesy phrase", lambda value: f"{value} Thank you."),
        ("alternate defanged dot token", lambda value: value.replace("[.]", "[dot]", 1)),
    )
    return operations[index % len(operations)][1](text)


PERTURBATION_NAMES = (
    "uppercase",
    "extra whitespace",
    "space around punctuation",
    "alternate defanged dot",
    "spaced defanged scheme",
    "terminal punctuation",
    "neutral courtesy phrase",
    "alternate defanged dot token",
)


def evaluate_models(
    models: dict[str, object],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_x: list[str],
    test_x: list[str],
) -> tuple[
    list[dict[str, object]],
    dict[str, list[str]],
    dict[str, dict[str, float]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if set(train_frame["campaign_group"]) & set(test_frame["campaign_group"]):
        raise ValueError("Train/test campaign groups overlap.")
    if len(test_x) != len(test_frame):
        raise ValueError("Holdout text and rows must have identical lengths.")
    y_true = test_frame["label"].tolist()
    predictions: dict[str, list[str]] = {
        "Rule-based baseline": [_rule_prediction(text) for text in test_x]
    }
    for name, model in models.items():
        predictions[name] = list(model.predict(test_x))

    urgent_mask = test_frame["label"].eq("legitimate") & test_frame["body_text"].str.contains(
        r"password|payroll|account reset", case=False, regex=True
    )
    rows: list[dict[str, object]] = []
    robustness: dict[str, dict[str, float]] = {}
    confusion_payload: dict[str, list[list[int]]] = {}
    errors: list[dict[str, object]] = []
    perturbation_rows: list[dict[str, object]] = []
    perturbation_index = next(
        (
            index
            for index, text in enumerate(test_x)
            if "hxxps://" in text.casefold() or "[.]" in text
        ),
        0,
    )
    base_text = test_x[perturbation_index]
    base_label = y_true[perturbation_index]
    variants = [
        {
            "index": index,
            "name": PERTURBATION_NAMES[index],
            "text": harmless_perturbation(base_text, index),
        }
        for index in range(len(PERTURBATION_NAMES))
    ]
    for name in MODEL_NAMES:
        pred = predictions[name]
        metrics = _safe_metrics(y_true, pred)
        for row_index, (actual, predicted) in enumerate(zip(y_true, pred, strict=True)):
            if actual != predicted:
                row = test_frame.iloc[row_index]
                errors.append(
                    {
                        "model": name,
                        "error_type": (
                            "false_positive" if actual == "legitimate" else "false_negative"
                        ),
                        "message_id": row["message_id"],
                        "campaign_group": row["campaign_group"],
                        "channel": row["channel"],
                        "subject_or_title": row["subject_or_title"],
                        "body_text": row["body_text"],
                        "true_label": actual,
                        "predicted_label": predicted,
                    }
                )
        urgent_total = int(urgent_mask.sum())
        urgent_false_positive = (
            int(sum(pred[i] == "phishing" for i in np.flatnonzero(urgent_mask)))
            if urgent_total
            else 0
        )
        perturb_x = [harmless_perturbation(text, index) for index, text in enumerate(test_x)]
        if name == "Rule-based baseline":
            perturbed = [_rule_prediction(text) for text in perturb_x]
            variant_predictions = [_rule_prediction(item["text"]) for item in variants]
        else:
            perturbed = list(models[name].predict(perturb_x))
            variant_predictions = list(models[name].predict([item["text"] for item in variants]))
        for variant, variant_prediction in zip(variants, variant_predictions, strict=True):
            perturbation_rows.append(
                {
                    "model": name,
                    "message_id": test_frame.iloc[perturbation_index]["message_id"],
                    "campaign_group": test_frame.iloc[perturbation_index]["campaign_group"],
                    "true_label": base_label,
                    "perturbation_index": variant["index"] + 1,
                    "perturbation": variant["name"],
                    "predicted_label": variant_prediction,
                    "prediction_changed": variant_prediction != pred[perturbation_index],
                    "base_prediction": pred[perturbation_index],
                    "text": variant["text"],
                }
            )
        flip_rate = (
            sum(a != b for a, b in zip(pred, perturbed, strict=True)) / len(pred)
            if pred
            else 0.0
        )
        robust_accuracy = float(accuracy_score(y_true, perturbed))
        robustness[name] = {
            "prediction_flip_rate": float(flip_rate),
            "perturbed_accuracy": robust_accuracy,
            "accuracy_change": robust_accuracy - metrics["accuracy"],
            "eight_variant_prediction_changes": int(
                sum(item != pred[perturbation_index] for item in variant_predictions)
            ),
        }
        row = {
            "model": name,
            **metrics,
            "legitimate_urgent_messages": urgent_total,
            "urgent_false_positives": urgent_false_positive,
            "urgent_false_positive_rate": (
                urgent_false_positive / urgent_total if urgent_total else 0.0
            ),
            "unseen_campaign_count": int(test_frame["campaign_group"].nunique()),
            "unseen_campaigns": ";".join(sorted(test_frame["campaign_group"].unique())),
            "holdout_messages": int(len(test_frame)),
            "legitimate_support": int(sum(label == "legitimate" for label in y_true)),
            "phishing_support": int(sum(label == "phishing" for label in y_true)),
            **robustness[name],
        }
        rows.append(row)
        confusion_payload[name] = confusion_matrix(
            y_true, pred, labels=("legitimate", "phishing")
        ).tolist()
    results = pd.DataFrame(rows).to_dict(orient="records")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "confusion_matrices.json").write_text(
        json.dumps(
            {"labels": ["legitimate", "phishing"], "matrices": confusion_payload},
            indent=2,
        ),
        encoding="utf-8",
    )
    return results, predictions, robustness, errors, perturbation_rows


def write_evaluation_artifacts(
    results: list[dict[str, object]],
    predictions: dict[str, list[str]],
    robustness: dict[str, dict[str, float]],
    test_frame: pd.DataFrame,
    errors: list[dict[str, object]],
    perturbation_rows: list[dict[str, object]],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(METRICS_PATH, index=False)
    METRICS_JSON_PATH.write_text(result_frame.to_json(orient="records", indent=2), encoding="utf-8")
    error_columns = [
        "model",
        "error_type",
        "message_id",
        "campaign_group",
        "channel",
        "subject_or_title",
        "body_text",
        "true_label",
        "predicted_label",
    ]
    pd.DataFrame(errors, columns=error_columns).to_csv(
        RESULTS_DIR / "error_analysis.csv", index=False
    )
    (REPORTS_DIR / "error_analysis.json").write_text(
        json.dumps(
            {
                "error_count_by_model": result_frame.set_index("model")[["fp", "fn"]].to_dict(
                    orient="index"
                ),
                "examples": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (REPORTS_DIR / "eight_perturbations.json").write_text(
        json.dumps(perturbation_rows, indent=2), encoding="utf-8"
    )

    sns.set_theme(style="whitegrid")
    score_frame = result_frame.melt(
        id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="Metric", value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=score_frame, x="Metric", y="Score", hue="model", ax=ax)
    ax.set(title="Campaign-Holdout Model Comparison", ylim=(0, 1), ylabel="Score (0–1)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "model_comparison.png", dpi=150)
    fig.savefig(STATIC_IMAGES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    first = results[0]
    matrix = np.array([[first["tn"], first["fp"]], [first["fn"], first["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
    ax.set(title=f"Confusion Matrix — {first['model']}", xlabel="Predicted label", ylabel="True label")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    drift = pd.DataFrame(
        [{"model": name, **values} for name, values in robustness.items()]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=drift, x="model", y="prediction_flip_rate", color="#2878b5", ax=ax)
    ax.set(title="Prediction Changes Under Harmless Text Perturbations",
           xlabel="Model", ylabel="Prediction flip rate (0–1)", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "robustness_drift.png", dpi=150)
    plt.close(fig)

    prediction_rows = []
    for model_name, values in predictions.items():
        for index, prediction in enumerate(values):
            row = test_frame.iloc[index]
            prediction_rows.append(
                {
                    "model": model_name,
                    "message_id": row["message_id"],
                    "campaign_group": row["campaign_group"],
                    "true_label": row["label"],
                    "prediction": prediction,
                }
            )
    pd.DataFrame(prediction_rows).to_csv(
        RESULTS_DIR / "holdout_predictions.csv", index=False
    )
