"""Reusable local prediction with conservative abstention."""

from pathlib import Path
import pickle
from typing import Any

import joblib

from config import ABSTENTION_THRESHOLD, MODEL_PATH
from src.preprocessing import combined_text
from src.safety import find_risk_cues


class ModelLoadError(RuntimeError):
    """Raised when the saved model bundle cannot be loaded."""


def load_model_bundle(path: Path = MODEL_PATH) -> dict[str, Any]:
    try:
        bundle = joblib.load(path)
        required_models = {
            "TF-IDF + Logistic Regression",
            "TF-IDF + Linear SVM",
        }
        if not isinstance(bundle, dict) or not isinstance(bundle.get("models"), dict):
            raise ValueError("Model bundle has no trained model mapping.")
        if not required_models.issubset(bundle["models"]):
            raise ValueError("Model bundle is missing one or more required classifiers.")
        return bundle
    except (
        OSError,
        ValueError,
        EOFError,
        KeyError,
        ImportError,
        AttributeError,
        TypeError,
        pickle.UnpicklingError,
    ) as exc:
        raise ModelLoadError(f"Unable to load trained models from {path}: {exc}") from exc


def predict_message(
    body_text: str,
    subject: str = "",
    channel: str = "sms",
    model_bundle: dict[str, Any] | None = None,
    abstention_threshold: float = ABSTENTION_THRESHOLD,
) -> dict[str, object]:
    if not isinstance(body_text, str) or not body_text.strip():
        raise ValueError("Enter a message to analyse.")
    if not 0.5 <= abstention_threshold <= 0.99:
        raise ValueError("Abstention threshold must be between 0.50 and 0.99.")
    if model_bundle is None:
        model_bundle = load_model_bundle()
    models = model_bundle["models"]
    text = combined_text(subject, body_text, channel)
    primary = models["TF-IDF + Logistic Regression"]
    probability = primary.predict_proba([text])[0]
    classes = list(primary.classes_)
    top_index = int(probability.argmax())
    raw_label = str(classes[top_index])
    confidence = float(probability[top_index])
    all_predictions = [str(model.predict([text])[0]) for model in models.values()]
    disagreement = len(set(all_predictions)) > 1
    uncertain = confidence < abstention_threshold or disagreement
    category = (
        "Uncertain — Manual Review Required"
        if uncertain
        else ("Likely Phishing" if raw_label == "phishing" else "Likely Legitimate")
    )
    cues = find_risk_cues(f"{subject} {body_text}")
    if uncertain:
        explanation = (
            "The model score is below the configured threshold or the trained models "
            "disagree. Review the message using trusted, independent channels."
        )
    elif raw_label == "phishing":
        explanation = (
            "Text patterns resemble simulated phishing examples. Do not follow links or "
            "share credentials based only on this result."
        )
    else:
        explanation = (
            "Text patterns resemble legitimate examples in this synthetic training set. "
            "This does not verify the sender or make links safe."
        )
    return {
        "category": category,
        "predicted_label": raw_label,
        "score_type": "Logistic Regression estimated class probability",
        "confidence": confidence,
        "risk_cues": cues,
        "explanation": explanation,
        "manual_review_recommended": uncertain,
        "models_disagree": disagreement,
        "threshold": abstention_threshold,
    }
