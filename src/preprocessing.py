"""Dataset checks and conservative text normalization."""

import re
import unicodedata
from collections.abc import Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "message_id",
    "channel",
    "subject_or_title",
    "body_text",
    "sanitized_url_features",
    "sender_domain_type",
    "label",
    "campaign_group",
    "collected_date",
}
VALID_LABELS = {"phishing", "legitimate"}


def validate_dataset(frame: pd.DataFrame) -> None:
    """Raise a helpful ValueError when the dataset schema or labels are invalid."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    invalid = set(frame["label"].dropna().astype(str).str.lower()) - VALID_LABELS
    if invalid:
        raise ValueError(f"Invalid labels found: {', '.join(sorted(invalid))}")
    if frame.empty:
        raise ValueError("Dataset contains no messages.")


def normalize_text(value: object) -> str:
    """Normalize comparable text while preserving word boundaries and punctuation."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate, fill missing text fields, and report removed/filled records."""
    validate_dataset(frame)
    cleaned = frame.copy()
    missing_cells = int(cleaned[list(REQUIRED_COLUMNS)].isna().sum().sum())
    for column in ("subject_or_title", "body_text", "sanitized_url_features"):
        cleaned[column] = cleaned[column].fillna("").astype(str)
    for column in ("channel", "sender_domain_type", "campaign_group"):
        cleaned[column] = cleaned[column].fillna("unknown").astype(str)
    cleaned["label"] = cleaned["label"].str.lower().str.strip()
    cleaned = cleaned[cleaned["label"].isin(VALID_LABELS)].copy()
    cleaned["normalized_text"] = (
        cleaned["subject_or_title"].map(normalize_text)
        + " "
        + cleaned["body_text"].map(normalize_text)
    ).str.strip()
    report = {
        "input_rows": int(len(frame)),
        "missing_cells_filled": missing_cells,
        "invalid_label_rows_removed": int(len(frame) - len(cleaned)),
        "rows_after_validation": int(len(cleaned)),
    }
    return cleaned.reset_index(drop=True), report


def combined_text(subject: object, body: object, channel: object = "") -> str:
    """Build the same feature text for training and interactive predictions."""
    return " ".join(part for part in (str(channel), str(subject), str(body)) if part).strip()
