"""Safe, local-only analysis helpers. Submitted URLs are never opened."""

import re


RISK_PATTERNS = {
    "credential request": re.compile(
        r"\b(password|passcode|verification code|one[- ]time code|login details)\b",
        re.IGNORECASE,
    ),
    "urgent or threatening language": re.compile(
        r"\b(urgent|immediately|within \d+ hours?|suspended|locked|final warning)\b",
        re.IGNORECASE,
    ),
    "payment or reward bait": re.compile(
        r"\b(refund|prize|gift card|payment failed|claim your|unusual payment)\b",
        re.IGNORECASE,
    ),
    "link-like text": re.compile(
        r"(?:hxxps?://|https?://|www\.|\[[.]|\(dot\))", re.IGNORECASE
    ),
    "account action request": re.compile(
        r"\b(verify|confirm|reset|update|restore)\b", re.IGNORECASE
    ),
}


def find_risk_cues(text: str) -> list[str]:
    """Return descriptive cues, not a verdict about whether a message is malicious."""
    return [name for name, pattern in RISK_PATTERNS.items() if pattern.search(text)]
