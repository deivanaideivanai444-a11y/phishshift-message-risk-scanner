"""Flask dashboard. Message text exists only in process memory during this run."""

import json
import secrets
import threading
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, session

from config import ABSTENTION_THRESHOLD, MAX_MESSAGE_LENGTH, METRICS_PATH
from src.predict import ModelLoadError, load_model_bundle, predict_message

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
_recent_by_session: dict[str, list[dict[str, object]]] = {}
_recent_lock = threading.Lock()


def _performance_summary() -> list[dict[str, object]]:
    try:
        if not METRICS_PATH.exists():
            return []
        return pd.read_csv(METRICS_PATH).to_dict(orient="records")
    except (OSError, pd.errors.ParserError, ValueError):
        app.logger.exception("Unable to read saved model comparison results.")
        return []


def _error_examples() -> list[dict[str, object]]:
    error_path = METRICS_PATH.parent / "error_analysis.csv"
    try:
        if not error_path.exists():
            return []
        errors = pd.read_csv(error_path)
        if errors.empty:
            return []
        return (
            errors.groupby("model", group_keys=False)
            .head(2)
            .fillna("")
            .to_dict(orient="records")
        )
    except (OSError, pd.errors.ParserError, ValueError):
        app.logger.exception("Unable to read saved holdout error analysis.")
        return []


def _recent_analyses() -> list[dict[str, object]]:
    token = session.get("analysis_token")
    if not token:
        return []
    with _recent_lock:
        return list(_recent_by_session.get(token, []))


def _render(error: str | None = None, result: dict[str, object] | None = None, values=None, status=200):
    return (
        render_template(
            "index.html",
            error=error,
            result=result,
            values=values or {},
            metrics=_performance_summary(),
            error_examples=_error_examples(),
            recent=_recent_analyses(),
            threshold=ABSTENTION_THRESHOLD,
        ),
        status,
    )


@app.get("/")
def index():
    return _render()


@app.post("/analyse")
def analyse():
    channel = request.form.get("channel", "").strip().lower()
    subject = request.form.get("subject", "").strip()
    body = request.form.get("body_text", "")
    values = {"channel": channel, "subject": subject, "body_text": body}
    if channel not in {"sms", "email"}:
        return _render("Choose SMS or Email before analysing.", values=values, status=400)
    if not body.strip():
        return _render("Enter a message to analyse.", values=values, status=400)
    if len(body) > MAX_MESSAGE_LENGTH or len(subject) > 300:
        return _render(
            f"Message must be at most {MAX_MESSAGE_LENGTH} characters; subject must be at most 300.",
            values=values,
            status=400,
        )
    try:
        bundle = load_model_bundle()
        result = predict_message(body, subject, channel, bundle, ABSTENTION_THRESHOLD)
    except ModelLoadError:
        app.logger.exception("The trained model could not be loaded.")
        return _render(
            "The scanner model is not available. Please train the model and try again.",
            values=values,
            status=503,
        )
    except ValueError as exc:
        return _render(str(exc), values=values, status=400)

    token = session.get("analysis_token")
    if not token:
        token = secrets.token_urlsafe(24)
        session["analysis_token"] = token
    entry = {
        "channel": channel.upper(),
        "category": result["category"],
        "preview": (body.strip()[:82] + ("…" if len(body.strip()) > 82 else "")),
    }
    with _recent_lock:
        recent = _recent_by_session.setdefault(token, [])
        recent.insert(0, entry)
        del recent[5:]
    # The submitted body is deliberately not put into Flask's cookie session or server history.
    return _render(result=result, values=values)


@app.get("/health")
def health():
    try:
        load_model_bundle()
    except ModelLoadError:
        return {"status": "model unavailable"}, 503
    return {"status": "ready"}, 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
