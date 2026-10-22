"""Project configuration with paths rooted at this file."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_IMAGES_DIR = BASE_DIR / "static" / "images"
DATASET_PATH = DATA_DIR / "synthetic_messages.csv"
MODEL_PATH = MODEL_DIR / "model_bundle.joblib"
METRICS_PATH = RESULTS_DIR / "model_comparison.csv"
METRICS_JSON_PATH = RESULTS_DIR / "model_comparison.json"
RANDOM_SEED = 37
ABSTENTION_THRESHOLD = 0.65
MAX_MESSAGE_LENGTH = 5000
