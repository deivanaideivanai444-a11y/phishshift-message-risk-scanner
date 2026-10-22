"""Regenerate evaluation artifacts from a newly trained model bundle."""

from config import DATASET_PATH
from src.train import fit_and_evaluate


if __name__ == "__main__":
    output = fit_and_evaluate(DATASET_PATH)
    print("Campaign-separated evaluation complete.")
    print(f"Results: {output['results']}")
