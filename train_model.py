"""Train and evaluate the campaign-separated model comparison."""

import argparse
from pathlib import Path

from config import DATASET_PATH, MODEL_PATH
from src.train import fit_and_evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    outcome = fit_and_evaluate(args.dataset, args.model)
    print(f"Model bundle saved to {outcome['model_path']}")
    print(f"Cleaning and deduplication report: reports/cleaning_deduplication_report.json")
    for row in outcome["results"]:
        print(
            f"{row['model']}: F1={row['f1']:.3f}, accuracy={row['accuracy']:.3f}, "
            f"urgent legitimate FPR={row['urgent_false_positive_rate']:.3f}"
        )
