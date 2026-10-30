"""Write a focused, aggregate-only classification data review."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.classification_review import review_classification
from src.utils.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Optional TP2 root containing classification/.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "classification_round1")
    args = parser.parse_args()
    result = review_classification(args.data_dir, args.output_dir)
    print(result["status"])
    print("Rows:", result["shapes"]["train"][0], "train;", result["shapes"]["test"][0], "test")
    print("Target counts:", result["target"]["counts"], "positive rate:", f'{result["target"]["positive_rate"]:.4%}')
    print("Saved aggregate-only report:", args.output_dir / "data_review.json")


if __name__ == "__main__":
    main()
