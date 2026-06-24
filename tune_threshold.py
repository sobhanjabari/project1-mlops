import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = PROJECT_DIR / "predictions.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "threshold_tuning_from_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple decision thresholds on a predictions CSV."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = (
        args.predictions if args.predictions.is_absolute() else PROJECT_DIR / args.predictions
    )
    output_path = args.output if args.output.is_absolute() else PROJECT_DIR / args.output

    df = pd.read_csv(predictions_path)
    y_true = df["actual_is_late"]
    y_proba = df["late_probability"]

    thresholds = np.arange(args.start, args.stop + args.step / 2, args.step)
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        results.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    print(result_df)
    print(f"\nSaved threshold table as: {output_path}")
    print("\nBest threshold by F1:")
    print(result_df.sort_values("f1", ascending=False).head(1))


if __name__ == "__main__":
    main()