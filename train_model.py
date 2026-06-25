import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "model_dataset_v2.csv"
MODEL_PATH = PROJECT_DIR / "late_delivery_model.joblib"
PREDICTIONS_PATH = PROJECT_DIR / "predictions.csv"
METADATA_PATH = PROJECT_DIR / "model_metadata.json"
THRESHOLD_REPORT_PATH = PROJECT_DIR / "threshold_report.csv"
MODEL_COMPARISON_PATH = PROJECT_DIR / "model_comparison.csv"


def get_feature_columns(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature columns, excluding leakage columns."""
    leakage_or_id_cols = [
        target_col,
        "order_status",
        "order_id",
        "customer_id",
        "customer_unique_id",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    ]

    X = df.drop(columns=[c for c in leakage_or_id_cols if c in df.columns])
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    return numeric_features, categorical_features


def make_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str = "hist_gradient_boosting",
) -> Pipeline:
    """
    Build a model pipeline for late-delivery classification.

    Supported models:
    - hist_gradient_boosting: stronger non-linear tree-based model used as the
      primary production artifact.
    - logistic_regression: linear baseline for transparent model comparison.
    """
    if model_name == "hist_gradient_boosting":
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                ),
            ]
        )

        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.1,
            class_weight=None,
            random_state=RANDOM_STATE,
        )

    elif model_name == "logistic_regression":
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=20,
                    ),
                ),
            ]
        )

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def find_best_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
) -> tuple[float, dict, pd.DataFrame]:
    """Choose a decision threshold on validation data by maximizing F1.

    Optional precision/recall floors make the threshold choice more business-aware.
    If no threshold satisfies the floors, the function falls back to best F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if len(thresholds) == 0:
        empty_report = pd.DataFrame(
            columns=["threshold", "precision", "recall", "f1", "meets_constraints"]
        )
        return (
            0.5,
            {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "min_precision": float(min_precision),
                "min_recall": float(min_recall),
                "constraints_satisfied": False,
            },
            empty_report,
        )

    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)

    threshold_report = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
            "f1": f1_scores[:-1],
        }
    )
    threshold_report["meets_constraints"] = (
        (threshold_report["precision"] >= min_precision)
        & (threshold_report["recall"] >= min_recall)
    )

    candidates = threshold_report[threshold_report["meets_constraints"]]
    if candidates.empty:
        candidates = threshold_report

    best_idx = int(candidates["f1"].idxmax())
    best_threshold = float(thresholds[best_idx])

    return (
        best_threshold,
        {
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1_scores[best_idx]),
            "min_precision": float(min_precision),
            "min_recall": float(min_recall),
            "constraints_satisfied": bool(
                threshold_report.loc[best_idx, "meets_constraints"]
            ),
        },
        threshold_report,
    )


def evaluate_split(name: str, y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print(f"\n{name} metrics")
    print("-" * (len(name) + 8))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return metrics


def flatten_metrics(model_name: str, threshold: float, val_metrics: dict, test_metrics: dict) -> dict:
    """Return one CSV-friendly row with key validation and test metrics."""
    row = {
        "model_name": model_name,
        "threshold": float(threshold),
    }
    for split_name, metrics in [("validation", val_metrics), ("test", test_metrics)]:
        for metric_name in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
            row[f"{split_name}_{metric_name}"] = metrics[metric_name]
    return row


def evaluate_candidate_model(
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_precision: float,
    min_recall: float,
) -> dict:
    """Train, tune threshold, refit, and evaluate one candidate model."""
    print(f"\nTraining candidate model: {model_name}")
    clf = make_pipeline(numeric_features, categorical_features, model_name=model_name)
    clf.fit(X_train, y_train)

    val_proba = clf.predict_proba(X_val)[:, 1]
    threshold, threshold_metrics, threshold_report = find_best_threshold(
        y_val,
        val_proba,
        min_precision=min_precision,
        min_recall=min_recall,
    )

    print(f"\n{model_name} selected threshold from validation set: {threshold:.4f}")
    print(f"{model_name} validation metrics at selected threshold:", threshold_metrics)

    val_metrics = evaluate_split(f"{model_name} Validation", y_val, val_proba, threshold)

    final_clf = make_pipeline(numeric_features, categorical_features, model_name=model_name)
    final_clf.fit(X_train_full, y_train_full)

    final_test_proba = final_clf.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_split(f"{model_name} Test", y_test, final_test_proba, threshold)

    return {
        "model_name": model_name,
        "model": final_clf,
        "threshold": threshold,
        "threshold_metrics": threshold_metrics,
        "threshold_report": threshold_report,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_probabilities": val_proba,
        "test_probabilities": final_test_proba,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Olist late-delivery model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the modelling dataset CSV.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.0,
        help="Optional minimum validation precision for threshold selection.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.0,
        help="Optional minimum validation recall for threshold selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset if args.dataset.is_absolute() else PROJECT_DIR / args.dataset

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run make_dataset_v2.py first."
        )

    df = pd.read_csv(dataset_path)
    target_col = "is_late"

    print("Dataset shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in {dataset_path}")

    numeric_features, categorical_features = get_feature_columns(df, target_col)
    feature_cols = numeric_features + categorical_features

    X = df[feature_cols]
    y = df[target_col].astype(int)

    print("\nTarget distribution:")
    print(y.value_counts())
    print(y.value_counts(normalize=True))
    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # 60/20/20 split: validation is used only for threshold tuning,
    # while test remains untouched until the final evaluation.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    print("\nTrain shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    candidate_model_names = ["hist_gradient_boosting", "logistic_regression"]
    comparison_results = []

    for model_name in candidate_model_names:
        comparison_results.append(
            evaluate_candidate_model(
                model_name=model_name,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                X_test=X_test,
                y_test=y_test,
                min_precision=args.min_precision,
                min_recall=args.min_recall,
            )
        )

    primary_result = comparison_results[0]
    threshold = primary_result["threshold"]
    threshold_metrics = primary_result["threshold_metrics"]
    threshold_report = primary_result["threshold_report"]
    threshold_report.to_csv(THRESHOLD_REPORT_PATH, index=False)

    comparison_rows = [
        flatten_metrics(
            result["model_name"],
            result["threshold"],
            result["validation_metrics"],
            result["test_metrics"],
        )
        for result in comparison_results
    ]
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by="test_pr_auc", ascending=False
    )
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\nThreshold report saved as: {THRESHOLD_REPORT_PATH}")
    print(f"Model comparison saved as: {MODEL_COMPARISON_PATH}")
    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))

    # Keep the original gradient boosting model as the final production artifact.
    final_clf = primary_result["model"]
    final_test_proba = primary_result["test_probabilities"]
    val_metrics = primary_result["validation_metrics"]
    test_metrics = primary_result["test_metrics"]

    joblib.dump(final_clf, MODEL_PATH)
    print(f"\nModel saved as: {MODEL_PATH}")

    predictions = X_test.copy()
    predictions["actual_is_late"] = y_test.values
    predictions["late_probability"] = final_test_proba
    predictions["predicted_is_late"] = (final_test_proba >= threshold).astype(int)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved as: {PREDICTIONS_PATH}")

    metadata = {
        "model_file": str(MODEL_PATH),
        "dataset": str(dataset_path),
        "model_type": "HistGradientBoostingClassifier",
        "comparison_model_types": ["HistGradientBoostingClassifier", "LogisticRegression"],
        "threshold_selection": "max_f1_on_validation_set",
        "threshold": threshold,
        "threshold_report": str(THRESHOLD_REPORT_PATH),
        "model_comparison": str(MODEL_COMPARISON_PATH),
        "primary_metrics": ["pr_auc", "f1", "precision", "recall", "roc_auc"],
        "validation_threshold_metrics": threshold_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "comparison_results": {
            result["model_name"]: {
                "threshold": result["threshold"],
                "validation_threshold_metrics": result["threshold_metrics"],
                "validation_metrics": result["validation_metrics"],
                "test_metrics": result["test_metrics"],
            }
            for result in comparison_results
        },
        "target_distribution": y.value_counts().to_dict(),
        "feature_columns": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Metadata saved as: {METADATA_PATH}")


if __name__ == "__main__":
    main()