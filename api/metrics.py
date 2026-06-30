from __future__ import annotations

from typing import Any

import csv

from dependencies import PREDICTIONS_PATH, load_metadata


def metrics_summary() -> dict[str, Any]:
    """Return model evaluation metrics and optional prediction-file summary."""
    metadata = load_metadata()
    summary: dict[str, Any] = {
        "model_type": metadata.get("model_type"),
        "threshold": metadata.get("threshold"),
        "primary_metrics": metadata.get("primary_metrics", []),
        "validation_metrics": metadata.get("validation_metrics", {}),
        "test_metrics": metadata.get("test_metrics", {}),
        "target_distribution": metadata.get("target_distribution", {}),
    }

    if PREDICTIONS_PATH.exists():
        total = 0
        predicted_late = 0
        probability_sum = 0.0
        high_risk = 0

        with PREDICTIONS_PATH.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                total += 1
                probability = float(row.get("late_probability") or 0.0)
                probability_sum += probability
                predicted_late += int(float(row.get("predicted_is_late") or 0.0) >= 1)
                high_risk += int(probability >= 0.7)

        summary["prediction_file_summary"] = {
            "file": str(PREDICTIONS_PATH),
            "rows": total,
            "predicted_late_count": predicted_late,
            "predicted_late_rate": predicted_late / total if total else 0.0,
            "average_late_probability": probability_sum / total if total else 0.0,
            "high_risk_count": high_risk,
        }
    else:
        summary["prediction_file_summary"] = {
            "file": str(PREDICTIONS_PATH),
            "available": False,
        }

    return summary