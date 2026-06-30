from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from dependencies import get_model_name, get_threshold, load_metadata, load_model
from models import BatchPredictionRequest, BatchPredictionResponse, PredictionRequest, PredictionResponse


router = APIRouter(tags=["prediction"])


def risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def recommendation(level: str) -> str:
    actions = {
        "high": "Prioritize fulfilment, contact the seller/logistics partner, and proactively notify the customer.",
        "medium": "Monitor shipment milestones and prepare customer support follow-up if tracking is delayed.",
        "low": "Continue standard fulfilment and monitoring.",
    }
    return actions[level]


def explanation(probability: float, level: str, threshold: float) -> str:
    return (
        f"Estimated late-delivery probability is {probability:.1%}. "
        f"This is classified as {level} risk using decision threshold {threshold:.3f}."
    )


def dataframe_from_requests(requests: list[PredictionRequest]) -> pd.DataFrame:
    metadata = load_metadata()
    feature_columns = metadata.get("feature_columns", [])
    if not feature_columns:
        raise HTTPException(status_code=500, detail="feature_columns missing from model metadata")

    rows: list[dict[str, Any]] = []
    missing_by_order: dict[str, list[str]] = {}

    for request in requests:
        missing = [column for column in feature_columns if column not in request.features]
        if missing:
            missing_by_order[request.order_id] = missing
        rows.append({column: request.features.get(column) for column in feature_columns})

    if missing_by_order:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required model feature(s).",
                "missing_features_by_order": missing_by_order,
                "required_features": feature_columns,
            },
        )

    return pd.DataFrame(rows, columns=feature_columns)


def predict_many(requests: list[PredictionRequest]) -> list[PredictionResponse]:
    model = load_model()
    metadata = load_metadata()
    threshold = get_threshold()
    model_name = get_model_name()
    model_version = metadata.get("trained_at_utc")

    frame = dataframe_from_requests(requests)
    try:
        probabilities = model.predict_proba(frame)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    responses: list[PredictionResponse] = []
    for request, probability in zip(requests, probabilities):
        probability_float = float(probability)
        level = risk_level(probability_float)
        responses.append(
            PredictionResponse(
                order_id=request.order_id,
                late_probability=probability_float,
                risk_level=level,
                predicted_is_late=probability_float >= threshold,
                model_name=model_name,
                model_version=str(model_version) if model_version else None,
                explanation=explanation(probability_float, level, threshold),
                recommended_action=recommendation(level),
            )
        )
    return responses


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    return predict_many([request])[0]


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = predict_many(request.orders)
    total = len(predictions)
    summary = {
        "total_orders": total,
        "predicted_late_count": sum(pred.predicted_is_late for pred in predictions),
        "average_late_probability": sum(pred.late_probability for pred in predictions) / total,
        "risk_level_counts": {
            "low": sum(pred.risk_level == "low" for pred in predictions),
            "medium": sum(pred.risk_level == "medium" for pred in predictions),
            "high": sum(pred.risk_level == "high" for pred in predictions),
        },
    }
    return BatchPredictionResponse(predictions=predictions, summary=summary)