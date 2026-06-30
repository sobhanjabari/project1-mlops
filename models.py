from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    order_id: str = Field(..., examples=["sample-order-001"])
    features: dict[str, Any] = Field(
        ...,
        description="Feature dictionary matching model_metadata.json feature_columns.",
    )


class BatchPredictionRequest(BaseModel):
    orders: list[PredictionRequest] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    order_id: str
    late_probability: float
    risk_level: str
    predicted_is_late: bool
    model_name: str
    model_version: str | None = None
    explanation: str
    recommended_action: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: dict[str, Any]