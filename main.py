from fastapi import FastAPI

from api.metrics import metrics_summary
from api.routes.health import router as health_router
from api.routes.predict import router as predict_router
from dependencies import load_metadata


app = FastAPI(
    title="Olist Late Delivery Prediction API",
    description="FastAPI service for predicting late-delivery risk for Olist orders.",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(predict_router)


@app.get("/model-info", tags=["model"])
def model_info() -> dict[str, object]:
    metadata = load_metadata()
    return {
        "model_name": metadata.get("model_type", "late_delivery_model"),
        "model_file": metadata.get("model_file"),
        "model_version": metadata.get("trained_at_utc"),
        "threshold": metadata.get("threshold"),
        "threshold_selection": metadata.get("threshold_selection"),
        "sklearn_version": metadata.get("sklearn_version"),
        "feature_columns": metadata.get("feature_columns", []),
        "numeric_features": metadata.get("numeric_features", []),
        "categorical_features": metadata.get("categorical_features", []),
    }


@app.get("/metrics", tags=["metrics"])
def metrics() -> dict[str, object]:
    return metrics_summary()