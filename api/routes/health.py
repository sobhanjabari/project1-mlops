from fastapi import APIRouter

from dependencies import METADATA_PATH, MODEL_PATH


router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "olist-late-delivery-prediction",
        "model_artifact_exists": MODEL_PATH.exists(),
        "metadata_exists": METADATA_PATH.exists(),
    }