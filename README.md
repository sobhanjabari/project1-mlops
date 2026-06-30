# Olist Late Delivery Prediction

A compact machine-learning project for predicting whether an Olist e-commerce order is likely to be delivered late. The repository includes data-preparation scripts, model training, threshold tuning, evaluation outputs, and exploratory notebooks.

## Project overview

Late deliveries can reduce customer satisfaction and increase operational costs. This project turns the public Olist order datasets into a supervised learning dataset and trains a scikit-learn classification pipeline to estimate late-delivery risk before the final delivery outcome is known.

## Main files

- `make_dataset_v2.py` — builds the modelling dataset from the raw Olist CSV files.
- `train_model.py` — trains and evaluates a `HistGradientBoostingClassifier` pipeline.
- `tune_threshold.py` — evaluates precision/recall/F1 across thresholds from `predictions.csv`.
- `main.py` — FastAPI application exposing health, model-info, prediction, and metrics endpoints.
- `olist_late_delivery_analysis.ipynb` — exploratory notebook.
- `model_metadata.json` — saved model metrics and training metadata.
- `late_delivery_model.joblib` — trained scikit-learn pipeline artifact.

## Repository structure

```text
.
├── make_dataset_v2.py                 # Feature engineering / modelling dataset builder
├── train_model.py                     # Model training, validation threshold selection, evaluation
├── tune_threshold.py                  # Threshold analysis from saved predictions
├── main.py                            # FastAPI app entrypoint
├── api/                               # API routes and metrics summary helpers
├── dependencies.py                    # Shared model/metadata loading helpers
├── models.py                          # Pydantic request/response schemas
├── check_data.py                      # Quick dataset sanity checks
├── olist_late_delivery_analysis.ipynb # Exploratory analysis notebook
├── model_metadata.json                # Latest saved metrics/metadata
├── requirements.txt                   # Python dependencies
└── README.md
```

Large generated artifacts and local environments are intentionally excluded from Git via `.gitignore`. If needed, regenerate them with the workflow below.

## Getting started

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Recommended workflow

Run commands from the project folder:

```bash
cd C:\Users\sobhan\Downloads\project1-mlops
```

Build the dataset:

```bash
python make_dataset_v2.py
```

Train the model:

```bash
python train_model.py
```

Optionally tune threshold with business constraints. For example, maximize F1 among thresholds with validation recall >= 0.50:

```bash
python train_model.py --min-recall 0.50
```

Inspect threshold trade-offs from saved test predictions:

```bash
python tune_threshold.py --start 0.05 --stop 0.95 --step 0.01
```

## FastAPI service

Start the API locally:

```bash
uvicorn main:app --reload
```

Required endpoints:

- `GET /health` — service and artifact health check.
- `GET /model-info` — model type/version, threshold, and required feature columns.
- `POST /predict` — single-order late-delivery prediction.
- `GET /metrics` — validation/test metrics plus a summary of `predictions.csv` when available.

Additional endpoint:

- `POST /predict/batch` — batch prediction with aggregate risk summary.

Example single prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "sample-order-001",
    "features": {
      "purchase_hour": 13,
      "purchase_dayofweek": 4,
      "purchase_month": 6,
      "is_weekend_purchase": 0,
      "estimated_delivery_days": 23.45,
      "has_approval_timestamp": 1,
      "approval_delay_hours": 0.19,
      "num_items": 1,
      "total_price": 99.99,
      "avg_price": 99.99,
      "max_price": 99.99,
      "total_freight": 17.95,
      "avg_freight": 17.95,
      "max_freight": 17.95,
      "num_sellers": 1,
      "num_products": 1,
      "num_product_categories": 1,
      "avg_product_weight_g": 1200,
      "max_product_weight_g": 1200,
      "avg_product_volume_cm3": 15750,
      "max_product_volume_cm3": 15750,
      "seller_zip_code_prefix": 3426,
      "num_seller_states": 1,
      "freight_price_ratio": 0.1795,
      "payment_value": 117.94,
      "payment_installments": 6,
      "num_payment_types": 2,
      "customer_zip_code_prefix": 23932,
      "same_state": 0,
      "same_city": 0,
      "zip_prefix_diff": 20506,
      "main_seller_city": "sao paulo",
      "main_seller_state": "SP",
      "main_product_category": "cool_stuff",
      "main_payment_type": "credit_card",
      "customer_city": "angra dos reis",
      "customer_state": "RJ"
    }
  }'
```

Prediction responses include `order_id`, `late_probability`, `risk_level`, `predicted_is_late`, `model_name`, model version, a short explanation, and a recommended action.

## Outputs

- `model_dataset_v2.csv`
- `late_delivery_model.joblib`
- `predictions.csv`
- `model_metadata.json`
- `threshold_report.csv`
- `threshold_tuning_from_predictions.csv`

## Model approach

- **Target:** binary late-delivery indicator.
- **Estimator:** `HistGradientBoostingClassifier` inside a scikit-learn `Pipeline`.
- **Preprocessing:** median imputation for numeric columns and ordinal encoding for categorical columns.
- **Evaluation:** ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, and threshold trade-off reports.
- **Thresholding:** decision threshold is selected on the validation set, preserving the test set for final evaluation.

## Notes

- The scripts now use paths relative to their own location, so they are safer to run from different working directories.
- `train_model.py` keeps the test set untouched until final evaluation and stores validation/test metrics in metadata.
- Threshold selection is done on the validation set, not the test set, to reduce evaluation bias.

## License

This project is licensed under the MIT License. See `LICENSE` for details.