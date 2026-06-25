from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "model_dataset_v2.csv"


def read_csv(name: str) -> pd.DataFrame:
    """Read a project CSV using a script-relative path."""
    path = PROJECT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path)


def mode_or_unknown(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return "unknown"
    return str(series.mode().iloc[0])


def build_dataset() -> pd.DataFrame:
    orders = read_csv("olist_orders_dataset.csv")
    items = read_csv("olist_order_items_dataset.csv")
    payments = read_csv("olist_order_payments_dataset.csv")
    customers = read_csv("olist_customers_dataset.csv")
    products = read_csv("olist_products_dataset.csv")
    sellers = read_csv("olist_sellers_dataset.csv")
    translation = read_csv("product_category_name_translation.csv")

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    orders = orders.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ).copy()

    orders["is_late"] = (
        orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
    ).astype(int)

    orders["purchase_hour"] = orders["order_purchase_timestamp"].dt.hour
    orders["purchase_dayofweek"] = orders["order_purchase_timestamp"].dt.dayofweek
    orders["purchase_month"] = orders["order_purchase_timestamp"].dt.month
    orders["is_weekend_purchase"] = orders["purchase_dayofweek"].isin([5, 6]).astype(int)
    orders["estimated_delivery_days"] = (
        orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / (3600 * 24)
    orders["has_approval_timestamp"] = orders["order_approved_at"].notna().astype(int)

    # This feature is valid if prediction happens after payment approval.
    # For a strict purchase-time model, remove it from train_model.py feature selection.
    orders["approval_delay_hours"] = (
        orders["order_approved_at"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    products = products.merge(translation, on="product_category_name", how="left")
    products["product_volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    )

    items_enriched = items.merge(products, on="product_id", how="left")
    items_enriched = items_enriched.merge(sellers, on="seller_id", how="left")

    items_agg = items_enriched.groupby("order_id").agg(
        num_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        avg_price=("price", "mean"),
        max_price=("price", "max"),
        total_freight=("freight_value", "sum"),
        avg_freight=("freight_value", "mean"),
        max_freight=("freight_value", "max"),
        num_sellers=("seller_id", "nunique"),
        num_products=("product_id", "nunique"),
        num_product_categories=("product_category_name_english", "nunique"),
        avg_product_weight_g=("product_weight_g", "mean"),
        max_product_weight_g=("product_weight_g", "max"),
        avg_product_volume_cm3=("product_volume_cm3", "mean"),
        max_product_volume_cm3=("product_volume_cm3", "max"),
        seller_zip_code_prefix=("seller_zip_code_prefix", "median"),
        num_seller_states=("seller_state", "nunique"),
        main_seller_city=("seller_city", mode_or_unknown),
        main_seller_state=("seller_state", mode_or_unknown),
        main_product_category=("product_category_name_english", mode_or_unknown),
    ).reset_index()

    items_agg["freight_price_ratio"] = (
        items_agg["total_freight"] / items_agg["total_price"].replace(0, np.nan)
    )

    payments_agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        num_payment_types=("payment_type", "nunique"),
        main_payment_type=("payment_type", mode_or_unknown),
    ).reset_index()

    df = orders.merge(items_agg, on="order_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")

    # Geolocation data is not available in this project, so we use robust
    # location-proximity proxies based on customer and seller city/state.
    df["same_state"] = (df["customer_state"] == df["main_seller_state"]).astype(int)
    df["same_city"] = (df["customer_city"] == df["main_seller_city"]).astype(int)
    df["zip_prefix_diff"] = (
        df["customer_zip_code_prefix"] - df["seller_zip_code_prefix"]
    ).abs()

    drop_cols = [
        "order_id",
        "customer_id",
        "customer_unique_id",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        # This changes during fulfilment and leaks post-purchase information.
        "order_status",
    ]

    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def main() -> None:
    model_df = build_dataset()
    model_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print("Shape:", model_df.shape)
    print("Target distribution:")
    print(model_df["is_late"].value_counts())
    print(model_df["is_late"].value_counts(normalize=True))
    print(model_df.head())


if __name__ == "__main__":
    main()