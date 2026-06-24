import pandas as pd
from pathlib import Path

DATA_DIR = Path(".")

orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")

date_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

orders = orders.dropna(subset=[
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "order_purchase_timestamp",
])

orders["is_late"] = (
    orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
).astype(int)

orders["purchase_hour"] = orders["order_purchase_timestamp"].dt.hour
orders["purchase_dayofweek"] = orders["order_purchase_timestamp"].dt.dayofweek
orders["purchase_month"] = orders["order_purchase_timestamp"].dt.month

orders["estimated_delivery_days"] = (
    orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]
).dt.total_seconds() / (3600 * 24)

items_agg = items.groupby("order_id").agg(
    num_items=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    num_sellers=("seller_id", "nunique"),
    num_products=("product_id", "nunique"),
).reset_index()

payments_agg = payments.groupby("order_id").agg(
    payment_value=("payment_value", "sum"),
    payment_installments=("payment_installments", "max"),
    num_payment_types=("payment_type", "nunique"),
).reset_index()

df = orders.merge(items_agg, on="order_id", how="left")
df = df.merge(payments_agg, on="order_id", how="left")
df = df.merge(customers, on="customer_id", how="left")

drop_cols = [
    "order_id",
    "customer_id",
    "customer_unique_id",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

model_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

model_df.to_csv("model_dataset.csv", index=False)

print("Saved: model_dataset.csv")
print("Shape:", model_df.shape)
print("Target distribution:")
print(model_df["is_late"].value_counts())
print(model_df["is_late"].value_counts(normalize=True))
print(model_df.head())