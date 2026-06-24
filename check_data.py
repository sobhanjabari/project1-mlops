import pandas as pd
from pathlib import Path

DATA_DIR = Path(".")

files = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "product_category_name_translation.csv",
]

for file in files:
    path = DATA_DIR / file
    print("=" * 80)
    print(file)

    if not path.exists():
        print("File not found:", path)
        continue

    df = pd.read_csv(path)
    print("shape:", df.shape)
    print(df.head())