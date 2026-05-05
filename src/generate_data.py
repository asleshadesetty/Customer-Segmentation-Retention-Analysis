"""
generate_data.py
----------------
Loads and preprocesses the Online Retail dataset from Kaggle:
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

The dataset contains transactional records from a UK-based online retailer
(2009-2011). This script:
  1. Loads the raw CSV
  2. Cleans and filters valid transactions
  3. Computes RFM (Recency, Frequency, Monetary) features per customer
  4. Saves cleaned transaction data and RFM table to data/

SETUP:
  1. Download the dataset from Kaggle:
     https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
  2. Place online_retail_II.csv inside the data/ folder
  3. Run: python src/generate_data.py

No Kaggle access? Run with --synthetic flag:
     python src/generate_data.py --synthetic
"""

import argparse
import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

DATA_DIR       = "data"
OUT_TRANS      = f"{DATA_DIR}/transactions.csv"
OUT_RFM        = f"{DATA_DIR}/rfm.csv"
os.makedirs(DATA_DIR, exist_ok=True)


# ── Real data loader ──────────────────────────────────────────────────────────
def load_kaggle(path="data/online_retail_II.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] '{path}' not found.\n"
            "Download online_retail_II.csv from:\n"
            "https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci\n"
            "and place it in the data/ folder.\n"
            "Or run with --synthetic to use generated data instead."
        )

    df = pd.read_csv(path, encoding="ISO-8859-1", parse_dates=["InvoiceDate"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Clean
    df = df[df["customer_id"].notna()]
    df = df[df["quantity"] > 0]
    df = df[df["price"] > 0]
    df = df[~df["invoice"].astype(str).str.startswith("C")]
    df["revenue"] = df["quantity"] * df["price"]
    df["customer_id"] = df["customer_id"].astype(int).astype(str)

    print(f"[INFO] Loaded {len(df):,} transactions from {df['customer_id'].nunique():,} customers.")
    return df


# ── Synthetic fallback ─────────────────────────────────────────────────────────
def generate_synthetic(n_customers=2000, n_transactions=25000):
    print("[INFO] Generating synthetic retail transaction dataset.")
    dates      = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    cust_ids   = [f"C{str(i).zfill(5)}" for i in range(1, n_customers + 1)]

    # Assign customer types: Champions, Loyal, At Risk, Lost
    types = np.random.choice(
        ["Champion", "Loyal", "At Risk", "Lost"],
        size=n_customers,
        p=[0.15, 0.35, 0.30, 0.20]
    )
    type_map = dict(zip(cust_ids, types))

    records = []
    for _ in range(n_transactions):
        cid   = np.random.choice(cust_ids)
        ctype = type_map[cid]

        if ctype == "Champion":
            date    = np.random.choice(dates[-90:])
            qty     = np.random.randint(3, 20)
            price   = round(np.random.uniform(10, 80), 2)
        elif ctype == "Loyal":
            date    = np.random.choice(dates[-180:])
            qty     = np.random.randint(2, 10)
            price   = round(np.random.uniform(8, 50), 2)
        elif ctype == "At Risk":
            date    = np.random.choice(dates[-365:])
            qty     = np.random.randint(1, 5)
            price   = round(np.random.uniform(5, 30), 2)
        else:  # Lost
            date    = np.random.choice(dates[:-300])
            qty     = np.random.randint(1, 3)
            price   = round(np.random.uniform(5, 20), 2)

        records.append({
            "invoice":      f"INV{np.random.randint(100000,999999)}",
            "invoicedate":  pd.Timestamp(date),
            "customer_id":  cid,
            "quantity":     qty,
            "price":        price,
            "revenue":      round(qty * price, 2),
        })

    df = pd.DataFrame(records)
    print(f"[INFO] Generated {len(df):,} transactions from {df['customer_id'].nunique():,} customers.")
    return df


# ── RFM computation ────────────────────────────────────────────────────────────
def compute_rfm(df):
    snapshot = df["invoicedate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency   = ("invoicedate",  lambda x: (snapshot - x.max()).days),
        frequency = ("invoice",      "nunique"),
        monetary  = ("revenue",      "sum"),
    ).reset_index()

    # Score each dimension 1-5
    rfm["r_score"] = pd.qcut(rfm["recency"],   5, labels=[5,4,3,2,1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"],  5, labels=[1,2,3,4,5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # Segment labels
    def segment(row):
        if row["rfm_score"] >= 13:
            return "Champion"
        elif row["rfm_score"] >= 10:
            return "Loyal"
        elif row["rfm_score"] >= 7:
            return "Potential"
        elif row["rfm_score"] >= 5:
            return "At Risk"
        else:
            return "Lost"

    rfm["segment"] = rfm.apply(segment, axis=1)

    # Churn label: churned if recency > 180 days
    rfm["churned"] = (rfm["recency"] > 180).astype(int)

    return rfm


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of Kaggle dataset")
    args = parser.parse_args()

    if args.synthetic:
        df = generate_synthetic()
    else:
        try:
            df = load_kaggle()
        except FileNotFoundError as e:
            print(e)
            exit(1)

    rfm = compute_rfm(df)

    df.to_csv(OUT_TRANS, index=False)
    rfm.to_csv(OUT_RFM, index=False)

    print(f"[INFO] Transactions saved to {OUT_TRANS}  |  Shape: {df.shape}")
    print(f"[INFO] RFM table saved to    {OUT_RFM}    |  Shape: {rfm.shape}")
    print(f"\nSegment distribution:\n{rfm['segment'].value_counts().to_string()}")
