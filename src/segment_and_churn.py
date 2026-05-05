"""
segment_and_churn.py
--------------------
Two-stage modelling pipeline:

Stage 1 — Customer Segmentation
  - Scales RFM features
  - Uses Elbow Method to select optimal k
  - Fits K-Means clustering
  - Maps clusters to business segment labels
  - Saves cluster assignments to outputs/

Stage 2 — Churn Prediction
  - Trains an XGBoost classifier on RFM + cluster features
  - Evaluates with precision, recall, F1, ROC-AUC
  - Saves churn predictions to outputs/
  - Reports feature importances

Metrics reported:
  - Clustering: Silhouette Score, Inertia per k
  - Churn:      Precision, Recall, F1, ROC-AUC
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, classification_report,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

SEED       = 42
DATA_PATH  = "data/rfm.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_RANGE    = range(2, 9)
BEST_K     = 5   # Updated by elbow analysis if needed


# ── Stage 1: K-Means Segmentation ────────────────────────────────────────────
def run_segmentation(rfm: pd.DataFrame):
    print("\n" + "="*60)
    print("  STAGE 1: K-Means Customer Segmentation")
    print("="*60)

    features = ["recency", "frequency", "monetary"]
    X        = rfm[features].copy()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + Silhouette analysis
    inertias    = []
    silhouettes = []
    print(f"\n{'k':>4} {'Inertia':>12} {'Silhouette':>12}")
    print("-" * 32)
    for k in K_RANGE:
        km  = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, lbl)
        silhouettes.append(sil)
        print(f"{k:>4} {km.inertia_:>12.1f} {sil:>12.4f}")

    # Save elbow data
    elbow_df = pd.DataFrame({
        "k": list(K_RANGE),
        "inertia": inertias,
        "silhouette": silhouettes
    })
    elbow_df.to_csv(f"{OUTPUT_DIR}/elbow_data.csv", index=False)

    # Use best silhouette k
    best_k = list(K_RANGE)[int(np.argmax(silhouettes))]
    print(f"\n[INFO] Best k by silhouette score: {best_k}")

    km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
    rfm["cluster"] = km_final.fit_predict(X_scaled)

    # Label clusters by RFM centroid profile
    centers = pd.DataFrame(
        scaler.inverse_transform(km_final.cluster_centers_),
        columns=features
    )
    centers["cluster"] = range(best_k)

    def label_cluster(row):
        if row["recency"] < centers["recency"].median() and row["monetary"] > centers["monetary"].median():
            return "Champion"
        elif row["recency"] < centers["recency"].median():
            return "Loyal"
        elif row["monetary"] > centers["monetary"].median():
            return "Potential"
        elif row["recency"] > centers["recency"].quantile(0.75):
            return "Lost"
        else:
            return "At Risk"

    centers["label"] = centers.apply(label_cluster, axis=1)
    cluster_map      = dict(zip(centers["cluster"], centers["label"]))
    rfm["cluster_label"] = rfm["cluster"].map(cluster_map)

    # Cluster summary
    summary = rfm.groupby("cluster_label").agg(
        n_customers  = ("customer_id", "count"),
        avg_recency  = ("recency",     "mean"),
        avg_frequency= ("frequency",   "mean"),
        avg_monetary = ("monetary",    "mean"),
    ).round(1).reset_index()

    summary["pct_customers"] = (summary["n_customers"] / summary["n_customers"].sum() * 100).round(1)

    print(f"\nCluster Summary:\n")
    print(summary.to_string(index=False))

    rfm.to_csv(f"{OUTPUT_DIR}/rfm_clustered.csv", index=False)
    summary.to_csv(f"{OUTPUT_DIR}/cluster_summary.csv", index=False)

    print(f"\n[INFO] Clustered RFM saved to {OUTPUT_DIR}/rfm_clustered.csv")
    return rfm, summary


# ── Stage 2: Churn Prediction ─────────────────────────────────────────────────
def run_churn_model(rfm: pd.DataFrame):
    print("\n" + "="*60)
    print("  STAGE 2: Churn Prediction (XGBoost)")
    print("="*60)

    feature_cols = ["recency", "frequency", "monetary",
                    "r_score", "f_score", "m_score", "rfm_score", "cluster"]
    X = rfm[feature_cols]
    y = rfm["churned"]

    print(f"\nChurn rate: {y.mean()*100:.1f}%  ({y.sum()} churned / {len(y)} total)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Cross-validated AUC
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc         = roc_auc_score(y_test, y_pred_prob)

    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Active", "Churned"]))
    print(f"ROC-AUC (test):   {auc:.4f}")
    print(f"ROC-AUC (CV-5):   {cv_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
        index=["Actual Active", "Actual Churned"],
        columns=["Predicted Active", "Predicted Churned"]
    )
    print(f"\nConfusion Matrix:\n{cm_df.to_string()}")

    # Feature importances
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\nFeature Importances:\n{fi.to_string(index=False)}")

    # Save predictions
    rfm_test               = rfm.iloc[X_test.index].copy()
    rfm_test["churn_prob"] = y_pred_prob
    rfm_test["churn_pred"] = y_pred
    rfm_test.to_csv(f"{OUTPUT_DIR}/churn_predictions.csv", index=False)
    fi.to_csv(f"{OUTPUT_DIR}/feature_importances.csv", index=False)

    metrics = {
        "roc_auc_test":    round(auc, 4),
        "roc_auc_cv":      round(cv_auc, 4),
    }
    report = classification_report(y_test, y_pred,
                                   target_names=["Active","Churned"],
                                   output_dict=True)
    metrics["precision_churned"] = round(report["Churned"]["precision"], 4)
    metrics["recall_churned"]    = round(report["Churned"]["recall"], 4)
    metrics["f1_churned"]        = round(report["Churned"]["f1-score"], 4)

    pd.DataFrame([metrics]).to_csv(f"{OUTPUT_DIR}/churn_metrics.csv", index=False)
    print(f"\n[INFO] Churn predictions saved to {OUTPUT_DIR}/churn_predictions.csv")
    return model, fi, metrics


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rfm = pd.read_csv(DATA_PATH)
    rfm, summary  = run_segmentation(rfm)
    model, fi, metrics = run_churn_model(rfm)
    print("\n" + "="*60)
    print("  Pipeline complete. Outputs saved to outputs/")
    print("="*60)
