"""
visualize.py
------------
Produces four publication-quality charts:

  1. Customer Segments Visualization  — 2D scatter of clusters (PCA reduced)
  2. RFM Score Heatmap                — segment x RFM dimension heatmap
  3. Churn Prediction Results         — ROC curve + feature importance bar
  4. ROI / Retention Metrics Dashboard — segment-level revenue and churn risk

All charts saved to visuals/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE  = {
    "Champion":  "#1F3864",
    "Loyal":     "#2E75B6",
    "Potential": "#70AD47",
    "At Risk":   "#ED7D31",
    "Lost":      "#C00000",
}
BG      = "#F7F9FC"
GRID    = "#E0E6EF"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    BG,
    "figure.facecolor":  "white",
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

OUTPUT_DIR = "visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load():
    rfm     = pd.read_csv("outputs/rfm_clustered.csv")
    summary = pd.read_csv("outputs/cluster_summary.csv")
    churn   = pd.read_csv("outputs/churn_predictions.csv")
    fi      = pd.read_csv("outputs/feature_importances.csv")
    return rfm, summary, churn, fi


# ── Chart 1: Customer Segments (PCA 2D) ──────────────────────────────────────
def plot_segments(rfm):
    features = ["recency", "frequency", "monetary"]
    X_scaled = StandardScaler().fit_transform(rfm[features])
    coords   = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    rfm = rfm.copy()
    rfm["pc1"] = coords[:, 0]
    rfm["pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(11, 8))
    segments = rfm["cluster_label"].unique()

    for seg in segments:
        sub = rfm[rfm["cluster_label"] == seg]
        color = PALETTE.get(seg, "#888888")
        ax.scatter(sub["pc1"], sub["pc2"], c=color, label=seg,
                   alpha=0.55, s=20, edgecolors="none")

    ax.set_title("Customer Segmentation — PCA Projection of RFM Space",
                 fontsize=14, fontweight="bold", color="#1F3864")
    ax.set_xlabel("Principal Component 1", fontsize=11)
    ax.set_ylabel("Principal Component 2", fontsize=11)
    legend = ax.legend(title="Segment", fontsize=10, title_fontsize=11,
                       loc="upper right", framealpha=0.9)

    # Segment count annotations
    counts = rfm["cluster_label"].value_counts()
    for seg in segments:
        sub   = rfm[rfm["cluster_label"] == seg]
        cx    = sub["pc1"].mean()
        cy    = sub["pc2"].mean()
        color = PALETTE.get(seg, "#888888")
        ax.annotate(f"{seg}\n(n={counts.get(seg,0):,})",
                    xy=(cx, cy), fontsize=8, color=color,
                    fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/1_customer_segments.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Chart 2: RFM Score Heatmap ────────────────────────────────────────────────
def plot_rfm_heatmap(rfm):
    pivot = rfm.groupby("cluster_label")[["r_score", "f_score", "m_score", "rfm_score"]].mean().round(2)
    pivot.columns = ["Recency Score", "Frequency Score", "Monetary Score", "Total RFM Score"]
    order = [s for s in ["Champion", "Loyal", "Potential", "At Risk", "Lost"] if s in pivot.index]
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="#CCCCCC",
        cbar_kws={"label": "Average Score", "shrink": 0.8},
        vmin=1, vmax=5
    )
    ax.set_title("RFM Score Heatmap by Customer Segment",
                 fontsize=14, fontweight="bold", color="#1F3864")
    ax.set_xlabel("RFM Dimension", fontsize=11)
    ax.set_ylabel("Segment", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right", fontsize=10)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/2_rfm_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Chart 3: Churn Model — ROC + Feature Importance ──────────────────────────
def plot_churn_results(churn, fi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(churn["churned"], churn["churn_prob"])
    roc_auc     = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#1F3864", lw=2.2,
             label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax1.plot([0,1],[0,1], "k--", lw=1.2, alpha=0.5, label="Random classifier")
    ax1.fill_between(fpr, tpr, alpha=0.08, color="#1F3864")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curve — Churn Prediction Model",
                  fontsize=13, fontweight="bold", color="#1F3864")
    ax1.legend(fontsize=10)

    # Feature Importance
    fi_sorted = fi.sort_values("importance", ascending=True).tail(8)
    colors    = ["#1F3864" if i == len(fi_sorted)-1 else "#2E75B6"
                 for i in range(len(fi_sorted))]
    ax2.barh(fi_sorted["feature"], fi_sorted["importance"],
             color=colors, edgecolor="white", height=0.6)
    ax2.set_xlabel("Feature Importance (XGBoost)", fontsize=11)
    ax2.set_title("Top Feature Importances — Churn Model",
                  fontsize=13, fontweight="bold", color="#1F3864")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/3_churn_model.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Chart 4: ROI / Retention Metrics Dashboard ───────────────────────────────
def plot_roi_dashboard(rfm, churn):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 4a — Revenue by segment (bar)
    ax = axes[0, 0]
    rev = rfm.groupby("cluster_label")["monetary"].sum().sort_values(ascending=False)
    colors = [PALETTE.get(s, "#888") for s in rev.index]
    rev.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.6)
    ax.set_title("Total Revenue by Segment", fontsize=12,
                 fontweight="bold", color="#1F3864")
    ax.set_xlabel("")
    ax.set_ylabel("Total Revenue ($)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    # 4b — Customer count by segment (donut)
    ax = axes[0, 1]
    counts = rfm["cluster_label"].value_counts()
    wedge_colors = [PALETTE.get(s, "#888") for s in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=wedge_colors, startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9}
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("Customer Distribution by Segment", fontsize=12,
                 fontweight="bold", color="#1F3864")

    # 4c — Churn rate by segment
    ax = axes[1, 0]
    churn_rate = (rfm.groupby("cluster_label")["churned"].mean() * 100).sort_values(ascending=False)
    colors     = [PALETTE.get(s, "#888") for s in churn_rate.index]
    churn_rate.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.6)
    ax.set_title("Churn Rate by Segment (%)", fontsize=12,
                 fontweight="bold", color="#1F3864")
    ax.set_xlabel("")
    ax.set_ylabel("Churn Rate (%)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    # 4d — Avg monetary value by segment
    ax = axes[1, 1]
    avg_val = rfm.groupby("cluster_label")["monetary"].mean().sort_values(ascending=False)
    colors  = [PALETTE.get(s, "#888") for s in avg_val.index]
    avg_val.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.6)
    ax.set_title("Avg Customer Lifetime Value by Segment", fontsize=12,
                 fontweight="bold", color="#1F3864")
    ax.set_xlabel("")
    ax.set_ylabel("Avg Revenue ($)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    fig.suptitle("Retention & ROI Metrics Dashboard",
                 fontsize=16, fontweight="bold", color="#1F3864", y=1.01)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/4_roi_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rfm, summary, churn, fi = load()
    print("\nGenerating visualizations...\n")
    plot_segments(rfm)
    plot_rfm_heatmap(rfm)
    plot_churn_results(churn, fi)
    plot_roi_dashboard(rfm, churn)
    print("\nAll charts saved to visuals/")
