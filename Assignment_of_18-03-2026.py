# Customer Segmentation using K-Means Clustering
# Dataset: Mall Customers (Age, Annual Income, Spending Score)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1. Create / Load Dataset
#    Using a built-in sample that mirrors the
#    popular "Mall_Customers.csv" from Kaggle.
# ─────────────────────────────────────────────
np.random.seed(42)

# To use the real Kaggle CSV instead, replace this block with:
#   df = pd.read_csv("Mall_Customers.csv")
#   df.rename(columns={"Annual Income (k$)": "Income",
#                       "Spending Score (1-100)": "Score"}, inplace=True)

n = 200
age      = np.concatenate([np.random.normal(25, 4, 40),   # young
                            np.random.normal(35, 5, 50),   # mid-young
                            np.random.normal(45, 6, 60),   # middle-aged
                            np.random.normal(55, 5, 30),   # senior
                            np.random.normal(32, 4, 20)])  # mixed

income   = np.concatenate([np.random.normal(30,  5, 40),  # low
                            np.random.normal(55,  8, 50),  # medium
                            np.random.normal(85, 10, 60),  # high
                            np.random.normal(40,  6, 30),  # low-mid
                            np.random.normal(70,  8, 20)]) # high-mid

score    = np.concatenate([np.random.normal(75, 8, 40),   # high spenders
                            np.random.normal(50, 10, 50),  # average
                            np.random.normal(25, 8, 60),   # low spenders
                            np.random.normal(80, 6, 30),   # impulsive
                            np.random.normal(45, 10, 20)]) # cautious

gender   = np.random.choice(["Male", "Female"], n)

df = pd.DataFrame({
    "CustomerID"  : range(1, n + 1),
    "Gender"      : gender,
    "Age"         : age.clip(18, 70).astype(int),
    "Income"      : income.clip(15, 120).astype(int),
    "Score"       : score.clip(1, 100).astype(int),
})

print("=" * 55)
print("   CUSTOMER SEGMENTATION — K-Means Clustering")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print(df.describe().round(1).to_string())

# ─────────────────────────────────────────────
# 2. Feature Selection & Scaling
# ─────────────────────────────────────────────
features = ["Age", "Income", "Score"]
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 3. Elbow Method — find optimal K
# ─────────────────────────────────────────────
inertias   = []
sil_scores = []
K_range    = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

OPTIMAL_K = 5   # chosen from elbow + silhouette

# ─────────────────────────────────────────────
# 4. Final K-Means Model
# ─────────────────────────────────────────────
km_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
df["Cluster"] = km_final.fit_predict(X_scaled)

# ─────────────────────────────────────────────
# 5. Describe Each Cluster
# ─────────────────────────────────────────────
cluster_labels = {
    0: "Young High Spenders",
    1: "Middle-Aged Savers",
    2: "High Income Low Spend",
    3: "Senior Cautious",
    4: "Mid Income Balanced",
}

summary = df.groupby("Cluster")[features].mean().round(1)
summary["Count"]       = df.groupby("Cluster").size()
summary["Description"] = summary.index.map(cluster_labels)
print("\n── Cluster Summary ──────────────────────────────────")
print(summary.to_string())
print()

# ─────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────
COLORS = ["#E85D24", "#185FA5", "#1D9E75", "#BA7517", "#7F77DD"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor("#F8F9FA")
fig.suptitle("Customer Segmentation — K-Means Clustering",
             fontsize=17, fontweight="bold", color="#2C2C2A", y=0.98)

# ── Plot 1: Elbow curve
ax = axes[0, 0]
ax.set_facecolor("#F8F9FA")
ax.plot(list(K_range), inertias, "o-", color="#185FA5", linewidth=2, markersize=7)
ax.axvline(OPTIMAL_K, color="#E85D24", linestyle="--", linewidth=1.5,
           label=f"Optimal K = {OPTIMAL_K}")
ax.set_title("Elbow Method", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia (WCSS)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
for spine in ax.spines.values():
    spine.set_visible(False)

# ── Plot 2: Silhouette scores
ax = axes[0, 1]
ax.set_facecolor("#F8F9FA")
ax.plot(list(K_range), sil_scores, "s-", color="#1D9E75", linewidth=2, markersize=7)
ax.axvline(OPTIMAL_K, color="#E85D24", linestyle="--", linewidth=1.5,
           label=f"Optimal K = {OPTIMAL_K}")
ax.set_title("Silhouette Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Silhouette Score")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
for spine in ax.spines.values():
    spine.set_visible(False)

# ── Plot 3: Income vs Spending Score (main cluster view)
ax = axes[1, 0]
ax.set_facecolor("#F8F9FA")
for c in range(OPTIMAL_K):
    mask = df["Cluster"] == c
    ax.scatter(df.loc[mask, "Income"], df.loc[mask, "Score"],
               color=COLORS[c], alpha=0.7, s=60, label=cluster_labels[c],
               edgecolors="white", linewidths=0.5)
centers = scaler.inverse_transform(km_final.cluster_centers_)
ax.scatter(centers[:, 1], centers[:, 2],
           c="black", marker="X", s=200, zorder=5, label="Centroids")
ax.set_title("Income vs Spending Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1–100)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
for spine in ax.spines.values():
    spine.set_visible(False)

# ── Plot 4: Age vs Income
ax = axes[1, 1]
ax.set_facecolor("#F8F9FA")
for c in range(OPTIMAL_K):
    mask = df["Cluster"] == c
    ax.scatter(df.loc[mask, "Age"], df.loc[mask, "Income"],
               color=COLORS[c], alpha=0.7, s=60, label=cluster_labels[c],
               edgecolors="white", linewidths=0.5)
ax.scatter(centers[:, 0], centers[:, 1],
           c="black", marker="X", s=200, zorder=5, label="Centroids")
ax.set_title("Age vs Annual Income", fontsize=13, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(OUTPUT_DIR, "customer_segmentation.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"Plot saved to: {out_path}")

# ─────────────────────────────────────────────
# 7. Predict segment for a new customer
# ─────────────────────────────────────────────
def predict_segment(age, income, score):
    sample = scaler.transform([[age, income, score]])
    cluster = km_final.predict(sample)[0]
    print(f"\nNew Customer  →  Age: {age}, Income: {income}k$, Score: {score}")
    print(f"Predicted Segment : Cluster {cluster} — {cluster_labels[cluster]}")

# Example predictions
predict_segment(age=28, income=30, score=82)
predict_segment(age=50, income=90, score=20)
predict_segment(age=35, income=60, score=55)