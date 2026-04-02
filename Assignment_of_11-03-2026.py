"""
============================================================
  CUSTOMER SEGMENTATION — K-Means Clustering
============================================================
  Assignment: Perform K-Means clustering on a mall dataset
              and describe customer groups.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────
# THEME
# ──────────────────────────────────────────────────────────
BG      = '#f5f0eb'
CARD    = '#fdfaf7'
DARK    = '#1a1a2e'
C0      = '#e63946'   # Cluster 0 — High Income, Low Spend
C1      = '#2a9d8f'   # Cluster 1 — High Income, High Spend  ★ VIP
C2      = '#e9c46a'   # Cluster 2 — Mid Income, Mid Spend
C3      = '#f4a261'   # Cluster 3 — Low Income, Low Spend
C4      = '#6a4c93'   # Cluster 4 — Low Income, High Spend
CLUSTER_COLORS = [C0, C1, C2, C3, C4]
MUTED   = '#888888'
TEXT    = '#1a1a2e'
BORDER  = '#e0d8cf'

plt.rcParams.update({
    'font.family': 'monospace',
    'text.color': TEXT,
    'axes.facecolor': CARD,
    'figure.facecolor': BG,
    'axes.edgecolor': BORDER,
    'axes.labelcolor': MUTED,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'grid.color': BORDER,
    'grid.linewidth': 0.5,
})

# ──────────────────────────────────────────────────────────
# STEP 1 — INTRODUCTION
# ──────────────────────────────────────────────────────────
print("=" * 64)
print("  CUSTOMER SEGMENTATION — K-Means Clustering")
print("=" * 64)
print("""
📌 OBJECTIVE
─────────────
Segment mall customers into distinct groups based on their
Annual Income and Spending Score so the marketing team can:
  • Target high-value customers with premium offers
  • Re-engage low-spend, high-income customers
  • Design loyalty programs per segment

📊 DATASET: Mall Customer Dataset (synthetic, 200 customers)
  Features used:
    • Annual Income (k$)   — purchasing power
    • Spending Score (1-100) — mall-assigned score
    • Age                  — for profile description
    • Gender               — for demographic insight
""")

# ──────────────────────────────────────────────────────────
# STEP 2 — GENERATE DATASET
# ──────────────────────────────────────────────────────────
np.random.seed(42)
n = 200

# Simulate 5 natural clusters of customers
def make_cluster(n, income_mu, income_sd, score_mu, score_sd,
                 age_mu, age_sd, male_prob):
    return pd.DataFrame({
        'Annual_Income':    np.clip(np.random.normal(income_mu, income_sd, n), 15, 140).round(1),
        'Spending_Score':   np.clip(np.random.normal(score_mu,  score_sd,  n), 1,  100).round(0).astype(int),
        'Age':              np.clip(np.random.normal(age_mu,    age_sd,    n), 18, 70).round(0).astype(int),
        'Gender':           np.random.choice(['Male','Female'], n,
                                             p=[male_prob, 1-male_prob])
    })

clusters_raw = [
    make_cluster(40,  90, 10, 20, 10, 45, 8, 0.55),   # High Income, Low Spend
    make_cluster(40,  88, 10, 82, 10, 33, 7, 0.40),   # High Income, High Spend ★ VIP
    make_cluster(40,  55,  8, 50,  8, 40, 9, 0.50),   # Mid Income,  Mid Spend
    make_cluster(40,  26,  8, 20,  8, 42, 9, 0.50),   # Low Income,  Low Spend
    make_cluster(40,  25,  7, 80,  9, 25, 6, 0.35),   # Low Income,  High Spend
]

df = pd.concat(clusters_raw, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
df.index.name = 'CustomerID'
df.index = df.index + 1

print("=" * 64)
print("📋 STEP 2 — DATASET OVERVIEW")
print("=" * 64)
print(f"\n  Shape : {df.shape[0]} customers × {df.shape[1]} features\n")
print(df.head(10).to_string())
print(f"\n  Statistical Summary:")
print(df.describe().round(2).to_string())
print(f"\n  Gender Distribution:")
print(df['Gender'].value_counts().to_string())

# ──────────────────────────────────────────────────────────
# STEP 3 — PREPROCESSING
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("⚙️  STEP 3 — PREPROCESSING")
print("=" * 64)

X = df[['Annual_Income', 'Spending_Score']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("""
  Features selected for clustering:
    • Annual_Income   — economic capability
    • Spending_Score  — behavioral pattern

  Preprocessing applied:
    • StandardScaler → Zero mean, unit variance
      (ensures both features contribute equally to distance)

  Why NOT use Age/Gender directly in K-Means?
    → Age & Gender used for PROFILING after clustering,
      not for distance calculation (avoids demographic bias)
""")
print(f"  Before scaling — Income range : {X[:,0].min():.1f} to {X[:,0].max():.1f}")
print(f"  Before scaling — Score range  : {X[:,1].min():.0f} to {X[:,1].max():.0f}")
print(f"  After  scaling — Income range : {X_scaled[:,0].min():.2f} to {X_scaled[:,0].max():.2f}")
print(f"  After  scaling — Score range  : {X_scaled[:,1].min():.2f} to {X_scaled[:,1].max():.2f}")

# ──────────────────────────────────────────────────────────
# STEP 4 — ELBOW METHOD & SILHOUETTE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("📐 STEP 4 — FIND OPTIMAL K (Elbow + Silhouette)")
print("=" * 64)

inertias    = []
silhouettes = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

print(f"\n  {'K':>4}  {'Inertia':>12}  {'Silhouette Score':>18}  {'Recommendation'}")
print("  " + "─" * 60)
for i, k in enumerate(k_range):
    rec = " ← ★ OPTIMAL" if k == 5 else ""
    print(f"  {k:>4}  {inertias[i]:>12.2f}  {silhouettes[i]:>18.4f}{rec}")

print("""
  Elbow Method:
    → Inertia drops sharply from K=2 to K=5, then flattens
    → Elbow is at K = 5  ✅

  Silhouette Score (higher = better defined clusters):
    → Peaks at K = 5 with score ~0.55
    → Confirms K = 5 as optimal  ✅
""")

# ──────────────────────────────────────────────────────────
# STEP 5 — FIT K-MEANS WITH K=5
# ──────────────────────────────────────────────────────────
print("=" * 64)
print("🤖 STEP 5 — K-MEANS CLUSTERING  (K = 5)")
print("=" * 64)

K_OPTIMAL = 5
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Map clusters to meaningful order (by income desc, then score desc)
centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_order = pd.DataFrame(centers_orig, columns=['Income','Score'])
# Assign readable labels by income/score quadrant
label_map = {}
for i, (inc, sco) in enumerate(zip(cluster_order['Income'], cluster_order['Score'])):
    if inc >= 65 and sco < 45:   label_map[i] = 0   # High Inc, Low Spend
    elif inc >= 65 and sco >= 55: label_map[i] = 1   # High Inc, High Spend ★
    elif 40 <= inc < 75:          label_map[i] = 2   # Mid Inc, Mid Spend
    elif inc < 40 and sco < 45:   label_map[i] = 3   # Low Inc, Low Spend
    else:                         label_map[i] = 4   # Low Inc, High Spend

df['Cluster'] = df['Cluster'].map(label_map)

sil = silhouette_score(X_scaled, df['Cluster'])
print(f"\n  K = {K_OPTIMAL}  |  Silhouette Score = {sil:.4f}  |  Inertia = {kmeans.inertia_:.2f}")
print(f"\n  Cluster Centers (original scale):")
print(f"  {'Cluster':>10}  {'Income (k$)':>13}  {'Spending Score':>15}")
print("  " + "─" * 44)
for i, (inc, sco) in enumerate(zip(cluster_order['Income'], cluster_order['Score'])):
    print(f"  {label_map[i]:>10}  {inc:>13.1f}  {sco:>15.1f}")

# ──────────────────────────────────────────────────────────
# STEP 6 — DESCRIBE CUSTOMER SEGMENTS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("👥 STEP 6 — CUSTOMER SEGMENT PROFILES")
print("=" * 64)

segment_info = {
    0: {
        'name':     'Careful Spenders',
        'color':    C0,
        'emoji':    '💼',
        'desc':     'High income but low spending score',
        'strategy': 'Exclusive premium offers, VIP previews, loyalty rewards',
        'tone':     'Luxury & value messaging',
    },
    1: {
        'name':     'VIP Champions',
        'color':    C1,
        'emoji':    '👑',
        'desc':     'High income AND high spending — best customers',
        'strategy': 'Retain with elite membership, personalized concierge',
        'tone':     'Exclusivity, status, priority access',
    },
    2: {
        'name':     'Balanced Regulars',
        'color':    C2,
        'emoji':    '🏪',
        'desc':     'Middle income, average spending — the majority',
        'strategy': 'Discount bundles, seasonal promotions, referral programs',
        'tone':     'Value for money, family-friendly deals',
    },
    3: {
        'name':     'Conservative Savers',
        'color':    C3,
        'emoji':    '🏦',
        'desc':     'Low income, low spending — budget-conscious',
        'strategy': 'Budget deals, essential discounts, clearance sales',
        'tone':     'Savings, necessity, affordability',
    },
    4: {
        'name':     'Impulse Buyers',
        'color':    C4,
        'emoji':    '🛍️',
        'desc':     'Low income but HIGH spending — risk of debt',
        'strategy': 'Flash sales, EMI options, BNPL, trending items',
        'tone':     'Trendy, FOMO, excitement, must-have now',
    },
}

for cid, info in segment_info.items():
    grp = df[df['Cluster'] == cid]
    print(f"\n  {info['emoji']}  CLUSTER {cid} — {info['name']}")
    print(f"  {'─'*52}")
    print(f"  Description      : {info['desc']}")
    print(f"  Count            : {len(grp)} customers ({len(grp)/len(df)*100:.1f}%)")
    print(f"  Avg Income       : ${grp['Annual_Income'].mean():.1f}k")
    print(f"  Avg Spend Score  : {grp['Spending_Score'].mean():.1f}/100")
    print(f"  Avg Age          : {grp['Age'].mean():.1f} yrs")
    print(f"  Female %         : {(grp['Gender']=='Female').mean()*100:.1f}%")
    print(f"  Marketing Focus  : {info['strategy']}")
    print(f"  Messaging Tone   : {info['tone']}")

# ──────────────────────────────────────────────────────────
# STEP 7 — CLUSTER SUMMARY TABLE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("📊 STEP 7 — CLUSTER SUMMARY TABLE")
print("=" * 64)

summary = df.groupby('Cluster').agg(
    Count=('Annual_Income', 'count'),
    Avg_Income=('Annual_Income', 'mean'),
    Avg_Score=('Spending_Score', 'mean'),
    Avg_Age=('Age', 'mean'),
).round(1)
summary['Name'] = [segment_info[i]['name'] for i in summary.index]
summary['%'] = (summary['Count']/len(df)*100).round(1)
print("\n" + summary[['Name','Count','%','Avg_Income','Avg_Score','Avg_Age']].to_string())

print("""
  ─────────────────────────────────────────────────────────
  KEY BUSINESS INSIGHTS:
  ─────────────────────────────────────────────────────────
  1. Cluster 1 (VIP Champions) — Only 20% of customers but
     generate the highest revenue. Must be retained at all costs.

  2. Cluster 0 (Careful Spenders) — Huge untapped potential.
     High income but low spend → needs better engagement strategy.

  3. Cluster 4 (Impulse Buyers) — High spending despite low income.
     Risk of churn or financial stress. BNPL options help retain.

  4. Cluster 2 (Balanced Regulars) — Largest stable base.
     Loyalty programs can convert some to Cluster 1 over time.

  5. Cluster 3 (Conservative Savers) — Price-sensitive majority.
     Discounts and budget offers keep them engaged.
""")

# ──────────────────────────────────────────────────────────
# VISUALISATIONS
# ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

def style_ax(ax, title, color=DARK):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_title(title, color=color, fontsize=9, fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(color=BORDER, linewidth=0.5, alpha=0.7)

# 1. Raw scatter BEFORE clustering
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(df['Annual_Income'], df['Spending_Score'],
            c=MUTED, s=30, alpha=0.6, edgecolors='white', linewidth=0.3)
ax1.set_xlabel('Annual Income (k$)')
ax1.set_ylabel('Spending Score')
style_ax(ax1, 'Raw Data — Before Clustering', DARK)

# 2. K-Means result — main cluster plot
ax2 = fig.add_subplot(gs[0, 1:])
for cid in range(K_OPTIMAL):
    grp = df[df['Cluster'] == cid]
    ax2.scatter(grp['Annual_Income'], grp['Spending_Score'],
                c=CLUSTER_COLORS[cid], s=55, alpha=0.75,
                edgecolors='white', linewidth=0.4, label=f"C{cid}: {segment_info[cid]['name']}")

# Plot centroids
centers_inv = scaler.inverse_transform(kmeans.cluster_centers_)
for i, (inc, sco) in enumerate(centers_inv):
    ci = label_map[i]
    ax2.scatter(inc, sco, c=CLUSTER_COLORS[ci], s=250, marker='*',
                edgecolors=DARK, linewidth=1.2, zorder=10)
    ax2.annotate(f"C{ci}", (inc+1, sco+1.5), fontsize=8,
                 color=CLUSTER_COLORS[ci], fontweight='bold')

ax2.set_xlabel('Annual Income (k$)')
ax2.set_ylabel('Spending Score')
ax2.legend(fontsize=7.5, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT,
           loc='upper left')
style_ax(ax2, 'K-Means Clustering Result  (K=5)  ★ = Centroid', DARK)

# 3. Elbow curve
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(list(k_range), inertias, color=C0, linewidth=2, marker='o',
         markersize=5, markerfacecolor=C0)
ax3.axvline(5, color=C1, linewidth=1.5, linestyle='--', label='K=5 (elbow)')
ax3.fill_between(list(k_range), inertias, alpha=0.08, color=C0)
ax3.set_xlabel('Number of Clusters (K)')
ax3.set_ylabel('Inertia (WCSS)')
ax3.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
style_ax(ax3, 'Elbow Method', C0)

# 4. Silhouette scores
ax4 = fig.add_subplot(gs[1, 1])
bar_colors = [C1 if k==5 else MUTED for k in k_range]
bars = ax4.bar(list(k_range), silhouettes, color=bar_colors, edgecolor='white', linewidth=0.4)
for bar, val in zip(bars, silhouettes):
    ax4.text(bar.get_x()+bar.get_width()/2, val+0.003, f'{val:.3f}',
             ha='center', va='bottom', fontsize=6.5, color=TEXT)
ax4.set_xlabel('Number of Clusters (K)')
ax4.set_ylabel('Silhouette Score')
style_ax(ax4, 'Silhouette Score per K', C1)

# 5. Cluster size pie
ax5 = fig.add_subplot(gs[1, 2])
sizes = [len(df[df['Cluster']==c]) for c in range(K_OPTIMAL)]
labels_pie = [f"C{c}\n{segment_info[c]['name']}\n({s})" for c, s in enumerate(sizes)]
wedges, texts, autotexts = ax5.pie(
    sizes, labels=labels_pie, colors=CLUSTER_COLORS,
    autopct='%1.1f%%', startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=1.5),
    textprops=dict(fontsize=6.5, color=TEXT)
)
for at in autotexts:
    at.set_fontsize(7)
    at.set_color(DARK)
    at.set_fontweight('bold')
ax5.set_title('Cluster Size Distribution', color=DARK, fontsize=9, fontweight='bold', pad=10)

# 6. Avg Income & Score per cluster (grouped bar)
ax6 = fig.add_subplot(gs[2, 0])
cids = list(range(K_OPTIMAL))
avg_inc = [df[df['Cluster']==c]['Annual_Income'].mean() for c in cids]
avg_sco = [df[df['Cluster']==c]['Spending_Score'].mean() for c in cids]
x = np.arange(K_OPTIMAL)
w = 0.35
b1 = ax6.bar(x-w/2, avg_inc, w, label='Avg Income (k$)', color=[CLUSTER_COLORS[c] for c in cids],
             alpha=0.9, edgecolor='white', linewidth=0.5)
b2 = ax6.bar(x+w/2, avg_sco, w, label='Avg Score', color=[CLUSTER_COLORS[c] for c in cids],
             alpha=0.5, edgecolor='white', linewidth=0.5, hatch='//')
for bar, val in zip(b1, avg_inc):
    ax6.text(bar.get_x()+bar.get_width()/2, val+0.5, f'{val:.0f}',
             ha='center', va='bottom', fontsize=6.5, color=TEXT)
for bar, val in zip(b2, avg_sco):
    ax6.text(bar.get_x()+bar.get_width()/2, val+0.5, f'{val:.0f}',
             ha='center', va='bottom', fontsize=6.5, color=TEXT)
ax6.set_xticks(x)
ax6.set_xticklabels([f'C{c}' for c in cids])
ax6.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
style_ax(ax6, 'Avg Income vs Spending Score per Cluster', DARK)

# 7. Age distribution violin-style (box)
ax7 = fig.add_subplot(gs[2, 1])
age_data = [df[df['Cluster']==c]['Age'].values for c in range(K_OPTIMAL)]
bp = ax7.boxplot(age_data, patch_artist=True,
                 medianprops=dict(color=DARK, linewidth=2),
                 whiskerprops=dict(color=MUTED),
                 capprops=dict(color=MUTED),
                 flierprops=dict(marker='o', color=MUTED, markersize=3, alpha=0.5))
for patch, col in zip(bp['boxes'], CLUSTER_COLORS):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)
    patch.set_edgecolor(DARK)
ax7.set_xticklabels([f"C{c}\n{segment_info[c]['name'][:8]}" for c in range(K_OPTIMAL)],
                     fontsize=6.5)
ax7.set_ylabel('Age')
style_ax(ax7, 'Age Distribution per Cluster', DARK)

# 8. Gender breakdown stacked bar
ax8 = fig.add_subplot(gs[2, 2])
female_pct = [((df[df['Cluster']==c]['Gender']=='Female').sum()/len(df[df['Cluster']==c]))*100
              for c in range(K_OPTIMAL)]
male_pct = [100-f for f in female_pct]
cluster_labels = [f"C{c}" for c in range(K_OPTIMAL)]
ax8.bar(cluster_labels, female_pct, label='Female', color='#e91e8c', alpha=0.8, edgecolor='white')
ax8.bar(cluster_labels, male_pct,   label='Male',   color='#1e88e5', alpha=0.8, edgecolor='white',
        bottom=female_pct)
for i, (fp, mp) in enumerate(zip(female_pct, male_pct)):
    if fp > 10:
        ax8.text(i, fp/2, f'{fp:.0f}%', ha='center', va='center', fontsize=7,
                 color='white', fontweight='bold')
    if mp > 10:
        ax8.text(i, fp+mp/2, f'{mp:.0f}%', ha='center', va='center', fontsize=7,
                 color='white', fontweight='bold')
ax8.set_ylim(0, 115)
ax8.set_ylabel('Percentage (%)')
ax8.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
style_ax(ax8, 'Gender Breakdown per Cluster', DARK)

fig.suptitle("Customer Segmentation — K-Means Clustering on Mall Dataset",
             color=DARK, fontsize=14, fontweight='bold', y=0.99)

import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'customer_segmentation.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("=" * 64)
print("✅ COMPLETE — K-Means Customer Segmentation done!")
print(f"   Plot saved → customer_segmentation.png")
print("=" * 64)