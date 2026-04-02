"""
============================================================
  HOUSE PRICE PREDICTOR — Linear Regression (scikit-learn)
============================================================
  Assignment: Train a Linear Regression model, predict
              prices, and test with new input.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────
# STEP 1 — DATASET
# ──────────────────────────────────────────────
print("=" * 58)
print("  HOUSE PRICE PREDICTOR  —  Linear Regression")
print("=" * 58)

data = {
    'Area_sqft': [1200,1500,1800,2100,2500, 900,1100,3000,1700,2200,
                  1350,2800,1600, 950,2000,1400,3200,1750,2300,1050],
    'Bedrooms':  [   2,   3,   3,   4,   4,   2,   2,   5,   3,   4,
                     2,   5,   3,   1,   3,   2,   5,   3,   4,   2],
    'Age_years': [  15,  10,   8,   5,   3,  25,  20,   2,  12,   7,
                    18,   4,   9,  30,   6,  14,   1,  11,   6,  22],
    'Price_K':   [ 180, 230, 275, 340, 410, 130, 160, 490, 255, 360,
                   195, 455, 245, 120, 315, 210, 530, 268, 375, 148],
}

df = pd.DataFrame(data)

print("\n📋 STEP 1 — Dataset (20 samples)")
print("-" * 58)
print(df.to_string(index=True))
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nStatistical Summary:")
print(df.describe().round(2))


# ──────────────────────────────────────────────
# STEP 2 — TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
X = df[['Area_sqft', 'Bedrooms', 'Age_years']]
y = df['Price_K']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n" + "=" * 58)
print("📊 STEP 2 — Train / Test Split (80% / 20%)")
print("-" * 58)
print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")


# ──────────────────────────────────────────────
# STEP 3 — TRAIN THE MODEL
# ──────────────────────────────────────────────
print("\n" + "=" * 58)
print("🧠 STEP 3 — Training Linear Regression Model")
print("-" * 58)

model = LinearRegression()
model.fit(X_train, y_train)

print("  Algorithm  : Ordinary Least Squares (OLS)")
print(f"  Intercept  (θ₀) : {model.intercept_:.4f}")
for feat, coef in zip(X.columns, model.coef_):
    print(f"  Coeff {feat:12s}: {coef:.4f}")

print("\n  Learned Equation:")
print(f"  Price = {model.intercept_:.2f}")
for feat, coef in zip(X.columns, model.coef_):
    sign = '+' if coef >= 0 else '-'
    print(f"          {sign} {abs(coef):.4f} × {feat}")


# ──────────────────────────────────────────────
# STEP 4 — EVALUATE
# ──────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)
mse_test  = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test  = mean_absolute_error(y_test, y_pred_test)

print("\n" + "=" * 58)
print("📈 STEP 4 — Model Evaluation")
print("-" * 58)
print(f"  R² Score  (Train) : {r2_train:.4f}")
print(f"  R² Score  (Test)  : {r2_test:.4f}")
print(f"  MSE       (Test)  : {mse_test:.2f} K²")
print(f"  RMSE      (Test)  : {rmse_test:.2f} K")
print(f"  MAE       (Test)  : {mae_test:.2f} K")

print("\n  Actual vs Predicted (Test Set):")
print(f"  {'Actual ($K)':>12}  {'Predicted ($K)':>14}  {'Error ($K)':>10}")
print("  " + "-" * 42)
for actual, predicted in zip(y_test, y_pred_test):
    err = predicted - actual
    print(f"  {actual:>12.1f}  {predicted:>14.1f}  {err:>+10.1f}")


# ──────────────────────────────────────────────
# STEP 5 — PREDICT NEW INPUT
# ──────────────────────────────────────────────
print("\n" + "=" * 58)
print("🔮 STEP 5 — Predict New House Prices")
print("-" * 58)

new_houses = pd.DataFrame({
    'Area_sqft': [2000, 3500, 1100],
    'Bedrooms':  [   3,    5,    2],
    'Age_years': [  10,    1,   20],
})

new_predictions = model.predict(new_houses)

print(f"\n  {'Area':>8}  {'Beds':>5}  {'Age':>5}  {'Predicted Price':>16}")
print("  " + "-" * 42)
for i, (_, row) in enumerate(new_houses.iterrows()):
    print(f"  {int(row.Area_sqft):>8} sqft  "
          f"{int(row.Bedrooms):>2} bed  "
          f"{int(row.Age_years):>3} yr  "
          f"  ${new_predictions[i]:>10.1f}K")


# ──────────────────────────────────────────────
# STEP 6 — VISUALISATIONS
# ──────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('#0b0f0e')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

GREEN  = '#00d68f'
ORANGE = '#ff6b35'
MUTED  = '#6b8580'
BG     = '#131918'
CARD   = '#1a2120'
TEXT   = '#e8f0ee'

ax_style = dict(facecolor=CARD, labelcolor=TEXT)

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_title(title, color=GREEN, fontsize=9, fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#243330')
    ax.grid(color='#243330', linewidth=0.5, alpha=0.6)

# 1. Actual vs Predicted (Test)
ax1 = fig.add_subplot(gs[0, 0])
all_vals = list(y_test) + list(y_pred_test)
mn, mx = min(all_vals)-20, max(all_vals)+20
ax1.plot([mn, mx], [mn, mx], '--', color=ORANGE, linewidth=1.2, label='Perfect fit', alpha=0.7)
ax1.scatter(y_test, y_pred_test, color=GREEN, s=70, zorder=5, edgecolors='#0b0f0e', linewidth=0.8)
ax1.set_xlabel('Actual Price ($K)')
ax1.set_ylabel('Predicted Price ($K)')
style_ax(ax1, 'Actual vs Predicted')
ax1.legend(fontsize=7, facecolor=CARD, edgecolor='#243330', labelcolor=MUTED)

# 2. Residuals
ax2 = fig.add_subplot(gs[0, 1])
residuals = y_test.values - y_pred_test
ax2.axhline(0, color=ORANGE, linewidth=1, linestyle='--', alpha=0.7)
ax2.scatter(y_pred_test, residuals, color=GREEN, s=70, edgecolors='#0b0f0e', linewidth=0.8, zorder=5)
ax2.set_xlabel('Predicted Price ($K)')
ax2.set_ylabel('Residual ($K)')
style_ax(ax2, 'Residual Plot')

# 3. Feature coefficients
ax3 = fig.add_subplot(gs[0, 2])
coefs = model.coef_
features = ['Area (sqft)', 'Bedrooms', 'Age (yrs)']
colors_bar = [GREEN if c > 0 else ORANGE for c in coefs]
bars = ax3.barh(features, coefs, color=colors_bar, edgecolor='#0b0f0e', linewidth=0.8)
for bar, val in zip(bars, coefs):
    ax3.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', ha='left' if val>=0 else 'right',
             color=TEXT, fontsize=7)
ax3.set_xlabel('Coefficient Value')
style_ax(ax3, 'Feature Coefficients')

# 4. Price vs Area scatter
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(df['Area_sqft'], df['Price_K'], color=GREEN, s=55,
            edgecolors='#0b0f0e', linewidth=0.8, zorder=5)
sort_idx = df['Area_sqft'].argsort()
x_line = df['Area_sqft'].iloc[sort_idx]
y_line = model.predict(df[['Area_sqft','Bedrooms','Age_years']].iloc[sort_idx])
ax4.plot(x_line, y_line, color=ORANGE, linewidth=1.5, alpha=0.8, label='Model fit')
ax4.set_xlabel('Area (sq ft)')
ax4.set_ylabel('Price ($K)')
style_ax(ax4, 'Price vs Area')

# 5. Price vs Bedrooms (box)
ax5 = fig.add_subplot(gs[1, 1])
bed_groups = [df[df['Bedrooms']==b]['Price_K'].values for b in sorted(df['Bedrooms'].unique())]
bp = ax5.boxplot(bed_groups, tick_labels=sorted(df['Bedrooms'].unique()),
                  patch_artist=True, medianprops=dict(color=ORANGE, linewidth=2))
for patch in bp['boxes']:
    patch.set_facecolor(CARD)
    patch.set_edgecolor(GREEN)
for element in ['whiskers','caps','fliers']:
    for item in bp[element]: item.set_color(MUTED)
ax5.set_xlabel('Bedrooms')
ax5.set_ylabel('Price ($K)')
style_ax(ax5, 'Price by Bedrooms')

# 6. Metrics summary bar
ax6 = fig.add_subplot(gs[1, 2])
metric_names = ['R² Train', 'R² Test']
metric_vals  = [r2_train, r2_test]
bars6 = ax6.bar(metric_names, metric_vals, color=[GREEN, '#00a86b'],
                edgecolor='#0b0f0e', linewidth=0.8, width=0.5)
for bar, val in zip(bars6, metric_vals):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.4f}',
             ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
ax6.set_ylim(0, 1.15)
ax6.set_ylabel('Score')
style_ax(ax6, 'R² Score Comparison')
ax6.text(0.5, 0.5, f'MAE: {mae_test:.1f}K\nRMSE: {rmse_test:.1f}K',
         transform=ax6.transAxes, ha='center', va='center',
         color=MUTED, fontsize=8, style='italic')

fig.suptitle('House Price Predictor — Linear Regression Analysis',
             color=TEXT, fontsize=13, fontweight='bold', y=0.98)

import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'house_price_plots.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"   Plots saved to: {save_path}")
plt.close()

print("\n" + "=" * 58)
print("✅ COMPLETE — Model trained, evaluated & predictions made!")
print("   Plots saved to: house_price_plots.png")
print("=" * 58)