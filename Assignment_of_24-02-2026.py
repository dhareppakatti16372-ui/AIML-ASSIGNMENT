# ============================================
#           DATASET DETECTIVE
#   Load → Explore → Analyse → Insights
# ============================================

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────

print("\n" + "=" * 62)
print("           DATASET DETECTIVE")
print("=" * 62)

# Using the built-in Titanic dataset (no download needed)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

print("\n  📂 Loading Titanic dataset ...", end=" ", flush=True)
try:
    df = pd.read_csv(url)
    print("Done!")
except Exception:
    # Fallback: create a representative mini-dataset
    print("Offline — using built-in sample data.")
    data = {
        "PassengerId": range(1, 11),
        "Survived":    [0,1,1,1,0,0,0,0,1,1],
        "Pclass":      [3,1,3,1,3,3,1,3,3,2],
        "Name":        ["Braund, Mr. Owen","Cumings, Mrs. John","Heikkinen, Miss. Laina",
                        "Futrelle, Mrs. Jacques","Allen, Mr. William","Moran, Mr. James",
                        "McCarthy, Mr. Timothy","Palsson, Master. Gosta","Johnson, Mrs. Oscar",
                        "Nasser, Mrs. Nicholas"],
        "Sex":         ["male","female","female","female","male","male","male","male","female","female"],
        "Age":         [22,38,26,35,35,None,54,2,27,14],
        "SibSp":       [1,1,0,1,0,0,0,3,0,1],
        "Parch":       [0,0,0,0,0,0,0,1,2,0],
        "Ticket":      ["A/5 21171","PC 17599","STON/O2","113803","373450",
                        "330877","17463","349909","347742","237736"],
        "Fare":        [7.25,71.28,7.92,53.10,8.05,8.46,51.86,21.07,11.13,30.07],
        "Cabin":       [None,"C85",None,"C123",None,None,"E46",None,None,None],
        "Embarked":    ["S","C","S","S","S","Q","S","S","S","C"],
    }
    df = pd.DataFrame(data)

# ─────────────────────────────────────────────
# STEP 2 — TOP ROWS
# ─────────────────────────────────────────────

print("\n" + "=" * 62)
print("  📋  TOP 5 ROWS")
print("=" * 62)
print(df.head().to_string(index=False))

# ─────────────────────────────────────────────
# STEP 3 — SHAPE & DATA TYPES
# ─────────────────────────────────────────────

print("\n" + "=" * 62)
print("  📐  DATASET SHAPE & COLUMN TYPES")
print("=" * 62)
print(f"  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
print()
print(f"  {'Column':<15}  {'Type':<10}  Non-Null Count")
print("  " + "-" * 42)
for col in df.columns:
    non_null = df[col].notna().sum()
    print(f"  {col:<15}  {str(df[col].dtype):<10}  {non_null:,} / {len(df):,}")

# ─────────────────────────────────────────────
# STEP 4 — MISSING VALUES
# ─────────────────────────────────────────────

print("\n" + "=" * 62)
print("  ❓  MISSING VALUES")
print("=" * 62)

missing      = df.isnull().sum()
missing_pct  = (missing / len(df) * 100).round(2)
missing_df   = pd.DataFrame({"Missing": missing, "Percentage %": missing_pct})
missing_df   = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)

if missing_df.empty:
    print("  ✅ No missing values found!")
else:
    print(f"  {'Column':<15}  {'Missing':>7}  {'%':>8}")
    print("  " + "-" * 34)
    for col, row in missing_df.iterrows():
        bar_len = int(row["Percentage %"] / 5)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {col:<15}  {int(row['Missing']):>7}  {row['Percentage %']:>7.2f}%  {bar}")

total_missing = missing.sum()
print(f"\n  Total missing cells : {total_missing:,} / {df.size:,} ({total_missing/df.size*100:.2f}%)")

# ─────────────────────────────────────────────
# STEP 5 — HIGHEST VALUE COLUMN
# ─────────────────────────────────────────────

print("\n" + "=" * 62)
print("  🏆  HIGHEST VALUE COLUMN (Numeric Columns — Mean)")
print("=" * 62)

numeric_cols = df.select_dtypes(include=[np.number])
col_means    = numeric_cols.mean().sort_values(ascending=False)

print(f"\n  {'Column':<15}  {'Mean':>10}  {'Max':>10}  {'Min':>10}")
print("  " + "-" * 50)
for col in col_means.index:
    print(f"  {col:<15}  {df[col].mean():>10.2f}  {df[col].max():>10.2f}  {df[col].min():>10.2f}")

top_col = col_means.idxmax()
print(f"\n  ✅ Column with highest mean value: [{top_col}] ({col_means[top_col]:.2f})")

# ─────────────────────────────────────────────
# STEP 6 — SUMMARY STATISTICS
# ─────────────────────────────────────────────

print("\n" + "=" * 62)
print("  📊  SUMMARY STATISTICS")
print("=" * 62)
print(df.describe().round(2).to_string())

# ─────────────────────────────────────────────
# STEP 7 — 5 DATA INSIGHTS
# ─────────────────────────────────────────────

print("\n" + "=" * 62)
print("  🔍  5 KEY INSIGHTS FROM THE TITANIC DATASET")
print("=" * 62)

# Insight 1 — Survival Rate
survival_rate = df["Survived"].mean() * 100
print(f"""
  1. 💀  SURVIVAL RATE
     Only {survival_rate:.1f}% of passengers survived the Titanic disaster.
     That means roughly 2 in 3 passengers did NOT make it.
""")

# Insight 2 — Gender & Survival
if "Sex" in df.columns:
    gender_survival = df.groupby("Sex")["Survived"].mean() * 100
    f_rate = gender_survival.get("female", 0)
    m_rate = gender_survival.get("male",   0)
    print(f"""  2. 👩‍✈️  GENDER DISPARITY IN SURVIVAL
     Female survival rate : {f_rate:.1f}%
     Male survival rate   : {m_rate:.1f}%
     Women were ~{f_rate/m_rate:.1f}× more likely to survive — reflecting the
     "women and children first" evacuation protocol.
""")

# Insight 3 — Class & Survival
if "Pclass" in df.columns:
    class_survival = df.groupby("Pclass")["Survived"].mean() * 100
    print(f"""  3. 🎟️   PASSENGER CLASS & SURVIVAL
     1st Class : {class_survival.get(1, 0):.1f}% survived
     2nd Class : {class_survival.get(2, 0):.1f}% survived
     3rd Class : {class_survival.get(3, 0):.1f}% survived
     Higher class = much better survival odds, exposing stark
     socioeconomic inequality even in life-or-death situations.
""")

# Insight 4 — Age Distribution
if "Age" in df.columns:
    avg_age    = df["Age"].mean()
    child_surv = df[df["Age"] < 16]["Survived"].mean() * 100
    print(f"""  4. 🧒  AGE & CHILDREN
     Average passenger age : {avg_age:.1f} years
     Survival rate (< 16)  : {child_surv:.1f}%
     Children had a notably higher survival rate, consistent
     with evacuation priority given to young passengers.
""")

# Insight 5 — Fare & Class Correlation
if "Fare" in df.columns and "Pclass" in df.columns:
    avg_fare_class = df.groupby("Pclass")["Fare"].mean()
    print(f"""  5. 💰  FARE vs CLASS
     Avg fare — 1st Class : £{avg_fare_class.get(1,0):>7.2f}
     Avg fare — 2nd Class : £{avg_fare_class.get(2,0):>7.2f}
     Avg fare — 3rd Class : £{avg_fare_class.get(3,0):>7.2f}
     1st class passengers paid ~{avg_fare_class.get(1,0)/avg_fare_class.get(3,1):.0f}× more than 3rd class.
     Fare is a strong predictor of survival in ML models.
""")

print("=" * 62)
print("  ✅  Analysis Complete!")
print("=" * 62 + "\n")