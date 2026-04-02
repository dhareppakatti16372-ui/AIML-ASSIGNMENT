# Decision Tree: Should You Play Outside?
# Uses sklearn to build and visualize the decision tree

import os
import numpy as np

# Save output image next to this script (works on Windows, Mac, Linux)
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "decision_tree_play_outside.png")
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# 1. Dataset
# Features:
#   [is_raining, is_heavy_rain, extreme_temp, have_rain_gear, enough_time]
#   All values: 0 = No, 1 = Yes
# Label: 1 = Play outside, 0 = Stay inside
# ─────────────────────────────────────────────

X = np.array([
    # rain  heavy  extreme  gear  time
    [0,     0,     0,       0,    1],   # nice day, enough time → play
    [0,     0,     0,       0,    0],   # nice day, no time     → stay
    [0,     0,     1,       0,    1],   # extreme temp          → stay
    [0,     0,     1,       0,    0],   # extreme temp, no time → stay
    [1,     0,     0,       1,    1],   # light rain, gear, time → play
    [1,     0,     0,       1,    0],   # light rain, gear, no time → stay
    [1,     0,     0,       0,    1],   # light rain, no gear   → stay
    [1,     0,     0,       0,    0],   # light rain, no gear, no time → stay
    [1,     1,     0,       1,    1],   # heavy rain            → stay
    [1,     1,     0,       0,    1],   # heavy rain, no gear   → stay
    [1,     1,     1,       0,    0],   # heavy rain, extreme   → stay
    [0,     0,     0,       1,    1],   # nice, has gear, time  → play
])

y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

feature_names = ["Raining", "Heavy Rain", "Extreme Temp", "Has Rain Gear", "Enough Time"]
class_names   = ["Stay Inside", "Play Outside"]

# ─────────────────────────────────────────────
# 2. Train the decision tree
# ─────────────────────────────────────────────
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

# ─────────────────────────────────────────────
# 3. Print text representation
# ─────────────────────────────────────────────
print("=" * 55)
print("   DECISION TREE: Should You Play Outside?")
print("=" * 55)
print(export_text(clf, feature_names=feature_names))

# ─────────────────────────────────────────────
# 4. Draw the tree visually
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 8))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=11,
    ax=ax,
    impurity=False,
    proportion=False,
    precision=0,
)

# Colour the leaf nodes green (play) / red (stay)
for artist in ax.get_children():
    if hasattr(artist, "get_facecolor"):
        fc = artist.get_facecolor()
        if fc is not None and len(fc) == 4:
            r, g, b, a = fc
            # sklearn colours Play Outside nodes with high green component
            if g > 0.55 and r < 0.6:
                artist.set_facecolor("#A8D5B5")   # soft green
                artist.set_edgecolor("#3B6D11")
            elif r > 0.55 and g < 0.5:
                artist.set_facecolor("#F5B7B1")   # soft red
                artist.set_edgecolor("#A32D2D")

# Legend
green_patch = mpatches.Patch(color="#A8D5B5", label="Play Outside ✓")
red_patch   = mpatches.Patch(color="#F5B7B1", label="Stay Inside ✗")
ax.legend(handles=[green_patch, red_patch], loc="upper right",
          fontsize=12, framealpha=0.9)

ax.set_title("Decision Tree — Should You Play Outside?",
             fontsize=16, fontweight="bold", pad=16, color="#2C2C2A")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print(f"\nTree diagram saved to: {OUTPUT_PATH}")

# ─────────────────────────────────────────────
# 5. Interactive prediction function
# ─────────────────────────────────────────────
def predict_play_outside():
    print("\n" + "=" * 55)
    print("   Interactive Prediction")
    print("=" * 55)

    def ask(question):
        while True:
            ans = input(f"  {question} (yes/no): ").strip().lower()
            if ans in ("yes", "y"):
                return 1
            if ans in ("no", "n"):
                return 0
            print("  Please enter 'yes' or 'no'.")

    raining    = ask("Is it raining?")
    heavy_rain = ask("Is it heavy rain?") if raining else 0
    extreme    = ask("Is the temperature extreme (very hot or very cold)?")
    gear       = ask("Do you have rain gear (umbrella/raincoat)?") if raining else 0
    time_ok    = ask("Do you have at least 30 minutes of free time?")

    features = [[raining, heavy_rain, extreme, gear, time_ok]]
    result   = clf.predict(features)[0]
    proba    = clf.predict_proba(features)[0]

    print("\n" + "-" * 55)
    if result == 1:
        print("  ✅  GO PLAY OUTSIDE!")
    else:
        print("  🏠  STAY INSIDE.")
    print(f"  Confidence: {max(proba)*100:.0f}%")
    print("-" * 55)

# Uncomment the line below to run the interactive predictor:
# predict_play_outside()