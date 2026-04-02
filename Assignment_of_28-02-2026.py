# ============================================
#        STORYTELLING WITH GRAPHS
#   Bar Chart | Pie Chart | Histogram
#   + Auto-generates data_story.txt
# ============================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Save folder — same folder as this script ──
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Shared style ──────────────────────────────
BG      = "#0f1117"
CARD    = "#1a1d27"
ACCENT  = ["#6c63ff", "#ff6584", "#43e97b", "#f7971e", "#38f9d7"]
TEXT    = "#e8eaf6"
SUBTEXT = "#9fa8da"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    "#2e3148",
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        "#2e3148",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
})

# ─────────────────────────────────────────────
# DATASET — Global Smartphone Sales by Brand
# ─────────────────────────────────────────────

brands       = ["Samsung", "Apple", "Xiaomi", "OPPO", "Vivo", "Others"]
market_share = [21.6, 17.3, 12.5, 10.2, 8.9, 29.5]
units_2023   = [226,  181,  130,  106,  93,  308]
units_2022   = [211,  166,  125,  100,  80,  290]

np.random.seed(42)
ages = np.concatenate([
    np.random.normal(28, 5,  400),
    np.random.normal(42, 7,  250),
    np.random.normal(55, 6,  100),
])
ages = ages[(ages >= 10) & (ages <= 75)]

# ─────────────────────────────────────────────
# CHART 1 — Grouped Bar Chart
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
fig.patch.set_facecolor(BG)

x     = np.arange(len(brands))
w     = 0.38
bars1 = ax.bar(x - w/2, units_2022, w, label="2022", color=ACCENT[0],
               alpha=0.85, linewidth=0, zorder=3)
bars2 = ax.bar(x + w/2, units_2023, w, label="2023", color=ACCENT[1],
               alpha=0.85, linewidth=0, zorder=3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{int(bar.get_height())}M", ha="center", va="bottom",
            fontsize=8, color=SUBTEXT)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{int(bar.get_height())}M", ha="center", va="bottom",
            fontsize=8, color=TEXT)

ax.set_xticks(x)
ax.set_xticklabels(brands, fontsize=10)
ax.set_ylabel("Units Sold (Millions)", fontsize=10, labelpad=10)
ax.set_title("Global Smartphone Sales by Brand\n2022 vs 2023",
             fontsize=14, fontweight="bold", color=TEXT, pad=18)
ax.legend(framealpha=0, labelcolor=TEXT, fontsize=10)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[:].set_visible(False)

plt.tight_layout(pad=2)
path1 = os.path.join(SAVE_DIR, "chart1_bar.png")
plt.savefig(path1, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Chart 1 saved -> {path1}")

# ─────────────────────────────────────────────
# CHART 2 — Donut Chart
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
fig.patch.set_facecolor(BG)

colors  = ACCENT + ["#a29bfe"]
explode = [0.05 if b in ["Samsung", "Apple"] else 0 for b in brands]

wedges, texts, autotexts = ax.pie(
    market_share,
    labels      = brands,
    autopct     = "%1.1f%%",
    startangle  = 140,
    colors      = colors,
    explode     = explode,
    pctdistance = 0.80,
    wedgeprops  = {"linewidth": 2, "edgecolor": BG, "width": 0.55},
)

for t in texts:
    t.set_color(TEXT);  t.set_fontsize(10)
for at in autotexts:
    at.set_color(BG);   at.set_fontsize(8.5); at.set_fontweight("bold")

ax.text(0, 0, "2023\nMarket\nShare", ha="center", va="center",
        fontsize=11, color=TEXT, fontweight="bold", linespacing=1.6)
ax.set_title("Global Smartphone Market Share - 2023",
             fontsize=14, fontweight="bold", color=TEXT, pad=20)

plt.tight_layout(pad=1.5)
path2 = os.path.join(SAVE_DIR, "chart2_pie.png")
plt.savefig(path2, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Chart 2 saved -> {path2}")

# ─────────────────────────────────────────────
# CHART 3 — Histogram
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
fig.patch.set_facecolor(BG)

n, bins, patches = ax.hist(ages, bins=26, edgecolor=BG, linewidth=0.6, zorder=3)

for patch, left in zip(patches, bins[:-1]):
    if   left < 25: patch.set_facecolor(ACCENT[4])
    elif left < 40: patch.set_facecolor(ACCENT[0])
    elif left < 55: patch.set_facecolor(ACCENT[1])
    else:           patch.set_facecolor(ACCENT[3])

ax.axvline(np.mean(ages),   color="#ffd700", linewidth=1.8, linestyle="--", zorder=4)
ax.axvline(np.median(ages), color="#ff6b6b", linewidth=1.8, linestyle=":",  zorder=4)

legend_patches = [
    mpatches.Patch(color=ACCENT[4], label="Teen / Young (<25)"),
    mpatches.Patch(color=ACCENT[0], label="Young Adult (25-39)"),
    mpatches.Patch(color=ACCENT[1], label="Middle-Aged (40-54)"),
    mpatches.Patch(color=ACCENT[3], label="Older Buyer (55+)"),
    plt.Line2D([0],[0], color="#ffd700", lw=1.8, ls="--",
               label=f"Mean age: {np.mean(ages):.1f} yrs"),
    plt.Line2D([0],[0], color="#ff6b6b", lw=1.8, ls=":",
               label=f"Median age: {np.median(ages):.1f} yrs"),
]
ax.legend(handles=legend_patches, framealpha=0, labelcolor=TEXT,
          fontsize=9, loc="upper right")

ax.set_xlabel("Age of Buyer (years)", fontsize=10, labelpad=10)
ax.set_ylabel("Number of Buyers",     fontsize=10, labelpad=10)
ax.set_title("Age Distribution of Smartphone Buyers",
             fontsize=14, fontweight="bold", color=TEXT, pad=18)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines[:].set_visible(False)

plt.tight_layout(pad=2)
path3 = os.path.join(SAVE_DIR, "chart3_histogram.png")
plt.savefig(path3, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Chart 3 saved -> {path3}")

# ─────────────────────────────────────────────
# WRITE DATA STORY — data_story.txt
# ─────────────────────────────────────────────

mean_age       = np.mean(ages)
median_age     = np.median(ages)
samsung_growth = (units_2023[0] - units_2022[0]) / units_2022[0] * 100
apple_growth   = (units_2023[1] - units_2022[1]) / units_2022[1] * 100
vivo_growth    = (units_2023[4] - units_2022[4]) / units_2022[4] * 100

story = f"""============================================================
        STORYTELLING WITH GRAPHS - DATA STORY
        Dataset: Global Smartphone Market (2022-2023)
============================================================

OVERVIEW
--------
This data story explores three key dimensions of the global
smartphone industry using three different chart types:

  Chart 1 - Bar Chart   : Brand-wise sales volume (2022 vs 2023)
  Chart 2 - Pie Chart   : Market share distribution (2023)
  Chart 3 - Histogram   : Age distribution of smartphone buyers


============================================================
  CHART 1 - BAR CHART STORY
  "The Battle for Billions"
============================================================

The grouped bar chart compares smartphone unit sales (in millions)
across six major brands between 2022 and 2023.

KEY OBSERVATIONS:

  * Samsung leads with {units_2023[0]}M units in 2023, up from {units_2022[0]}M in 2022
    — a healthy +{samsung_growth:.1f}% YoY increase. This signals sustained
    consumer trust and a strong mid-range product lineup.

  * Apple surged from {units_2022[1]}M to {units_2023[1]}M units (+{apple_growth:.1f}%), making it
    the fastest-growing top brand. The iPhone 15 series and
    expanding emerging market presence drove this jump.

  * Xiaomi held steady (125M to 130M), continuing its dominance
    in budget-conscious markets across South and Southeast Asia.

  * OPPO and Vivo both posted solid gains. Vivo grew +{vivo_growth:.0f}%,
    showing Chinese brands quietly eating into the fragmented
    "Others" category.

  * The "Others" bucket ({units_2023[5]}M in 2023) remains the largest
    slice overall but grew slower than named brands, suggesting
    market consolidation is underway.

TREND SUMMARY:
The smartphone market is recovering post-pandemic. Premium
brands (Apple, Samsung) are growing faster than average,
suggesting consumers are willing to spend more on flagship
devices. The "good enough" budget phone era may be giving
way to an aspirational upgrade cycle.


============================================================
  CHART 2 - PIE / DONUT CHART STORY
  "A Slice of 1.4 Billion Units"
============================================================

The donut chart visualises 2023 market share percentages across
six brands, offering a bird's-eye view of who dominates the
global smartphone landscape.

KEY OBSERVATIONS:

  * Samsung ({market_share[0]}%) and Apple ({market_share[1]}%) together command
    {market_share[0] + market_share[1]:.0f}% of the global market — an extraordinary duopoly.
    Their combined revenue likely accounts for 60%+ of total
    industry profit due to higher Average Selling Prices.

  * Xiaomi ({market_share[2]}%) is a clear #3, punching above its weight
    in the sub-$300 segment across India, Europe, and LatAm.

  * OPPO ({market_share[3]}%) and Vivo ({market_share[4]}%) — both under BBK Electronics
    — together hold {market_share[3] + market_share[4]:.0f}% market share. Their shared parent
    gives them supply chain advantages others cannot match.

  * "Others" ({market_share[5]}%) hides hundreds of regional OEMs: Tecno
    and Itel in Africa, Lava in India, Honor in China. As
    these brands mature, they could disrupt rankings by 2030.

TREND SUMMARY:
The market is simultaneously concentrated (top 2 own {market_share[0] + market_share[1]:.0f}%)
and fragmented (Others own {market_share[5]}%). This duality creates
intense competition in both premium and budget segments.


============================================================
  CHART 3 - HISTOGRAM STORY
  "Who's Actually Buying These Phones?"
============================================================

The histogram shows the age distribution of smartphone buyers,
with colour-coded age bands and reference lines for mean and
median age.

KEY OBSERVATIONS:

  * The distribution peaks between ages 22-32. This "young
    adult" cohort is the core buyer demographic — earning
    first salaries, upgrading devices, and brand-loyal.

  * Mean age   : {mean_age:.1f} years
    Median age : {median_age:.1f} years
    The mean is slightly higher than the median, confirming a
    right-skew. Older buyers (50-70) in developing regions
    pull the average upward.

  * Teen buyers (<25) form a sizable secondary peak. Heavily
    influenced by social media and peer behaviour — a critical
    target for brands like Apple and Samsung.

  * The 40-55 segment is thinner but has the HIGHEST average
    spend per device, often choosing premium flagships.

  * Very few buyers are above 65, though this is a growing
    segment as digital literacy improves, especially in India
    and Southeast Asia.

TREND SUMMARY:
Brands must think in two tracks: (1) hook younger buyers
early for lifetime loyalty, and (2) offer intuitive, large-
display, accessible devices for the rising 50+ segment.
A one-size-fits-all strategy is no longer viable.


============================================================
  OVERALL CONCLUSION
============================================================

  1. GROWTH IS BACK    - Apple (+{apple_growth:.1f}%) and Samsung (+{samsung_growth:.1f}%) led
                         strong YoY recovery post-pandemic.

  2. CONSOLIDATION     - "Others" shrinking as named brands
                         absorb market share globally.

  3. DIVERSE BUYERS    - From teens to seniors, segmented
                         strategies are now essential.

  4. PROFIT > VOLUME   - Apple & Samsung earn outsized profit
                         despite not having the most units.

  5. NEXT BATTLEGROUND - 50+ demographic and first-time users
                         in Africa & rural Asia by 2030.

------------------------------------------------------------
  Tool  : Python | matplotlib | numpy
  Data  : Simulated based on public industry trends
  Files :
    chart1_bar.png
    chart2_pie.png
    chart3_histogram.png
    data_story.txt        <- this file
============================================================
"""

story_path = os.path.join(SAVE_DIR, "data_story.txt")
with open(story_path, "w", encoding="utf-8") as f:
    f.write(story)

print(f"  Data story saved -> {story_path}")
print("\n  All files generated successfully!")
print(f"  Folder: {SAVE_DIR}")