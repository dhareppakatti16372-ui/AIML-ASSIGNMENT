"""
============================================================
  SPAM CLASSIFIER — System Design & Thinking
============================================================
  Assignment: Design a spam detection system:
              features, data needed, possible mistakes.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import textwrap

# ──────────────────────────────────────────────────────────
# THEME
# ──────────────────────────────────────────────────────────
BG      = '#0d0d0d'
CARD    = '#141414'
RED     = '#ff3b3b'
ORANGE  = '#ff8c42'
YELLOW  = '#ffd166'
GREEN   = '#06d6a0'
BLUE    = '#118ab2'
PURPLE  = '#9b5de5'
MUTED   = '#555555'
TEXT    = '#e0e0e0'
BORDER  = '#2a2a2a'

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
# STEP 1 — PROBLEM STATEMENT
# ──────────────────────────────────────────────────────────
print("=" * 62)
print("  SPAM CLASSIFIER — System Design Thinking")
print("=" * 62)

print("""
📌 PROBLEM STATEMENT
────────────────────
A Spam Classifier automatically distinguishes unwanted/malicious
messages (spam) from legitimate ones (ham).

  Input  → Raw text message (email / SMS / comment)
  Output → Binary label: SPAM (1) or HAM (0)

Use Cases:
  • Email spam filtering  (Gmail, Outlook)
  • SMS spam detection    (telecom providers)
  • Comment moderation    (YouTube, Reddit)
  • Phishing detection    (bank alerts, OTPs)
""")

# ──────────────────────────────────────────────────────────
# STEP 2 — FEATURES
# ──────────────────────────────────────────────────────────
print("=" * 62)
print("📐 STEP 2 — FEATURE ENGINEERING")
print("=" * 62)

features = {
    "TEXT / NLP FEATURES": [
        ("Bag of Words (BoW)",         "Count of each word token in the message"),
        ("TF-IDF Scores",              "Term freq × Inverse doc freq for word importance"),
        ("Word n-grams (1,2,3)",       "Unigrams, bigrams, trigrams from tokenized text"),
        ("Character n-grams",          "Useful for catching obfuscated spam (fr33, v1agra)"),
        ("Presence of spam keywords",  "win, free, click here, congratulations, urgent"),
        ("Sentiment score",            "Over-excited positive tone is common in spam"),
    ],
    "STRUCTURAL FEATURES": [
        ("Message length",             "Spam often very short (SMS) or very long (phishing)"),
        ("Number of URLs",             "Spam frequently contains suspicious links"),
        ("URL shortener detected",     "bit.ly, tinyurl = common spam tactics"),
        ("Number of exclamation marks","Excessive !! is a spam signal"),
        ("All-caps word count",        "WINNER! FREE! — caps used for urgency"),
        ("Special char ratio",         "% of $, #, @, * in the message"),
        ("HTML tag count",             "Embedded scripts/images in email spam"),
        ("Digit ratio",                "Phone numbers, amounts embedded in text"),
    ],
    "METADATA FEATURES": [
        ("Sender domain reputation",   "Known blacklisted domains / new domains"),
        ("Email header anomalies",     "Mismatched From / Reply-To fields"),
        ("Time of day sent",           "Bulk spam often sent late night / off hours"),
        ("Send rate / frequency",      "Same sender flooding with many messages"),
        ("IP geolocation",             "Unusual origin countries for the recipient"),
        ("Attachment type",            ".exe, .zip attachments are high-risk"),
    ],
}

for category, feats in features.items():
    print(f"\n  ▸ {category}")
    print(f"  {'Feature':<35} Description")
    print("  " + "─" * 70)
    for name, desc in feats:
        print(f"  {name:<35} {desc}")

# ──────────────────────────────────────────────────────────
# STEP 3 — DATA NEEDED
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("🗄️  STEP 3 — DATA NEEDED")
print("=" * 62)

print("""
  A) LABELLED DATASETS
  ─────────────────────
  • SMS Spam Collection (UCI)   — 5,574 SMS: 87% ham / 13% spam
  • SpamAssassin Public Corpus  — 6,000+ emails, used in industry
  • Enron Spam Dataset          — 33,000+ real corporate emails
  • TREC Spam Track Corpus      — large-scale email benchmark

  B) DATA REQUIREMENTS
  ─────────────────────
  • Minimum recommended: 10,000+ labelled samples
  • Class balance: ideally ~50/50 or use resampling (SMOTE)
  • Diversity: multiple languages, domains, time periods
  • Freshness: spam evolves — data must be regularly updated

  C) DATA COLLECTION METHODS
  ────────────────────────────
  • Honeypot accounts (email addresses that attract spam)
  • User-reported spam buttons
  • Web scraping public spam databases
  • Crowdsourcing labels (Amazon MTurk)
  • Synthetic augmentation for rare spam types

  D) DATA PREPROCESSING PIPELINE
  ────────────────────────────────
  Raw Text
    → Lowercase normalisation
    → Remove HTML tags / decode MIME
    → Tokenisation
    → Stop-word removal
    → Stemming / Lemmatisation
    → Vectorisation (TF-IDF / Word2Vec / BERT embeddings)
    → Feature matrix X  →  Model training
""")

# ──────────────────────────────────────────────────────────
# STEP 4 — MODEL OPTIONS
# ──────────────────────────────────────────────────────────
print("=" * 62)
print("🤖 STEP 4 — MODEL OPTIONS & COMPARISON")
print("=" * 62)

models = [
    ("Naive Bayes (MultinomialNB)", "Fast, simple, great baseline for text",
     97, 94, "Very Fast", "Low",   "★★★★☆"),
    ("Logistic Regression",        "Probabilistic, interpretable coefficients",
     97, 95, "Fast",      "Low",   "★★★★☆"),
    ("Support Vector Machine",     "Strong with high-dim TF-IDF features",
     98, 96, "Medium",    "Low",   "★★★★★"),
    ("Random Forest",              "Handles non-linear patterns, robust",
     97, 95, "Medium",    "Medium","★★★★☆"),
    ("Gradient Boosting (XGBoost)","Best structured feature performance",
     98, 97, "Slow",      "Medium","★★★★★"),
    ("BERT / Transformer",         "Understands context & semantics deeply",
     99, 98, "Very Slow", "High",  "★★★★★"),
]

print(f"\n  {'Model':<30} {'Train Acc':>9} {'Test Acc':>8} {'Speed':>10} {'Complexity':>12} {'Rating':>8}")
print("  " + "─" * 82)
for name, desc, tr, te, spd, cmp, rat in models:
    print(f"  {name:<30} {tr:>8}%  {te:>7}%  {spd:>10}  {cmp:>12}  {rat:>8}")
    print(f"  {'':30} → {desc}")
    print()

print("  ✅ Recommended: Naive Bayes for baseline → SVM/XGBoost for production")
print("  ✅ Advanced:    Fine-tuned BERT for high-stakes (phishing, fraud)")

# ──────────────────────────────────────────────────────────
# STEP 5 — POSSIBLE MISTAKES (ERRORS)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("⚠️  STEP 5 — POSSIBLE MISTAKES & HOW TO FIX THEM")
print("=" * 62)

mistakes = [
    ("FALSE NEGATIVES (Miss Rate)",
     "Spam classified as Ham",
     "DANGEROUS — spam reaches the user inbox",
     ["Lower decision threshold (e.g. 0.3 instead of 0.5)",
      "Optimise for Recall over Precision",
      "Add rule-based blocklist for known spam domains"]),
    ("FALSE POSITIVES (False Alarm)",
     "Ham classified as Spam",
     "ANNOYING — important emails go to spam folder",
     ["Raise decision threshold",
      "Whitelist trusted senders / domains",
      "Allow user feedback to retrain model"]),
    ("OVERFITTING",
     "Model memorises training data",
     "Poor generalisation on new/unseen spam",
     ["Cross-validation (k-fold)",
      "Regularisation (L1/L2)",
      "Reduce model complexity"]),
    ("CONCEPT DRIFT",
     "Spam tactics evolve over time",
     "Old model fails on new spam patterns",
     ["Retrain model periodically",
      "Monitor precision/recall on live data",
      "Use online learning algorithms"]),
    ("CLASS IMBALANCE",
     "Far more ham than spam in real data",
     "Model biased toward predicting ham",
     ["Oversample spam (SMOTE)",
      "Use class_weight='balanced' in sklearn",
      "Use F1-score not Accuracy as metric"]),
    ("FEATURE LEAKAGE",
     "Target info leaks into features",
     "Inflated accuracy that doesn't generalise",
     ["Strict train/test split before feature extraction",
      "Fit vectoriser ONLY on training data",
      "Use pipelines (sklearn Pipeline)"]),
    ("SHORT TEXT PROBLEM",
     "SMS too short for TF-IDF to work well",
     "Low vocabulary → poor classification",
     ["Use character n-grams",
      "Augment with metadata features",
      "Use pre-trained embeddings (Word2Vec)"]),
    ("ADVERSARIAL SPAM",
     "Spammers intentionally evade filters",
     "Obfuscated text bypasses keyword filters",
     ["Character-level models",
      "Image-based text detection (OCR)",
      "Ensemble models (harder to fool all)"]),
]

for i, (title, what, impact, fixes) in enumerate(mistakes, 1):
    print(f"\n  {i}. {title}")
    print(f"     What   : {what}")
    print(f"     Impact : {impact}")
    print(f"     Fixes  :")
    for fix in fixes:
        print(f"            → {fix}")

# ──────────────────────────────────────────────────────────
# STEP 6 — EVALUATION METRICS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("📊 STEP 6 — EVALUATION METRICS")
print("=" * 62)

print("""
  CONFUSION MATRIX STRUCTURE:
  ┌─────────────────┬──────────────┬──────────────┐
  │                 │ Predicted HAM│ Predicted SPAM│
  ├─────────────────┼──────────────┼──────────────┤
  │ Actual HAM      │  TN (good)   │  FP (bad!)   │
  │ Actual SPAM     │  FN (bad!)   │  TP (good)   │
  └─────────────────┴──────────────┴──────────────┘

  KEY METRICS:
  ────────────────────────────────────────────────
  Accuracy   = (TP + TN) / Total        ← misleading if imbalanced
  Precision  = TP / (TP + FP)           ← how many predicted spam are real spam
  Recall     = TP / (TP + FN)           ← how many real spam were caught
  F1-Score   = 2 × (P × R) / (P + R)   ← balance of Precision & Recall
  ROC-AUC    = area under ROC curve     ← overall model quality

  PRIORITY:
  • Recall    → prioritise if missing spam is costly (phishing)
  • Precision → prioritise if false alarms are costly (enterprise)
  • F1-Score  → best single metric for spam in general
""")

# ──────────────────────────────────────────────────────────
# STEP 7 — SIMULATION
# ──────────────────────────────────────────────────────────
print("=" * 62)
print("🔬 STEP 7 — SIMULATED CLASSIFIER DEMO")
print("=" * 62)

spam_keywords = ['free', 'win', 'click', 'urgent', 'congratulations',
                 'offer', 'prize', 'cash', 'limited', 'act now',
                 'buy now', 'guarantee', 'discount', '!!!', '$$$']

def simple_spam_classifier(text):
    text_lower = text.lower()
    score = 0
    matched = []
    for kw in spam_keywords:
        if kw in text_lower:
            score += 1
            matched.append(kw)
    if len(text) < 30:   score += 1
    if text.count('!') > 2: score += 1
    if sum(1 for c in text if c.isupper()) / max(len(text),1) > 0.4: score += 1
    label = 'SPAM 🚨' if score >= 2 else 'HAM  ✅'
    confidence = min(0.5 + score * 0.12, 0.99)
    return label, confidence, matched

test_messages = [
    "Congratulations! You've WON a FREE iPhone! Click NOW to claim!!!",
    "Hi John, please review the attached project report by Friday.",
    "URGENT: Your account will be closed! Act now and click here!",
    "Can we reschedule our meeting to 3pm tomorrow?",
    "Buy now and get 80% discount! Limited offer. Guaranteed cash prize!!!",
    "Hey, are you joining us for dinner tonight?",
    "You have been selected for a FREE vacation. Claim your prize today!",
    "Please find attached the invoice for last month's services.",
]

print(f"\n  {'Message':<52} {'Label':<10} {'Confidence':>10}  Matched Keywords")
print("  " + "─" * 100)
for msg in test_messages:
    label, conf, kws = simple_spam_classifier(msg)
    short = msg[:50] + ".." if len(msg) > 50 else msg
    kw_str = ", ".join(kws[:3]) if kws else "—"
    print(f"  {short:<52} {label:<10} {conf:>9.0%}  {kw_str}")

print("\n  Note: This is a rule-based demo. Real classifiers use ML models.")

# ──────────────────────────────────────────────────────────
# STEP 8 — VISUALISATIONS
# ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

def style_ax(ax, title, color=GREEN):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_title(title, color=color, fontsize=9, fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(color=BORDER, linewidth=0.5, alpha=0.7)

# 1. Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
cm = np.array([[950, 50], [30, 970]])
im = ax1.imshow(cm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1000)
for i in range(2):
    for j in range(2):
        color = 'black' if cm[i,j] > 600 else TEXT
        ax1.text(j, i, str(cm[i,j]), ha='center', va='center',
                 fontsize=14, fontweight='bold', color=color)
ax1.set_xticks([0,1]); ax1.set_xticklabels(['Pred HAM','Pred SPAM'], fontsize=8)
ax1.set_yticks([0,1]); ax1.set_yticklabels(['Actual HAM','Actual SPAM'], fontsize=8)
ax1.set_title('Confusion Matrix', color=GREEN, fontsize=9, fontweight='bold', pad=10)
for spine in ax1.spines.values(): spine.set_edgecolor(BORDER)
ax1.tick_params(colors=MUTED)

# 2. Feature Importance (simulated)
ax2 = fig.add_subplot(gs[0, 1])
feat_names = ['TF-IDF\nKeywords', 'URL\nCount', 'Excl.\nMarks', 'All\nCAPS',
              'Message\nLength', 'Sender\nRep', 'Special\nChars', 'Send\nRate']
importances = [0.28, 0.18, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04]
colors_bar = [RED if v > 0.15 else ORANGE if v > 0.10 else YELLOW for v in importances]
bars = ax2.barh(feat_names, importances, color=colors_bar, edgecolor=BG, linewidth=0.5)
for bar, val in zip(bars, importances):
    ax2.text(val+0.003, bar.get_y()+bar.get_height()/2,
             f'{val:.0%}', va='center', color=TEXT, fontsize=7)
ax2.set_xlabel('Importance Score')
style_ax(ax2, 'Feature Importance', RED)

# 3. ROC Curve
ax3 = fig.add_subplot(gs[0, 2])
fpr = np.linspace(0, 1, 100)
tpr_nb  = np.clip(1 - (1-fpr)**3.5, 0, 1)
tpr_svm = np.clip(1 - (1-fpr)**5.0, 0, 1)
tpr_bert= np.clip(1 - (1-fpr)**7.0, 0, 1)
ax3.plot(fpr, tpr_bert, color=GREEN,  linewidth=2, label='BERT (AUC=0.99)')
ax3.plot(fpr, tpr_svm,  color=ORANGE, linewidth=2, label='SVM  (AUC=0.97)')
ax3.plot(fpr, tpr_nb,   color=BLUE,   linewidth=2, label='NB   (AUC=0.94)')
ax3.plot([0,1],[0,1], '--', color=MUTED, linewidth=1, label='Random')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
style_ax(ax3, 'ROC Curve Comparison', GREEN)

# 4. Class distribution
ax4 = fig.add_subplot(gs[1, 0])
labels_pie = ['HAM (87%)', 'SPAM (13%)']
sizes = [87, 13]
wedge_colors = [GREEN, RED]
wedges, texts, autotexts = ax4.pie(
    sizes, labels=labels_pie, colors=wedge_colors,
    autopct='%1.0f%%', startangle=90,
    wedgeprops=dict(edgecolor=BG, linewidth=2),
    textprops=dict(color=TEXT, fontsize=8)
)
for at in autotexts: at.set_color(BG); at.set_fontweight('bold')
ax4.set_title('Dataset Class Balance\n(SMS Spam Collection)', color=YELLOW,
              fontsize=9, fontweight='bold')

# 5. Error types impact
ax5 = fig.add_subplot(gs[1, 1])
error_types = ['False\nNegative\n(Miss)', 'False\nPositive\n(False Alarm)',
               'Concept\nDrift', 'Class\nImbalance', 'Feature\nLeakage', 'Adversarial\nSpam']
severity = [9, 6, 8, 7, 8, 9]
cols = [RED if s>=8 else ORANGE if s>=6 else YELLOW for s in severity]
bars5 = ax5.bar(error_types, severity, color=cols, edgecolor=BG, linewidth=0.5)
for bar, val in zip(bars5, severity):
    ax5.text(bar.get_x()+bar.get_width()/2, val+0.1, str(val)+'/10',
             ha='center', va='bottom', color=TEXT, fontsize=7, fontweight='bold')
ax5.set_ylim(0, 11)
ax5.set_ylabel('Severity (1–10)')
ax5.tick_params(axis='x', labelsize=7)
style_ax(ax5, 'Error Severity Rating', RED)

# 6. Metrics comparison across models
ax6 = fig.add_subplot(gs[1, 2])
model_names = ['Naive\nBayes', 'Logistic\nReg', 'SVM', 'XGBoost', 'BERT']
precision_scores = [0.93, 0.95, 0.96, 0.97, 0.98]
recall_scores    = [0.91, 0.93, 0.95, 0.96, 0.99]
f1_scores        = [0.92, 0.94, 0.955, 0.965, 0.985]
x = np.arange(len(model_names))
w = 0.25
ax6.bar(x-w, precision_scores, w, label='Precision', color=BLUE,   edgecolor=BG)
ax6.bar(x,   recall_scores,    w, label='Recall',    color=ORANGE,  edgecolor=BG)
ax6.bar(x+w, f1_scores,        w, label='F1-Score',  color=GREEN,   edgecolor=BG)
ax6.set_xticks(x); ax6.set_xticklabels(model_names, fontsize=7)
ax6.set_ylim(0.85, 1.01)
ax6.set_ylabel('Score')
ax6.legend(fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
style_ax(ax6, 'Model Metrics Comparison', BLUE)

# 7. Demo classifier results (bar)
ax7 = fig.add_subplot(gs[2, :2])
msg_labels = [f'Msg {i+1}' for i in range(len(test_messages))]
confidences = []
is_spam = []
for msg in test_messages:
    label, conf, _ = simple_spam_classifier(msg)
    confidences.append(conf)
    is_spam.append('SPAM' in label)
bar_colors = [RED if s else GREEN for s in is_spam]
bars7 = ax7.bar(msg_labels, confidences, color=bar_colors, edgecolor=BG, linewidth=0.5)
ax7.axhline(0.5, color=YELLOW, linewidth=1.2, linestyle='--', label='Decision Threshold (0.5)')
for bar, val, spam in zip(bars7, confidences, is_spam):
    tag = 'SPAM' if spam else 'HAM'
    ax7.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{tag}\n{val:.0%}',
             ha='center', va='bottom', color=TEXT, fontsize=7, fontweight='bold')
ax7.set_ylim(0, 1.15)
ax7.set_ylabel('Confidence Score')
ax7.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
spam_patch = mpatches.Patch(color=RED,   label='SPAM')
ham_patch  = mpatches.Patch(color=GREEN, label='HAM')
ax7.legend(handles=[spam_patch, ham_patch,
           mpatches.Patch(color=YELLOW, label='Threshold')],
           fontsize=7, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
style_ax(ax7, 'Demo: Rule-based Classifier on Test Messages', ORANGE)

# 8. Pipeline diagram (text-based in plot)
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor(CARD)
ax8.set_xlim(0,10); ax8.set_ylim(0,10)
ax8.axis('off')
ax8.set_title('ML Pipeline', color=GREEN, fontsize=9, fontweight='bold', pad=10)
for spine in ax8.spines.values(): spine.set_edgecolor(BORDER)

steps_pipe = [
    ('Raw Text', RED),
    ('Preprocess', ORANGE),
    ('Vectorise', YELLOW),
    ('Train Model', GREEN),
    ('Evaluate', BLUE),
    ('Deploy', PURPLE),
]
for i, (label, col) in enumerate(steps_pipe):
    y = 9 - i*1.5
    rect = FancyBboxPatch((1, y-0.45), 8, 0.9,
                           boxstyle="round,pad=0.1",
                           facecolor=col+'22', edgecolor=col, linewidth=1.5)
    ax8.add_patch(rect)
    ax8.text(5, y, label, ha='center', va='center',
             color=col, fontsize=9, fontweight='bold')
    if i < len(steps_pipe)-1:
        ax8.annotate('', xy=(5, y-0.55), xytext=(5, y-0.45),
                     arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.2))

fig.suptitle('Spam Classifier — System Design Thinking',
             color=TEXT, fontsize=14, fontweight='bold', y=0.99)

import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spam_classifier_thinking.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("\n" + "=" * 62)
print("✅ COMPLETE!")
print(f"   Plots saved → spam_classifier_thinking.png")
print("=" * 62)