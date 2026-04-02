# ============================================================
#                    DATA DOCTOR
#   Clean a Dataset — Missing Values | Duplicates |
#   Standardize Text | Outliers | Explain Why It Matters
# ============================================================

import os
import copy

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1 — RAW (DIRTY) DATASET
# ─────────────────────────────────────────────

raw_data = [
    # id  name              age    city           marks   grade   email
    [1,  "Aarav Sharma",    20,    "Mumbai",       85,    "A",   "aarav@gmail.com"],
    [2,  "priya verma",     None,  "delhi",        92,    "a+",  "priya@gmail.com"],
    [3,  "ROHIT SINGH",     21,    "Bangalore",    None,  "B",   "rohit@gmail.com"],
    [4,  "Sneha Patil",     22,    "pune",         78,    "b+",  "sneha@gmail.com"],
    [5,  "  Karan Mehta  ", 19,    "MUMBAI",       55,    "C",   "karan@gmail.com"],
    [6,  "priya verma",     None,  "delhi",        92,    "a+",  "priya@gmail.com"],  # duplicate
    [7,  "Divya Nair",      23,    "Chennai",      None,  "B",   "divya@gmail.com"],
    [8,  "ARJUN REDDY",     21,    "hyderabad",    73,    "b",   "arjun@gmail.com"],
    [9,  "Meera Iyer",      None,  "Chennai",      88,    "A",   "meera@gmail.com"],
    [10, "Vikram Das",      25,    "Kolkata",      150,   "A+",  "vikram@gmail.com"], # outlier marks
    [11, "  pooja Gupta  ", 22,    "PUNE",         61,    "c+",  "pooja@gmail.com"],
    [12, "Rahul Joshi",     20,    "Mumbai",       79,    "B+",  "rahul@gmail.com"],
    [13, "ROHIT SINGH",     21,    "Bangalore",    None,  "B",   "rohit@gmail.com"],  # duplicate
    [14, "Ananya Roy",      None,  "Kolkata",      83,    "A",   "ananya@gmail.com"],
    [15, "Nikhil Pillai",   24,    "bangalore",    -10,   "F",   "nikhil@gmail.com"], # outlier marks
    [16, "Swati Tiwari",    21,    "Jaipur",       None,  "B",   "swati@gmail.com"],
    [17, "Aditya Kumar",    20,    "jaipur",       90,    "A+",  "aditya@gmail.com"],
    [18, "kavya menon",     23,    "Chennai",      76,    "B",   "kavya@gmail.com"],
    [19, "Sahil Sheikh",    None,  "Hyderabad",    68,    "C+",  "sahil@gmail.com"],
    [20, "Riya Kapoor",     22,    "Mumbai",       None,  "B+",  "riya@gmail.com"],
]

COLUMNS = ["ID", "Name", "Age", "City", "Marks", "Grade", "Email"]

# ─────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────

def print_table(data, title):
    print(f"\n  {title}")
    print("  " + "─" * 82)
    print(f"  {'ID':<4} {'Name':<20} {'Age':<5} {'City':<12} "
          f"{'Marks':<7} {'Grade':<7} {'Email'}")
    print("  " + "─" * 82)
    for row in data:
        id_, name, age, city, marks, grade, email = row
        print(f"  {str(id_):<4} {str(name):<20} {str(age):<5} {str(city):<12} "
              f"{str(marks):<7} {str(grade):<7} {str(email)}")
    print(f"  {'─'*82}")
    print(f"  Total rows: {len(data)}\n")

def section(title):
    print("\n" + "=" * 82)
    print(f"  {title}")
    print("=" * 82)

def mean_of(values):
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 1) if valid else None

# ─────────────────────────────────────────────
# STEP 2 — PRINT RAW DATA
# ─────────────────────────────────────────────

section("STEP 1 — RAW (DIRTY) DATASET")
print_table(raw_data, "Original dataset with all problems intact:")

# Deep copy for cleaning
data = copy.deepcopy(raw_data)

# Track all issues
report = {
    "missing_filled_age"   : [],
    "missing_filled_marks" : [],
    "duplicates_removed"   : [],
    "text_standardized"    : [],
    "outliers_fixed"       : [],
}

# ─────────────────────────────────────────────
# STEP 3 — DETECT ALL ISSUES FIRST
# ─────────────────────────────────────────────

section("STEP 2 — DIAGNOSIS (Detecting All Issues)")

# Missing values
missing_age   = [row[1] for row in data if row[2] is None]
missing_marks = [row[1] for row in data if row[4] is None]
print(f"\n  MISSING VALUES:")
print(f"    Age   missing in : {missing_age}")
print(f"    Marks missing in : {missing_marks}")

# Duplicates (compare name + city)
seen, dup_ids = {}, []
for row in data:
    key = (row[1].strip().lower(), str(row[4]))
    if key in seen:
        dup_ids.append(row[0])
    else:
        seen[key] = row[0]
print(f"\n  DUPLICATES:")
print(f"    Row IDs flagged as duplicates : {dup_ids}")

# Text issues
text_issues = []
for row in data:
    name = row[1]; city = row[3]; grade = row[5]
    if name != name.strip() or name != name.title():
        text_issues.append(f"Name '{name}' (ID {row[0]})")
    if city != city.title():
        text_issues.append(f"City '{city}' (ID {row[0]})")
    if grade != grade.upper():
        text_issues.append(f"Grade '{grade}' (ID {row[0]})")
print(f"\n  TEXT / FORMAT ISSUES:")
for t in text_issues:
    print(f"    {t}")

# Outliers
outliers = [(row[0], row[1], row[4]) for row in data
            if row[4] is not None and (row[4] > 100 or row[4] < 0)]
print(f"\n  OUTLIERS IN MARKS (valid range 0–100):")
for oid, oname, omarks in outliers:
    print(f"    ID {oid} ({oname}) — Marks = {omarks}  ← INVALID")

# ─────────────────────────────────────────────
# STEP 4 — CLEAN : STANDARDIZE TEXT
# ─────────────────────────────────────────────

section("STEP 3 — CLEANING: Standardize Text (Name, City, Grade)")

for row in data:
    old_name  = row[1];  row[1] = row[1].strip().title()
    old_city  = row[3];  row[3] = row[3].strip().title()
    old_grade = row[5];  row[5] = row[5].strip().upper()
    if old_name != row[1] or old_city != row[3] or old_grade != row[5]:
        report["text_standardized"].append(
            f"ID {row[0]}: Name '{old_name}'->'{row[1]}'  "
            f"City '{old_city}'->'{row[3]}'  Grade '{old_grade}'->'{row[5]}'"
        )

print(f"\n  Changes made ({len(report['text_standardized'])}):")
for change in report["text_standardized"]:
    print(f"    {change}")

# ─────────────────────────────────────────────
# STEP 5 — CLEAN : REMOVE DUPLICATES
# ─────────────────────────────────────────────

section("STEP 4 — CLEANING: Remove Duplicates")

before_count = len(data)
seen_keys    = {}
cleaned      = []
for row in data:
    key = (row[1].lower(), row[3].lower())
    if key not in seen_keys:
        seen_keys[key] = True
        cleaned.append(row)
    else:
        report["duplicates_removed"].append(
            f"ID {row[0]} — '{row[1]}' from {row[3]} (exact duplicate)")

data = cleaned
print(f"\n  Rows before : {before_count}")
print(f"  Duplicates  : {len(report['duplicates_removed'])}")
for d in report["duplicates_removed"]:
    print(f"    Removed -> {d}")
print(f"  Rows after  : {len(data)}")

# ─────────────────────────────────────────────
# STEP 6 — CLEAN : HANDLE MISSING VALUES
# ─────────────────────────────────────────────

section("STEP 5 — CLEANING: Handle Missing Values (Fill with Mean)")

avg_age   = mean_of([row[2] for row in data])
avg_marks = mean_of([row[4] for row in data])

print(f"\n  Strategy  : Fill missing numeric values with column mean")
print(f"  Mean Age  : {avg_age}")
print(f"  Mean Marks: {avg_marks}\n")

for row in data:
    if row[2] is None:
        report["missing_filled_age"].append(
            f"ID {row[0]} ({row[1]}) Age: None -> {avg_age}")
        row[2] = avg_age
    if row[4] is None:
        report["missing_filled_marks"].append(
            f"ID {row[0]} ({row[1]}) Marks: None -> {avg_marks}")
        row[4] = avg_marks

print(f"  Age filled ({len(report['missing_filled_age'])}):")
for m in report["missing_filled_age"]:
    print(f"    {m}")
print(f"\n  Marks filled ({len(report['missing_filled_marks'])}):")
for m in report["missing_filled_marks"]:
    print(f"    {m}")

# ─────────────────────────────────────────────
# STEP 7 — CLEAN : FIX OUTLIERS
# ─────────────────────────────────────────────

section("STEP 6 — CLEANING: Fix Outliers (Clamp Marks to 0–100)")

for row in data:
    if row[4] is not None and (row[4] > 100 or row[4] < 0):
        old_val = row[4]
        row[4]  = max(0, min(100, row[4]))
        report["outliers_fixed"].append(
            f"ID {row[0]} ({row[1]}): Marks {old_val} -> {row[4]} (clamped)")

print(f"\n  Outliers fixed ({len(report['outliers_fixed'])}):")
for o in report["outliers_fixed"]:
    print(f"    {o}")

# ─────────────────────────────────────────────
# STEP 8 — PRINT CLEAN DATASET
# ─────────────────────────────────────────────

section("STEP 7 — CLEAN DATASET (Final Result)")
print_table(data, "Fully cleaned dataset:")

# ─────────────────────────────────────────────
# STEP 9 — CLEANING REPORT SUMMARY
# ─────────────────────────────────────────────

section("STEP 8 — CLEANING REPORT SUMMARY")

total_fixes = (len(report["text_standardized"]) +
               len(report["duplicates_removed"]) +
               len(report["missing_filled_age"]) +
               len(report["missing_filled_marks"]) +
               len(report["outliers_fixed"]))

print(f"""
  ┌─────────────────────────────────────────┬──────┐
  │  Issue Type                             │ Count│
  ├─────────────────────────────────────────┼──────┤
  │  Text standardized (Name/City/Grade)    │  {len(report['text_standardized']):>3}  │
  │  Duplicate rows removed                 │  {len(report['duplicates_removed']):>3}  │
  │  Missing Age values filled              │  {len(report['missing_filled_age']):>3}  │
  │  Missing Marks values filled            │  {len(report['missing_filled_marks']):>3}  │
  │  Outlier values clamped                 │  {len(report['outliers_fixed']):>3}  │
  ├─────────────────────────────────────────┼──────┤
  │  TOTAL FIXES APPLIED                    │  {total_fixes:>3}  │
  └─────────────────────────────────────────┴──────┘

  Raw dataset rows  : {len(raw_data)}
  Clean dataset rows: {len(data)}
  Rows removed      : {len(raw_data) - len(data)} (duplicates)
""")

# ─────────────────────────────────────────────
# STEP 10 — WHY DATA CLEANING MATTERS
# ─────────────────────────────────────────────

section("STEP 9 — WHY DATA CLEANING MATTERS")
print("""
  "Garbage In, Garbage Out" — the most fundamental rule in ML.

  No matter how powerful your ML algorithm is, it will produce
  WRONG, BIASED, or UNRELIABLE results if the data fed into it
  is dirty. Here's why each cleaning step is critical:

  ┌────────────────────────┬────────────────────────────────────────┐
  │  Problem               │  Why It Matters                        │
  ├────────────────────────┼────────────────────────────────────────┤
  │  Missing Values        │  Most ML models cannot process None/   │
  │                        │  NaN. They crash or silently ignore    │
  │                        │  rows, leading to biased results.      │
  ├────────────────────────┼────────────────────────────────────────┤
  │  Duplicates            │  Duplicate rows make the model think   │
  │                        │  certain patterns are more common than │
  │                        │  they are, skewing predictions.        │
  ├────────────────────────┼────────────────────────────────────────┤
  │  Inconsistent Text     │  'mumbai', 'MUMBAI', 'Mumbai' are      │
  │                        │  treated as 3 DIFFERENT cities by ML.  │
  │                        │  This creates false categories.        │
  ├────────────────────────┼────────────────────────────────────────┤
  │  Outliers              │  A marks value of 150 or -10 corrupts  │
  │                        │  mean calculations and misleads        │
  │                        │  regression and clustering models.     │
  └────────────────────────┴────────────────────────────────────────┘

  REAL-WORLD IMPACT:
  * Healthcare  : A dirty dataset could cause a model to misdiagnose
                  patients because of missing lab values.
  * Finance     : Duplicate transactions inflate reported revenue,
                  misleading fraud detection models.
  * E-Commerce  : Inconsistent city names cause wrong delivery zone
                  predictions and logistics failures.

  RULE OF THUMB: Data Scientists spend 60-80% of their time
  cleaning and preparing data — because clean data is the
  foundation of every trustworthy AI/ML model.
""")

# ─────────────────────────────────────────────
# STEP 11 — WRITE OUTPUT FILES
# ─────────────────────────────────────────────

section("STEP 10 — SAVING OUTPUT FILES")

# ── Clean dataset as CSV ──
csv_path = os.path.join(SAVE_DIR, "clean_dataset.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(",".join(COLUMNS) + "\n")
    for row in data:
        f.write(",".join(str(v) for v in row) + "\n")
print(f"\n  Clean dataset CSV  -> {csv_path}")

# ── Cleaning report as TXT ──
txt_lines = []
txt_lines.append("=" * 70)
txt_lines.append("        DATA DOCTOR — CLEANING REPORT")
txt_lines.append("=" * 70)
txt_lines.append(f"\n  Raw rows    : {len(raw_data)}")
txt_lines.append(f"  Clean rows  : {len(data)}")
txt_lines.append(f"  Total fixes : {total_fixes}\n")

txt_lines.append("  TEXT STANDARDIZED:")
for t in report["text_standardized"]:
    txt_lines.append(f"    {t}")
txt_lines.append("\n  DUPLICATES REMOVED:")
for d in report["duplicates_removed"]:
    txt_lines.append(f"    {d}")
txt_lines.append("\n  MISSING AGE FILLED:")
for m in report["missing_filled_age"]:
    txt_lines.append(f"    {m}")
txt_lines.append("\n  MISSING MARKS FILLED:")
for m in report["missing_filled_marks"]:
    txt_lines.append(f"    {m}")
txt_lines.append("\n  OUTLIERS FIXED:")
for o in report["outliers_fixed"]:
    txt_lines.append(f"    {o}")

txt_lines.append("\n" + "=" * 70)
txt_lines.append("  WHY DATA CLEANING MATTERS:")
txt_lines.append("=" * 70)
txt_lines.append("""
  Missing Values  -> ML models crash or produce biased results
  Duplicates      -> Model overfits to repeated patterns
  Inconsistent Text -> Creates false categories (Mumbai != MUMBAI)
  Outliers        -> Corrupts mean/regression/clustering results

  'Garbage In, Garbage Out' — clean data is the foundation
  of every reliable AI/ML model. Data scientists spend 60-80%
  of their time on data cleaning for this reason.
""")
txt_lines.append("=" * 70)

rpt_path = os.path.join(SAVE_DIR, "cleaning_report.txt")
with open(rpt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(txt_lines))
print(f"  Cleaning report TXT-> {rpt_path}")

print(f"\n  All done! Folder: {SAVE_DIR}\n")