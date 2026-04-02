# ============================================================
#           BUILD YOUR FIRST DATASET
#   Study Hours vs Marks — Features, Labels & Prediction
# ============================================================

import numpy as np
import os

# ── Save folder — same folder as this script ──
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# STEP 1 — CREATE THE DATASET
# ─────────────────────────────────────────────

# Features (Inputs / X)
student_names   = [
    "Aarav", "Priya", "Rohit", "Sneha", "Karan",
    "Divya", "Arjun", "Meera", "Vikram", "Pooja",
    "Rahul", "Ananya", "Nikhil", "Swati", "Aditya",
    "Kavya", "Sahil", "Riya", "Harish", "Nisha"
]

study_hours     = [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6,
                   6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]

sleep_hours     = [5, 5, 6, 6, 7, 6, 7, 7, 8, 7,
                   8, 7, 8, 8, 7, 9, 8, 9, 8, 9]

attendance_pct  = [55, 60, 62, 65, 68, 70, 72, 75, 76, 80,
                   82, 84, 85, 87, 88, 90, 91, 93, 95, 97]

# Label (Output / Y)
marks           = [32, 40, 45, 50, 54, 58, 62, 65, 67, 70,
                   73, 75, 77, 80, 82, 85, 87, 90, 93, 96]

n = len(student_names)

# ─────────────────────────────────────────────
# STEP 2 — DISPLAY THE DATASET
# ─────────────────────────────────────────────

def print_dataset():
    print("\n" + "=" * 72)
    print("                    FULL DATASET (20 Students)")
    print("=" * 72)
    print(f"  {'#':<3}  {'Name':<10}  {'Study Hrs':>9}  {'Sleep Hrs':>9}  "
          f"{'Attendance%':>11}  {'Marks':>6}")
    print("  " + "-" * 66)
    for i in range(n):
        print(f"  {i+1:<3}  {student_names[i]:<10}  {study_hours[i]:>9.1f}  "
              f"{sleep_hours[i]:>9}  {attendance_pct[i]:>11}  {marks[i]:>6}")
    print("=" * 72)

# ─────────────────────────────────────────────
# STEP 3 — FEATURES & LABELS EXPLANATION
# ─────────────────────────────────────────────

def print_features_labels():
    print("\n" + "=" * 72)
    print("           FEATURES (X — Inputs) vs LABEL (Y — Output)")
    print("=" * 72)
    print("""
  FEATURES  : These are the INPUT variables we feed into the ML model.
               They are the factors that may INFLUENCE the outcome.

    Feature 1 — study_hours    : Hours spent studying per day
    Feature 2 — sleep_hours    : Hours of sleep per night
    Feature 3 — attendance_pct : Attendance percentage in class

  LABEL     : This is the OUTPUT variable the model tries to PREDICT.

    Label     — marks          : Exam marks scored (out of 100)

  WHY THIS MATTERS:
    In supervised learning, the model learns the RELATIONSHIP between
    features (X) and the label (Y) using historical data.
    Once trained, it can PREDICT marks for a new student whose
    study hours, sleep, and attendance are known.

  NOTATION:
    X  =  [ study_hours,  sleep_hours,  attendance_pct ]
    Y  =  [ marks ]
    Goal: Learn a function  f(X) -> Y
""")
    print("=" * 72)

# ─────────────────────────────────────────────
# STEP 4 — BASIC STATISTICS
# ─────────────────────────────────────────────

def print_statistics():
    sh  = np.array(study_hours)
    sl  = np.array(sleep_hours)
    att = np.array(attendance_pct)
    mk  = np.array(marks)

    print("\n" + "=" * 72)
    print("                   DATASET STATISTICS")
    print("=" * 72)
    print(f"  {'Column':<18}  {'Min':>6}  {'Max':>6}  {'Mean':>7}  {'Std Dev':>8}")
    print("  " + "-" * 52)
    for name, arr in [("Study Hours", sh), ("Sleep Hours", sl),
                      ("Attendance %", att), ("Marks", mk)]:
        print(f"  {name:<18}  {arr.min():>6.1f}  {arr.max():>6.1f}  "
              f"{arr.mean():>7.2f}  {arr.std():>8.2f}")
    print("=" * 72)

    # Correlation
    corr_study   = np.corrcoef(sh, mk)[0, 1]
    corr_sleep   = np.corrcoef(sl, mk)[0, 1]
    corr_attend  = np.corrcoef(att, mk)[0, 1]

    print("\n  CORRELATION WITH MARKS (how strongly each feature relates):")
    print(f"    Study Hours   vs Marks : {corr_study:.4f}  "
          + ("(Very Strong)" if abs(corr_study)  > 0.9 else "(Strong)"))
    print(f"    Sleep Hours   vs Marks : {corr_sleep:.4f}  "
          + ("(Moderate)" if abs(corr_sleep)  > 0.5 else "(Weak)"))
    print(f"    Attendance %  vs Marks : {corr_attend:.4f}  "
          + ("(Very Strong)" if abs(corr_attend) > 0.9 else "(Strong)"))
    print()
    best = max([("Study Hours", corr_study),
                ("Sleep Hours", corr_sleep),
                ("Attendance %", corr_attend)],
               key=lambda x: abs(x[1]))
    print(f"  >>> Best predictor of Marks : {best[0]} (r = {best[1]:.4f})")
    print("=" * 72)

# ─────────────────────────────────────────────
# STEP 5 — SIMPLE LINEAR REGRESSION (from scratch)
# ─────────────────────────────────────────────

def linear_regression(x, y):
    """Compute slope (m) and intercept (b) using least squares formula."""
    x, y  = np.array(x, dtype=float), np.array(y, dtype=float)
    n     = len(x)
    m     = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
            (n * np.sum(x**2) - np.sum(x)**2)
    b     = (np.sum(y) - m * np.sum(x)) / n
    return m, b

def predict(x_val, m, b):
    return round(m * x_val + b, 2)

def r_squared(x, y, m, b):
    x, y   = np.array(x, dtype=float), np.array(y, dtype=float)
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def print_model():
    m, b = linear_regression(study_hours, marks)
    r2   = r_squared(study_hours, marks, m, b)

    print("\n" + "=" * 72)
    print("       LINEAR REGRESSION MODEL  (Study Hours -> Marks)")
    print("=" * 72)
    print(f"""
  Formula : Marks  =  m * Study_Hours  +  b

  Trained values:
    Slope (m)     = {m:.4f}   <- each extra study hour adds ~{m:.1f} marks
    Intercept (b) = {b:.4f}   <- base marks with 0 study hours
    R² Score      = {r2:.4f}   <- model explains {r2*100:.1f}% of variation

  Equation : Marks  =  {m:.2f} * Study_Hours  +  ({b:.2f})
""")
    print("  " + "-" * 68)
    print("  PREDICTIONS on training data:")
    print(f"  {'Name':<10}  {'Study Hrs':>9}  {'Actual':>7}  {'Predicted':>10}  {'Error':>7}")
    print("  " + "-" * 50)
    for i in range(n):
        pred  = predict(study_hours[i], m, b)
        error = marks[i] - pred
        print(f"  {student_names[i]:<10}  {study_hours[i]:>9.1f}  "
              f"{marks[i]:>7}  {pred:>10.2f}  {error:>+7.2f}")
    print("=" * 72)
    return m, b

# ─────────────────────────────────────────────
# STEP 6 — PREDICT FOR NEW STUDENTS
# ─────────────────────────────────────────────

def print_new_predictions(m, b):
    new_students = [
        ("Deepak",  3.0),
        ("Lakshmi", 6.5),
        ("Suresh",  9.0),
        ("Preethi", 1.5),
        ("Mohan",  11.5),
    ]
    print("\n" + "=" * 72)
    print("          PREDICTING MARKS FOR NEW (UNSEEN) STUDENTS")
    print("=" * 72)
    print(f"\n  {'Name':<10}  {'Study Hrs':>9}  {'Predicted Marks':>16}  Grade")
    print("  " + "-" * 48)
    for name, hrs in new_students:
        pred  = max(0, min(100, predict(hrs, m, b)))
        grade = ("A+" if pred >= 90 else "A"  if pred >= 80 else
                 "B"  if pred >= 70 else "C"  if pred >= 60 else
                 "D"  if pred >= 50 else "F")
        print(f"  {name:<10}  {hrs:>9.1f}  {pred:>16.2f}  {grade}")
    print()
    print("  Note: Predicted marks are clamped between 0 and 100.")
    print("=" * 72)

# ─────────────────────────────────────────────
# STEP 7 — ASCII SCATTER PLOT
# ─────────────────────────────────────────────

def ascii_scatter():
    print("\n" + "=" * 72)
    print("        ASCII SCATTER PLOT  (Study Hours vs Marks)")
    print("=" * 72)

    rows, cols = 20, 55
    grid       = [[" "] * cols for _ in range(rows)]

    sh_arr = np.array(study_hours)
    mk_arr = np.array(marks)

    for i in range(n):
        col = int((study_hours[i] - sh_arr.min()) /
                  (sh_arr.max() - sh_arr.min()) * (cols - 1))
        row = rows - 1 - int((marks[i] - mk_arr.min()) /
                              (mk_arr.max() - mk_arr.min()) * (rows - 1))
        row = max(0, min(rows-1, row))
        col = max(0, min(cols-1, col))
        grid[row][col] = "●"

    y_labels = np.linspace(mk_arr.max(), mk_arr.min(), rows)
    print()
    for r, (row, yl) in enumerate(zip(grid, y_labels)):
        print(f"  {int(yl):>3} |{''.join(row)}")

    x_axis  = "─" * cols
    print(f"       └{x_axis}")
    print(f"        {sh_arr.min():.0f}hr"
          + " " * (cols - 8) + f"{sh_arr.max():.0f}hr")
    print(f"\n         X-axis: Study Hours    Y-axis: Marks\n")
    print("  Trend: As study hours increase, marks increase (positive correlation)")
    print("=" * 72)

# ─────────────────────────────────────────────
# STEP 8 — WRITE DATASET TO TEXT FILE
# ─────────────────────────────────────────────

def write_dataset_file():
    lines = []
    lines.append("=" * 72)
    lines.append("          BUILD YOUR FIRST DATASET")
    lines.append("          Study Hours vs Marks — 20 Students")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  {'#':<3}  {'Name':<10}  {'Study Hrs':>9}  {'Sleep Hrs':>9}  "
                 f"{'Attendance%':>11}  {'Marks':>6}")
    lines.append("  " + "-" * 56)
    for i in range(n):
        lines.append(
            f"  {i+1:<3}  {student_names[i]:<10}  {study_hours[i]:>9.1f}  "
            f"{sleep_hours[i]:>9}  {attendance_pct[i]:>11}  {marks[i]:>6}"
        )
    lines.append("")
    lines.append("=" * 72)
    lines.append("  FEATURES (X — Inputs):")
    lines.append("    1. study_hours    — Hours studied per day")
    lines.append("    2. sleep_hours    — Hours of sleep per night")
    lines.append("    3. attendance_pct — Attendance percentage")
    lines.append("")
    lines.append("  LABEL (Y — Output):")
    lines.append("    marks             — Exam marks scored out of 100")
    lines.append("")
    lines.append("  RELATIONSHIP:")
    lines.append("    More study hours -> Higher marks (strong positive correlation)")
    lines.append("    Better sleep     -> Slightly better marks (moderate)")
    lines.append("    Higher attendance-> Higher marks (strong positive correlation)")
    lines.append("=" * 72)

    path = os.path.join(SAVE_DIR, "student_dataset.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Dataset text file saved -> {path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 72)
    print("            BUILD YOUR FIRST DATASET")
    print("     Study Hours vs Marks — Features, Labels & Prediction")
    print("=" * 72)

    print_dataset()
    print_features_labels()
    print_statistics()
    m, b = print_model()
    print_new_predictions(m, b)
    ascii_scatter()
    write_dataset_file()

    print(f"\n  Files saved in: {SAVE_DIR}")
    print("  student_dataset.txt — full dataset + explanation\n")

if __name__ == "__main__":
    main()