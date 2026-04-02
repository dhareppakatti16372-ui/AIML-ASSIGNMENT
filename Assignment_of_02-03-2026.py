# ============================================================
#               ML IDEA GENERATOR
#   ML Problems in College | Healthcare | Shopping
#   Input -> Output descriptions for each
# ============================================================

import os

# ── Save folder — same folder as this script ──
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# DATA — All ML Ideas
# ─────────────────────────────────────────────

ideas = {
    "COLLEGE / EDUCATION": [
        {
            "title"  : "Student Dropout Prediction",
            "problem": "Predict which students are at risk of dropping out so counselors can intervene early.",
            "ml_type": "Binary Classification (Supervised Learning)",
            "input"  : "Attendance %, assignment submission rate, mid-term scores, LMS login frequency, socioeconomic background, backlogs",
            "output" : "Dropout Risk = High / Medium / Low + Top 3 contributing reasons",
            "impact" : "Colleges proactively assign mentors and reduce dropout rates."
        },
        {
            "title"  : "Personalized Study Plan Generator",
            "problem": "Generate a custom weekly study schedule based on each student's strengths and weaknesses.",
            "ml_type": "Recommendation System + Regression",
            "input"  : "Subject-wise test scores, time available per day, learning style, upcoming exam dates",
            "output" : "Day-wise study plan with topic priority + recommended resources (videos/notes/practice)",
            "impact" : "Students cover weak areas first, reducing last-minute cramming."
        },
        {
            "title"  : "Automated Answer Sheet Evaluation",
            "problem": "Manually checking descriptive answers is slow and inconsistent. ML can grade answers fairly and instantly.",
            "ml_type": "NLP — Semantic Similarity + Regression",
            "input"  : "Student's written answer, model answer from teacher, maximum marks",
            "output" : "Marks awarded (e.g. 7/10) + Feedback: 'Missing mention of X concept'",
            "impact" : "Faster results, consistent grading, and detailed feedback at scale."
        },
        {
            "title"  : "Campus Placement Predictor",
            "problem": "Predict whether a student will get placed and estimate the expected salary range.",
            "ml_type": "Classification + Regression (Supervised)",
            "input"  : "CGPA, internships, certifications, coding rankings, communication score, technical skills",
            "output" : "Placement Probability (e.g. 82%) + Predicted Salary Range (e.g. 5–8 LPA) + Skill gaps",
            "impact" : "Students and placement officers plan preparation strategy well in advance."
        },
        {
            "title"  : "Lecture Engagement Analyzer",
            "problem": "Measure real-time student engagement during online/offline classes and alert teachers.",
            "ml_type": "Computer Vision + Time Series Analysis",
            "input"  : "Webcam video frames, LMS activity (mouse/keyboard), quiz response time",
            "output" : "Engagement Score (0–100%) per student + Attention dip timeline + Alert if avg < 40%",
            "impact" : "Teachers adjust pace or add activities when engagement drops."
        },
    ],

    "HEALTHCARE": [
        {
            "title"  : "Disease Risk Prediction",
            "problem": "Predict the likelihood of a patient developing diabetes, heart disease, or hypertension before symptoms appear.",
            "ml_type": "Binary / Multi-class Classification",
            "input"  : "Age, gender, BMI, blood pressure, blood glucose, cholesterol, family history, lifestyle habits",
            "output" : "Risk Level = High / Medium / Low + Predicted condition + Recommended lifestyle changes",
            "impact" : "Early intervention saves lives and reduces long-term treatment costs."
        },
        {
            "title"  : "Medical Image Diagnosis (X-Ray / MRI)",
            "problem": "Assist radiologists by instantly flagging abnormalities in X-rays, MRIs, and CT scans.",
            "ml_type": "CNN — Image Classification + Object Detection",
            "input"  : "X-Ray or MRI scan image (DICOM/JPEG), patient age, known symptoms",
            "output" : "Normal / Abnormal + Abnormality location highlighted + Diagnosis (e.g. 'Pneumonia: 91%')",
            "impact" : "Faster diagnosis in rural hospitals where specialist radiologists are scarce."
        },
        {
            "title"  : "Hospital Readmission Predictor",
            "problem": "Predict readmission risk before a patient is discharged to prevent early returns.",
            "ml_type": "Binary Classification (Supervised Learning)",
            "input"  : "Diagnosis, length of stay, previous admissions, medications, age, discharge condition score",
            "output" : "Readmission Risk = High / Low + Recommended post-discharge care plan",
            "impact" : "Hospitals reduce costly readmissions; patients get better follow-up care."
        },
        {
            "title"  : "Medicine Demand Forecasting",
            "problem": "Predict how much medicine stock is needed each week/month to prevent shortages and wastage.",
            "ml_type": "Time Series Forecasting (LSTM / ARIMA)",
            "input"  : "Historical consumption data, seasonal disease patterns, population, disease outbreaks",
            "output" : "Predicted units needed per medicine per period + Reorder alert when stock is low",
            "impact" : "Zero stockouts of critical medicines and reduced wastage of expensive drugs."
        },
        {
            "title"  : "Mental Health Sentiment Tracker",
            "problem": "Analyze patient text/speech to detect early signs of depression or anxiety before crises occur.",
            "ml_type": "NLP — Sentiment Analysis + Classification",
            "input"  : "Patient's journal entries (text), questionnaire responses, therapy session speech recordings",
            "output" : "Status = Stable / At Risk / Critical + Emotion breakdown + Alert to therapist if critical",
            "impact" : "Therapists prioritize patients who need urgent attention, preventing mental health crises."
        },
    ],

    "SHOPPING / E-COMMERCE": [
        {
            "title"  : "Product Recommendation Engine",
            "problem": "Show customers products they are most likely to buy, increasing sales and reducing search time.",
            "ml_type": "Collaborative Filtering + Deep Learning",
            "input"  : "Browsing history, past purchases, wishlist/cart items, ratings, demographic info, time of day",
            "output" : "Top 10 personalized recommendations + 'Frequently bought together' + 'Others also bought'",
            "impact" : "Amazon attributes ~35% of its revenue to its recommendation engine alone."
        },
        {
            "title"  : "Dynamic Pricing Optimizer",
            "problem": "Automatically adjust product prices based on demand, competition, and user behavior.",
            "ml_type": "Reinforcement Learning + Regression",
            "input"  : "Stock level, competitor prices, demand trend (7/30 days), time of day, festival/sale flags",
            "output" : "Optimal price at that moment + Expected revenue impact of the change",
            "impact" : "Maximises profit during peak demand; stays competitive during slow periods."
        },
        {
            "title"  : "Fake Review Detector",
            "problem": "Detect AI-generated or incentivised reviews automatically before they go live.",
            "ml_type": "NLP — Binary Text Classification",
            "input"  : "Review text, reviewer account age, reviews posted by same user, verified purchase flag, rating",
            "output" : "FAKE / GENUINE + Confidence score (e.g. '87% likely fake') + Reason for flagging",
            "impact" : "Builds buyer trust and protects authentic sellers on the platform."
        },
        {
            "title"  : "Customer Churn Prediction",
            "problem": "Predict which customers are about to stop shopping so targeted retention offers can be made.",
            "ml_type": "Binary Classification (Supervised Learning)",
            "input"  : "Days since last purchase, average order value, lifetime spend, returns, support tickets, app usage",
            "output" : "Churn Probability (e.g. 73%) + Segment (Loyal/At Risk/Lost) + Recommended retention offer",
            "impact" : "Targeted campaigns reduce churn, saving significant customer acquisition cost."
        },
        {
            "title"  : "Visual Search — Shop by Image",
            "problem": "Let customers upload a photo of any product they like and instantly find it in the store.",
            "ml_type": "CNN + Image Similarity (Siamese Network)",
            "input"  : "Photo uploaded by customer (any angle) + optional filters (price range, color, brand)",
            "output" : "Top 5 visually similar products + Match % for each + Direct 'Add to Cart' links",
            "impact" : "Reduces search friction, increases impulse purchases, improves customer experience."
        },
    ],
}

# ─────────────────────────────────────────────
# PRINT FUNCTION
# ─────────────────────────────────────────────

def print_ideas(ideas):
    print("\n" + "=" * 62)
    print("              ML IDEA GENERATOR")
    print("    College | Healthcare | Shopping — Input -> Output")
    print("=" * 62)

    idea_number = 1
    for domain, domain_ideas in ideas.items():
        print(f"\n{'#' * 62}")
        print(f"  DOMAIN : {domain}")
        print(f"{'#' * 62}")

        for idea in domain_ideas:
            print(f"\n  [{idea_number:02d}] {idea['title']}")
            print(f"  {'─' * 56}")
            print(f"  Problem  : {idea['problem']}")
            print(f"  ML Type  : {idea['ml_type']}")
            print(f"  Input    : {idea['input']}")
            print(f"  Output   : {idea['output']}")
            print(f"  Impact   : {idea['impact']}")
            idea_number += 1

    print("\n" + "=" * 62)
    print("  Summary : 15 ideas across 3 domains")
    print("  ML Types: Classification, NLP, CNN, Recommendation,")
    print("            Reinforcement Learning, Time Series")
    print("=" * 62)

# ─────────────────────────────────────────────
# WRITE TEXT FILE FUNCTION
# ─────────────────────────────────────────────

def write_text_file(ideas, save_dir):
    lines = []
    lines.append("=" * 62)
    lines.append("          ML IDEA GENERATOR — DATA FILE")
    lines.append("  College | Healthcare | Shopping — Input -> Output")
    lines.append("=" * 62 + "\n")

    idea_number = 1
    for domain, domain_ideas in ideas.items():
        lines.append("#" * 62)
        lines.append(f"  DOMAIN : {domain}")
        lines.append("#" * 62)

        for idea in domain_ideas:
            lines.append(f"\n  [{idea_number:02d}] {idea['title']}")
            lines.append(f"  {'─' * 56}")
            lines.append(f"  Problem  : {idea['problem']}")
            lines.append(f"  ML Type  : {idea['ml_type']}")
            lines.append(f"  Input    : {idea['input']}")
            lines.append(f"  Output   : {idea['output']}")
            lines.append(f"  Impact   : {idea['impact']}")
            idea_number += 1
        lines.append("")

    lines.append("=" * 62)
    lines.append("  Total Ideas : 15  (5 per domain)")
    lines.append("  ML Types    : Classification, Regression, NLP, CNN,")
    lines.append("                Recommendation, Reinforcement Learning,")
    lines.append("                Time Series Forecasting")
    lines.append("=" * 62)

    txt_path = os.path.join(save_dir, "ml_ideas.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Text file saved -> {txt_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print_ideas(ideas)
    write_text_file(ideas, SAVE_DIR)
    print(f"  Folder        -> {SAVE_DIR}\n")

if __name__ == "__main__":
    main()