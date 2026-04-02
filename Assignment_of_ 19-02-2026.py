# ============================================
#         STUDENT DATA MANAGER
# ============================================

def assign_grade(marks):
    """Assigns a grade based on marks."""
    if marks >= 90:
        return "A+"
    elif marks >= 80:
        return "A"
    elif marks >= 70:
        return "B"
    elif marks >= 60:
        return "C"
    elif marks >= 50:
        return "D"
    else:
        return "F"

def get_grade_remark(grade):
    """Returns a remark for each grade."""
    remarks = {
        "A+": "Outstanding! 🌟",
        "A" : "Excellent!   🎉",
        "B" : "Good Work!   👍",
        "C" : "Average.     📘",
        "D" : "Below Avg.   ⚠️",
        "F" : "Needs Help.  ❌"
    }
    return remarks[grade]

def input_students(count):
    """Takes input for 'count' students and returns a list of dicts."""
    students = []
    print("\n" + "=" * 50)
    print("         ENTER STUDENT DETAILS")
    print("=" * 50)

    for i in range(1, count + 1):
        print(f"\n  Student {i}:")
        name  = input("    Name    : ").strip()
        roll  = input("    Roll No : ").strip()
        marks = int(input("    Marks   : ").strip())

        grade = assign_grade(marks)
        students.append({
            "name" : name,
            "roll" : roll,
            "marks": marks,
            "grade": grade
        })

    return students

def print_report(students):
    """Prints the full student report."""
    print("\n" + "=" * 60)
    print("                  STUDENT REPORT CARD")
    print("=" * 60)
    print(f"  {'Roll':<6}  {'Name':<18}  {'Marks':>5}  {'Grade':>5}  Remark")
    print("-" * 60)

    for s in students:
        remark = get_grade_remark(s["grade"])
        print(f"  {s['roll']:<6}  {s['name']:<18}  {s['marks']:>5}  {s['grade']:>5}  {remark}")

    print("=" * 60)

def print_analytics(students):
    """Prints topper, lowest scorer, and class average."""
    total      = sum(s["marks"] for s in students)
    average    = total / len(students)
    topper     = max(students, key=lambda s: s["marks"])
    lowest     = min(students, key=lambda s: s["marks"])

    print("\n" + "=" * 50)
    print("            CLASS ANALYTICS")
    print("=" * 50)
    print(f"  📊 Class Average  : {average:.2f} marks")
    print(f"  🏆 Topper         : {topper['name']} ({topper['marks']} marks, Grade {topper['grade']})")
    print(f"  📉 Lowest Scorer  : {lowest['name']} ({lowest['marks']} marks, Grade {lowest['grade']})")

    passed  = [s for s in students if s["marks"] >= 50]
    failed  = [s for s in students if s["marks"] <  50]
    print(f"  ✅ Passed         : {len(passed)} student(s)")
    print(f"  ❌ Failed         : {len(failed)} student(s)")
    print("=" * 50)

def main():
    print("\n" + "=" * 50)
    print("       WELCOME TO STUDENT DATA MANAGER")
    print("=" * 50)

    students = input_students(5)
    print_report(students)
    print_analytics(students)

if __name__ == "__main__":
    main()