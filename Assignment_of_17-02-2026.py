# ============================================
#           LOGIC BUILDER — FIZZBUZZ
# ============================================

def classify(n):
    """Returns FizzBuzz classification for a number."""
    if n % 15 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    else:
        return str(n)

def print_sequence(start, end):
    """Prints the FizzBuzz sequence and returns counts."""
    counts = {"Fizz": 0, "Buzz": 0, "FizzBuzz": 0, "Number": 0}

    print("=" * 45)
    print("         FIZZBUZZ SEQUENCE (1 – 50)")
    print("=" * 45)

    row = []
    for n in range(start, end + 1):
        label = classify(n)

        # Update counts
        if label == "Fizz":
            counts["Fizz"] += 1
        elif label == "Buzz":
            counts["Buzz"] += 1
        elif label == "FizzBuzz":
            counts["FizzBuzz"] += 1
        else:
            counts["Number"] += 1

        # Format each item neatly
        row.append(f"{label:>8}")

        # Print 5 items per row
        if n % 5 == 0:
            print("  ".join(row))
            row = []

    return counts

def print_summary(counts):
    """Prints the occurrence summary."""
    print("=" * 45)
    print("            OCCURRENCE SUMMARY")
    print("=" * 45)
    print(f"  🔵 Fizz     (divisible by 3)     : {counts['Fizz']:>2} times")
    print(f"  🟡 Buzz     (divisible by 5)     : {counts['Buzz']:>2} times")
    print(f"  🟢 FizzBuzz (divisible by 3 & 5) : {counts['FizzBuzz']:>2} times")
    print(f"  ⚪ Numbers  (no rule matched)    : {counts['Number']:>2} times")
    print("-" * 45)
    print(f"  📊 Total                         : {sum(counts.values()):>2} items")
    print("=" * 45)

def main():
    print("\n" + "=" * 45)
    print("        WELCOME TO LOGIC BUILDER")
    print("=" * 45 + "\n")

    counts = print_sequence(1, 50)
    print()
    print_summary(counts)

if __name__ == "__main__":
    main()