# ============================================
#        SMART INPUT PROGRAM
# ============================================

def categorize_age(age):
    """Returns an age category based on the given age."""
    if age <= 12:
        return "Child"
    elif age <= 17:
        return "Teenager"
    elif age <= 25:
        return "Young Adult"
    elif age <= 59:
        return "Adult"
    else:
        return "Senior"

def get_age_message(category):
    """Returns a personalized message based on age category."""
    messages = {
        "Child":       "Keep exploring, learning, and having fun!",
        "Teenager":    "These are your years to grow — dream big!",
        "Young Adult": "The world is full of possibilities for you!",
        "Adult":       "Keep achieving great things every day!",
        "Senior":      "Your wisdom and experience are truly inspiring!"
    }
    return messages[category]

def main():
    print("=" * 45)
    print("        WELCOME TO SMART INPUT PROGRAM")
    print("=" * 45)

    # --- Take Inputs ---
    name  = input("\nEnter your name  : ").strip()
    age   = int(input("Enter your age   : ").strip())
    hobby = input("Enter your hobby : ").strip()

    # --- Process ---
    category    = categorize_age(age)
    age_message = get_age_message(category)

    # --- Output ---
    print("\n" + "=" * 45)
    print("        YOUR PERSONALIZED MESSAGE")
    print("=" * 45)
    print(f"\nHello, {name}! 👋")
    print(f"You are {age} years old, which makes you a: {category}")
    print(f"\n💬 {age_message}")
    print(f"\n🎯 It's wonderful that you enjoy {hobby}!")
    print(f"   Keep pursuing your passion for {hobby} —")
    print(f"   it makes you uniquely YOU!")
    print("\n" + "=" * 45)

if __name__ == "__main__":
    main()