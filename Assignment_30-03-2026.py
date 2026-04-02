# Install: pip install openai

from openai import OpenAI

# Initialize client
client = OpenAI(api_key="YOUR_API_KEY")

# Prompts dictionary
prompts = {
    "Resume": [
        "Write a resume",
        "Write a professional resume for a Computer Science student with skills in Java, Python, and Data Structures, including projects and internships in 1 page format."
    ],
    "Business Idea": [
        "Give me a business idea",
        "Suggest a low-investment startup idea in the food industry suitable for college students in India, including cost, target customers, and profit potential."
    ],
    "Study Plan": [
        "Make a study plan",
        "Create a 30-day study plan to learn Python for beginners, covering basics, practice problems, and mini projects with daily time allocation."
    ]
}

print("---- Prompt Engineering Comparison ----\n")

# Loop through prompts
for category, (weak, strong) in prompts.items():
    
    print(f"Category: {category}\n")
    
    # Weak Prompt
    weak_response = client.responses.create(
        model="gpt-4.1-mini",
        input=weak
    )
    
    # Strong Prompt
    strong_response = client.responses.create(
        model="gpt-4.1-mini",
        input=strong
    )
    
    print("Weak Prompt:", weak)
    print("Response:", weak_response.output[0].content[0].text)
    
    print("\nStrong Prompt:", strong)
    print("Response:", strong_response.output[0].content[0].text)
    
    print("\n" + "="*60 + "\n")