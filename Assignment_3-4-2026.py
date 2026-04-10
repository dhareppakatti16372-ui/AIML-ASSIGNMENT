import string

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Chatbot response function
def chatbot_response(user_input):
    text = clean_text(user_input)
    
    if "hello" in text or "hi" in text:
        return "Hello! How can I help you?"
    
    elif "how are you" in text:
        return "I'm just a bot, but I'm doing great!"
    
    elif "your name" in text:
        return "I am a simple NLP chatbot."
    
    elif "bye" in text or "exit" in text:
        return "Goodbye! Have a nice day!"
    
    elif "help" in text:
        return "I can answer simple questions. Try asking something!"
    
    else:
        return "Sorry, I don't understand that."

# Chat loop
print("---- Simple Chatbot ----")
print("Type 'exit' to stop\n")

while True:
    user_input = input("You: ")
    
    response = chatbot_response(user_input)
    print("Bot:", response)
    
    if "exit" in user_input.lower():
        break