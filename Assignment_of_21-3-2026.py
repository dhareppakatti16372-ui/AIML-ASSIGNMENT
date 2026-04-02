import string

# Sample text
text = "This is an Example sentence, showing how to CLEAN text! It is very useful."

# Define stopwords
stopwords = {
    "is", "an", "the", "to", "it", "this", "how", "very", "a", "of", "and", "in", "on"
}

# Text cleaning function
def clean_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenize
    words = text.split()
    
    # 4. Remove stopwords
    filtered_words = [word for word in words if word not in stopwords]
    
    # 5. Join words
    cleaned_text = " ".join(filtered_words)
    
    return cleaned_text

# Test the function
cleaned = clean_text(text)

print("Original Text:\n", text)
print("\nCleaned Text:\n", cleaned)