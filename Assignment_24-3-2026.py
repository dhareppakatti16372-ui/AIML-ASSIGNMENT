from sklearn.feature_extraction.text import TfidfVectorizer

# 5 documents
documents = [
    "Machine learning is very useful for data analysis",
    "Data science and machine learning are closely related",
    "Artificial intelligence and deep learning are trending topics",
    "Data analysis requires statistics and machine learning",
    "Deep learning is a part of artificial intelligence"
]

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert to array
tfidf_array = tfidf_matrix.toarray()

# Extract top 3 keywords for each document
print("---- Top Keywords per Document ----\n")

for i, doc in enumerate(tfidf_array):
    print(f"Document {i+1}:")
    
    # Get top 3 word indices
    top_indices = doc.argsort()[-3:][::-1]
    
    for idx in top_indices:
        print(f"{feature_names[idx]} : {doc[idx]:.3f}")
    
    print()