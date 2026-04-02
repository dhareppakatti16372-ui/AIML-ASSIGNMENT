from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
train_reviews = [
    "I love this movie",
    "This film was fantastic",
    "Amazing acting and great story",
    "I hate this movie",
    "This film was terrible",
    "Worst movie ever"
]

# Labels (1 = Positive, 0 = Negative)
train_labels = [1, 1, 1, 0, 0, 0]

# Test data (5 reviews)
test_reviews = [
    "This movie was amazing and full of suspense",
    "I really hated this film, it was boring",
    "What a fantastic performance by the actors",
    "The plot was dull and predictable",
    "An excellent movie with great visuals"
]

# Convert text to numerical data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_reviews)
X_test = vectorizer.transform(test_reviews)

# Train model
model = MultinomialNB()
model.fit(X_train, train_labels)

# Predict sentiments
predictions = model.predict(X_test)

# Display results
print("---- Sentiment Analysis Results ----\n")

for review, pred in zip(test_reviews, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}\n")